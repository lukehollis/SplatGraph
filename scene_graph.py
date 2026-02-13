import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import json
import cv2
from sklearn.cluster import DBSCAN
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# Add LangSplatV2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../LangSplatV2"))

from scene import Scene
from gaussian_renderer import GaussianModel, render
import gaussian_renderer

# Force patch GaussianRasterizer in gaussian_renderer ONLY if we are mocking
if not torch.cuda.is_available() and isinstance(sys.modules.get("diff_gaussian_rasterization"), MagicMock):
    print(f"DEBUG: gaussian_renderer.GaussianRasterizer is {gaussian_renderer.GaussianRasterizer}")
    gaussian_renderer.GaussianRasterizer = rasterizer_class
    print(f"DEBUG: Patched gaussian_renderer.GaussianRasterizer to {gaussian_renderer.GaussianRasterizer}")

from arguments import ModelParams, PipelineParams
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SplatSceneGraph:
    def __init__(self, model_path, dataset_path, output_dir):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.gaussians = None
        self.scene = None
        self.views = []
        self.objects = []  # List of unique objects
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize SAM
        self.sam_checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), "../LangSplatV2/ckpts/sam_vit_h_4b8939.pth"))
        print(f"DEBUG: Checking SAM checkpoint at {self.sam_checkpoint}")
        if not os.path.exists(self.sam_checkpoint):
             print(f"Warning: SAM checkpoint not found at {self.sam_checkpoint}")
        
        self.mask_generator = None

    def load_model(self, iteration=30000, level=3, max_train_views=300):
        """
        Load LangSplatV2 model. 
        For multi-level models (in langsplat_output), level should be 1, 2, or 3.
        Level 3 contains the full hierarchical features.
        """
        print(f"Loading model from {self.model_path} (level {level}) at iteration {iteration}")
        
        # Mock arguments expected by LangSplatV2
        class Args:
            pass
        
        args = Args()
        args.sh_degree = 3
        args.source_path = self.dataset_path
        args.model_path = self.model_path
        args.images = "images"
        args.resolution = -1
        args.white_background = False
        args.data_device = "cpu"
        args.eval = True
        args.include_feature = True
        args.quick_render = False
        args.topk = 1
        args.ply_path = os.path.join(self.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        self.args = args # Save for render
        
        self.gaussians = GaussianModel(args.sh_degree)
        self.scene = Scene(args, self.gaussians, load_iteration=iteration, shuffle=False, max_train_views=max_train_views)
        
        # Load checkpoint for language features
        checkpoint_path = os.path.join(self.model_path, f"chkpnt{iteration}.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            (model_params, first_iter) = torch.load(checkpoint_path, map_location=torch.device(self.device))
            self.gaussians.restore(model_params, args, mode='test')
            
            # Verify language features were loaded
            if self.gaussians._language_feature_codebooks is None:
                raise ValueError(f"Checkpoint at {checkpoint_path} does not contain language features. "
                                 "Make sure you're using a LangSplatV2 trained model (e.g., from langsplat_output).")
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        self.views = self.scene.getTrainCameras()
        
        # Initialize SAM
        # Note: We might need to download the checkpoint if not present
        if os.path.exists(self.sam_checkpoint):
            sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
        else:
            print(f"Warning: SAM checkpoint not found at {self.sam_checkpoint}")

    def render_language_feature_map(self, view, pipeline, background):
        # Adapted from eval_3d_ovs.py
        with torch.no_grad():
            output = render(view, self.gaussians, pipeline, background, self.args)
            language_feature_weight_map = output['language_feature_weight_map']
            language_feature_map = self.gaussians.compute_final_feature_map(language_feature_weight_map)
            
            # Normalize
            language_feature_map = language_feature_map / (language_feature_map.norm(dim=0, keepdim=True) + 1e-10)
            
        return output['render'], language_feature_map

    def segment_scene(self, skip_frames=10, cluster_eps=0.1):
        print("Segmenting scene with Depth & Opacity Filtering...")
        import argparse
        parser = argparse.ArgumentParser()
        pipeline = PipelineParams(parser) 
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        
        all_detections = [] 
        crops_dir = os.path.join(self.output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        
        # 1. Pre-fetch Data
        means3D = self.gaussians.get_xyz
        ones = torch.ones((means3D.shape[0], 1), device=self.device)
        means4D = torch.cat((means3D, ones), dim=1)

        # 2. Global Opacity Filter (Ignore invisible "floaters")
        # Check for get_opacity (LangSplat) or _opacity (Standard)
        if hasattr(self.gaussians, "get_opacity"):
            opacities = self.gaussians.get_opacity
        else:
            opacities = torch.sigmoid(self.gaussians._opacity)
        
        # Ignore points with < 0.1 opacity
        is_visible_global = (opacities.squeeze() > 0.1)

        for i, view in enumerate(tqdm(self.views[::skip_frames])):
            try:
                rgb, lang_map = self.render_language_feature_map(view, pipeline, background)
                
                # Project Gaussians
                P = view.full_proj_transform
                p_hom = means4D @ P
                p_w = 1.0 / (p_hom[:, 3] + 1e-7)
                p_proj = p_hom[:, :3] * p_w.unsqueeze(1)
                
                u_ndc = p_proj[:, 0]
                v_ndc = p_proj[:, 1]
                H, W = view.image_height, view.image_width
                u_pix = ((u_ndc + 1.0) * W / 2.0)
                v_pix = ((v_ndc + 1.0) * H / 2.0)
                
                # Bounds check + Opacity check
                in_bounds = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H)
                valid_mask = in_bounds & is_visible_global
                valid_indices = torch.where(valid_mask)[0]
                
                valid_u = u_pix[valid_indices].long()
                valid_v = v_pix[valid_indices].long()
                
                # Get Depth (W component is view-space depth)
                valid_depths = p_hom[valid_indices, 3]

                # Run SAM
                rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                if self.mask_generator:
                    masks = self.mask_generator.generate(rgb_np)
                else:
                    continue

                lang_map_np = lang_map.permute(1, 2, 0).cpu().numpy()

                for j, mask_data in enumerate(masks):
                    # Handle case where SAM returns a list of lists (batch mode artifact)
                    if isinstance(mask_data, list) and len(mask_data) > 0:
                        mask_data = mask_data[0]
                        
                    if not isinstance(mask_data, dict):
                        continue
                        
                    mask = mask_data['segmentation']
                    if mask.sum() == 0: continue

                    # Identify points in this 2D mask
                    points_in_mask_bool = mask[valid_v.cpu().numpy(), valid_u.cpu().numpy()]
                    indices_in_mask = valid_indices[torch.tensor(points_in_mask_bool, device=self.device)]
                    
                    centroid = None
                    filtered_indices_np = []
                    
                    if len(indices_in_mask) > 0:
                        # --- CRITICAL FIX: DEPTH FILTERING ---
                        # 1. Get depths of points in this mask
                        z_vals = p_hom[indices_in_mask, 3]
                        
                        # 2. Find the "front" surface (5th percentile to ignore outliers)
                        surface_depth = torch.quantile(z_vals, 0.05)
                        
                        # 3. Filter: Keep only points close to the surface (e.g., within 10%)
                        # This discards the wall/floor behind the object
                        depth_threshold = surface_depth * 1.10 
                        foreground_mask = z_vals <= depth_threshold
                        
                        final_indices = indices_in_mask[foreground_mask]
                        
                        if len(final_indices) > 0:
                            # Use MEDIAN for robust centroid
                            centroid_med = means3D[final_indices].median(dim=0).values
                            centroid = centroid_med.cpu().tolist()
                            filtered_indices_np = final_indices.cpu().numpy()
                    
                    # If we filtered everything out, skip
                    if len(filtered_indices_np) == 0:
                        continue

                    # Extract Feature
                    mask_bool = mask.astype(bool)
                    feature = lang_map_np[mask_bool].mean(axis=0)

                    # Save Crop
                    x, y, w, h = mask_data['bbox']
                    crop_filename = f"view_{i}_obj_{j}.png"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    # (Optional: check existence to speed up)
                    Image.fromarray(rgb_np[int(y):int(y+h), int(x):int(x+w)]).save(crop_path)
                    
                    all_detections.append({
                        'feature': feature,
                        'crop_path': crop_path,
                        'view_idx': i,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': mask_data['area'],
                        'centroid': centroid,
                        'point_indices': filtered_indices_np # Store indices directly!
                    })
                
                del rgb, lang_map, rgb_np, valid_depths
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error on view {i}: {e}")
                torch.cuda.empty_cache()
                continue

        self.xyz = self.gaussians.get_xyz.detach().cpu().numpy()
        self.cluster_objects(all_detections, eps=cluster_eps)
        self.assign_points_and_compute_obbs()

    def cluster_objects(self, detections, eps=0.1, min_samples=3):
        print(f"Clustering {len(detections)} detections...")
        if not detections: return

        features = np.array([d['feature'] for d in detections])
        
        # Clustering Logic
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}
        
        print(f"Found {len(unique_labels)} unique objects.")
        
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            obj_detections = [detections[i] for i in indices]
            
            # --- IMPROVED CROP SELECTION ---
            # Instead of max(area), pick the detection closest to the cluster center (Medoid)
            # This avoids picking large, ambiguous "leak" masks that might have huge area.
            
            # 1. Compute Mean Feature of this cluster
            cluster_features = features[indices]
            mean_feature = np.mean(cluster_features, axis=0)
            
            # 2. Find detection with minimal Euclidean distance to mean
            best_dist = float('inf')
            best_detection = obj_detections[0]
            
            for d in obj_detections:
                dist = np.linalg.norm(d['feature'] - mean_feature)
                # Bias slightly towards larger crops to avoid tiny fragments, 
                # but primary factor is feature representativeness.
                # Heuristic: penalize extremely small crops
                if d['area'] < 1000: dist *= 2.0 
                
                if dist < best_dist:
                    best_dist = dist
                    best_detection = d

            # --- MERGE POINT INDICES ---
            # Combine indices from all views to get the full 3D object
            all_idx_arrays = [d['point_indices'] for d in obj_detections]
            if all_idx_arrays:
                merged_indices = np.unique(np.concatenate(all_idx_arrays))
            else:
                merged_indices = np.array([], dtype=np.int32)
            
            # Compute Mean Centroid
            centroids = [d['centroid'] for d in obj_detections if d['centroid'] is not None]
            mean_centroid = np.mean(centroids, axis=0).tolist() if centroids else None

            self.objects.append({
                'id': int(label),
                'feature': best_detection['feature'].tolist(),
                'best_crop_path': best_detection['crop_path'],
                'best_view_idx': best_detection['view_idx'],
                'bbox': best_detection['bbox'],
                'area': best_detection['area'],
                'centroid': mean_centroid,
                'point_indices': merged_indices, # Save merged indices
                'children': []
            })


    def assign_points_and_compute_obbs(self):
        print("Computing 3D OBBs from segmented surface points...", flush=True)
        
        obb_count = 0
        for obj in self.objects:
            # 1. Retrieve the clean indices we stored during segmentation
            indices = obj.get('point_indices')
            
            if indices is None or len(indices) < 4:
                continue
                
            points_np = self.xyz[indices]
            
            # 2. Compute Robust Gravity-Aligned OBB
            obb_corners = self.compute_obb(points_np)
            
            if obb_corners is not None:
                obj['obb'] = obb_corners
                obb_count += 1
            
            # Persist point indices for precise export
            if indices is not None:
                if hasattr(indices, 'tolist'):
                    obj['point_indices'] = indices.tolist()
                else:
                    obj['point_indices'] = list(indices)

    def compute_obb(self, points):
        """
        Compute a Gravity-Aligned Bounding Box (assuming Y is UP).
        This prevents 'tilted' boxes for objects sitting on tables.
        """
        if len(points) < 4: return None
            
        # 1. Statistical Outlier Removal (Cleanup "spider webs")
        try:
            from sklearn.neighbors import LocalOutlierFactor
            # Fast outlier detection
            clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
            y_pred = clf.fit_predict(points)
            if (y_pred == 1).sum() > 4:
                points = points[y_pred == 1]
        except:
            pass # Fallback

        # 2. Compute 2D Footprint on Floor Plane (XZ)
        # Assuming Y (index 1) is UP. Change indices if Z is UP.
        points_xz = points[:, [0, 2]] 
        
        # PCA on 2D footprint
        center_xz = points_xz.mean(axis=0)
        centered_xz = points_xz - center_xz
        cov_xz = np.cov(centered_xz, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_xz)
        
        # Sort eigenvectors
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        
        # Project points to 2D PCA axes
        proj_xz = centered_xz @ evecs
        min_xz = np.percentile(proj_xz, 2, axis=0) # Robust Min (2nd percentile)
        max_xz = np.percentile(proj_xz, 98, axis=0) # Robust Max (98th percentile)
        
        # 3. Compute Vertical Extent (Y)
        y_vals = points[:, 1]
        y_min = np.percentile(y_vals, 2)
        y_max = np.percentile(y_vals, 98)
        
        # 4. Reconstruct 3D Corners
        corners = []
        # Axis U = [evecs[0,0], 0, evecs[1,0]]
        # Axis V = [evecs[0,1], 0, evecs[1,1]]
        
        for x in [min_xz[0], max_xz[0]]:
            for z in [min_xz[1], max_xz[1]]:
                # Map back from 2D PCA space to World XZ
                pt_xz = np.dot(np.array([x, z]), evecs.T) + center_xz
                
                # Bottom cap
                corners.append([pt_xz[0], y_min, pt_xz[1]])
                # Top cap
                corners.append([pt_xz[0], y_max, pt_xz[1]])
                
        return corners

    def load_graph(self, filename="scene_graph_buildings.json"):
        """
        Loads an existing scene graph JSON file.
        """
        graph_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(graph_path):
             # Try without _buildings suffix
             graph_path = os.path.join(self.output_dir, "scene_graph.json")
             
        if os.path.exists(graph_path):
            print(f"Loading scene graph from {graph_path}")
            with open(graph_path, 'r') as f:
                data = json.load(f)
                self.objects = data.get('objects', [])
            print(f"Loaded {len(self.objects)} objects.")
        else:
            print(f"Warning: Graph file not found at {graph_path}")

    def analyze_semantics(self):
        """
        Analyzes objects to identify type (Building, Tree, Car, etc.) and properties.
        """
        print("Analyzing semantics (Buildings, Trees, Cars)...", flush=True)
        from vlm_utils import analyze_building_crop
        
        for obj in tqdm(self.objects):
            # Lower threshold to 500 to catch cars/trees
            if obj.get('area', 0) < 500: 
                continue

            # 1. Geometric Analysis
            obb = obj.get('obb')
            height = 0
            if obb:
                ys = [p[1] for p in obb]
                height = max(ys) - min(ys)
                
            obj['height'] = height
            
            # Simple heuristic: 3m per story (only relevant if building)
            estimated_stories_geo = int(height / 3.0)
            
            # 2. Semantic Analysis (VLM)
            crop_path = obj.get('best_crop_path')
            # path correction logic
            if crop_path and not os.path.isabs(crop_path):
                 crop_path = os.path.join(self.output_dir, os.path.basename(crop_path))
            
            if crop_path and not os.path.exists(crop_path):
                 crop_name = os.path.basename(crop_path)
                 possible_path = os.path.join(self.output_dir, "crops", crop_name)
                 if os.path.exists(possible_path):
                     crop_path = possible_path

            if crop_path and os.path.exists(crop_path):
                vlm_result = analyze_building_crop(crop_path)
                
                if vlm_result:
                    obj_type = vlm_result.get('object_type', 'Unknown')
                    # Fallback for old prompt format
                    if 'is_building' in vlm_result and 'object_type' not in vlm_result:
                         if vlm_result['is_building']: obj_type = "Building"
                    
                    obj['building_info'] = {
                        'object_type': obj_type,
                        'stories_visual': vlm_result.get('stories_visual', 0),
                        'stories_estimated_geo': estimated_stories_geo,
                        'usage': vlm_result.get('usage', 'Unknown'),
                        'description': vlm_result.get('description', '')
                    }
                    print(f" ID {obj['id']}: {obj_type} ({vlm_result.get('usage', '')})")
            else:
                 pass # No crop found

    def build_hierarchy(self, skip_frames=10):
        """
        [DEPRECATED] Builds a hierarchy based on 3D centroid projection.
        Current default is a flat hierarchy.
        
        If object A's 2D bbox in A's best view contains object B's projected 3D centroid,
        then B is a child of A.
        """
        print("Building scene graph hierarchy (3D-aware)...")
        sorted_objects = sorted(self.objects, key=lambda x: x['area'], reverse=True)
        assigned_children = set()
        root_objects = []
        
        for i, parent in enumerate(sorted_objects):
            if parent['id'] in assigned_children:
                continue
            
            px, py, pw, ph = parent['bbox']
            
            # Retrieve parent view
            # view_idx is the index in the subsampled list
            view_index_in_full_list = parent['best_view_idx'] * skip_frames
            if view_index_in_full_list >= len(self.views):
                # Fallback or error
                # print(f"Warning: View index {view_index_in_full_list} out of bounds.")
                pass 
            elif view_index_in_full_list < len(self.views):
                 pass
                
            # parent_view = self.views[view_index_in_full_list] # This was buggy if views list changed
            # P = parent_view.full_proj_transform
            # H, W = parent_view.image_height, parent_view.image_width
            
            # Skipping hierarchy logic for now as it was buggy and relied on exact view alignment
            
            root_objects.append(parent)
            
        self.objects = root_objects

    def save_graph(self, filename="scene_graph.json"):
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump({'objects': self.objects}, f, indent=4)
        print(f"Saved scene graph to {output_path}")

