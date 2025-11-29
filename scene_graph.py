import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import json
import cv2
from sklearn.cluster import DBSCAN
from PIL import Image

# Add LangSplatV2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../LangSplatV2"))

# Mock CUDA modules for macOS/CPU execution if CUDA is not available
if not torch.cuda.is_available():
    try:
        import simple_knn
    except ImportError:
        from unittest.mock import MagicMock
        
        # Mock simple_knn
        simple_knn = MagicMock()
        simple_knn_c = MagicMock()
        simple_knn.distCUDA2 = MagicMock()
        simple_knn_c.distCUDA2 = MagicMock()
        
        sys.modules["simple_knn"] = simple_knn
        sys.modules["simple_knn._C"] = simple_knn_c
        simple_knn._C = simple_knn_c
        
        # Mock diff_gaussian_rasterization
        diff_gaussian_rasterization = MagicMock()
        diff_gaussian_rasterization_c = MagicMock()
        
        # Configure GaussianRasterizer to return dummy tensors
        def mock_rasterize(*args, **kwargs):
            dummy_image = torch.zeros((3, 512, 512), dtype=torch.float32)
            dummy_radii = torch.zeros((100,), dtype=torch.float32)
            # D must match codebook size (64 based on error)
            dummy_lang = torch.zeros((64, 512, 512), dtype=torch.float32) 
            # render() expects: rendered_image, language_feature_weight_map, radii
            return dummy_image, dummy_lang, dummy_radii

        class MockRasterizerInstance:
            def __call__(self, *args, **kwargs):
                return mock_rasterize(*args, **kwargs)

        rasterizer_class = MagicMock()
        rasterizer_class.return_value = MockRasterizerInstance()
        
        diff_gaussian_rasterization.GaussianRasterizer = rasterizer_class
        diff_gaussian_rasterization.GaussianRasterizationSettings = MagicMock()
        
        sys.modules["diff_gaussian_rasterization"] = diff_gaussian_rasterization
        sys.modules["diff_gaussian_rasterization._C"] = diff_gaussian_rasterization_c
        diff_gaussian_rasterization._C = diff_gaussian_rasterization_c
        
        print("Mocked CUDA modules for macOS compatibility.")

    # Monkey-patch torch.Tensor.cuda to allow running on CPU
    def no_op_cuda(self, *args, **kwargs):
        return self

    torch.Tensor.cuda = no_op_cuda
    print("Monkey-patched torch.Tensor.cuda to run on CPU.")

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

    def load_model(self, iteration=30000, level=3):
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
        args.data_device = self.device
        args.eval = True
        args.include_feature = True
        args.quick_render = False
        args.topk = 1
        args.ply_path = os.path.join(self.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        self.args = args # Save for render
        
        self.gaussians = GaussianModel(args.sh_degree)
        self.scene = Scene(args, self.gaussians, load_iteration=iteration, shuffle=False)
        
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

    def segment_scene(self, skip_frames=10):
        print("Segmenting scene...")
        import argparse
        parser = argparse.ArgumentParser()
        pipeline = PipelineParams(parser) # Default pipeline params
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        
        all_detections = [] # List of {feature, crop_path, view_idx, mask}

        # Create crops directory
        crops_dir = os.path.join(self.output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        for i, view in enumerate(tqdm(self.views[::skip_frames])):
            try:
                rgb, lang_map = self.render_language_feature_map(view, pipeline, background)
                
                # Convert RGB to numpy for SAM
                rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rgb_np = np.clip(rgb_np, 0, 255)
                
                # Run SAM
                if self.mask_generator:
                    masks = self.mask_generator.generate(rgb_np)
                    if isinstance(masks, tuple):
                        # Handle SAM returning a tuple (likely (masks, scores, logits))
                        if len(masks) > 0 and isinstance(masks[0], list):
                            masks = masks[0]
                        elif len(masks) > 0 and isinstance(masks[0], dict):
                             pass
                else:
                    continue

                lang_map_np = lang_map.permute(1, 2, 0).cpu().numpy() # H, W, C

                for j, mask_data in enumerate(masks):
                    mask = mask_data['segmentation']
                    
                    # Extract average language feature
                    # Mask is boolean, lang_map is H,W,C
                    # We want mean feature over the mask
                    mask_bool = mask.astype(bool)
                    if mask_bool.sum() == 0:
                        continue
                        
                    feature = lang_map_np[mask_bool].mean(axis=0)
                    
                    # Save crop
                    x, y, w, h = mask_data['bbox']
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    crop = rgb_np[y:y+h, x:x+w]
                    
                    crop_filename = f"view_{i}_obj_{j}.png"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    Image.fromarray(crop).save(crop_path)
                    
                    all_detections.append({
                        'feature': feature,
                        'crop_path': crop_path,
                        'view_idx': i,
                        'bbox': [x, y, w, h],
                        'area': mask_data['area']
                    })
                
                # Free memory after each frame
                del rgb, lang_map, rgb_np, lang_map_np
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM on view {i}, skipping. Error: {e}")
                torch.cuda.empty_cache()
                continue

        self.cluster_objects(all_detections)

    def cluster_objects(self, detections, eps=0.1, min_samples=3):
        print(f"Clustering {len(detections)} detections...")
        if not detections:
            return

        features = np.array([d['feature'] for d in detections])
        
        # Use DBSCAN on features
        # Cosine distance might be better, but features are normalized so Euclidean is fine?
        # LangSplat features are CLIP embeddings, so cosine similarity is standard.
        # But DBSCAN uses Euclidean by default.
        # Normalized vectors: Euclidean distance is related to Cosine distance.
        # dist^2 = 2(1 - cos_sim)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        print(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} unique objects.")
        
        for label in unique_labels:
            if label == -1:
                continue
                
            indices = np.where(labels == label)[0]
            obj_detections = [detections[i] for i in indices]
            
            # Pick the best crop (e.g., largest area)
            best_detection = max(obj_detections, key=lambda x: x['area'])
            
            self.objects.append({
                'id': int(label),
                'feature': best_detection['feature'].tolist(),
                'best_crop_path': best_detection['crop_path'],
                'bbox': best_detection['bbox'], # [x, y, w, h]
                'area': best_detection['area'],
                'all_crops': [d['crop_path'] for d in obj_detections],
                'detection_count': len(obj_detections),
                'children': []
            })

    def build_hierarchy(self):
        """
        Builds a simple hierarchy based on 2D bounding box containment in the best view.
        This is a heuristic: if object A's bbox contains object B's bbox, B is a child of A.
        """
        print("Building scene graph hierarchy...")
        
        # Sort objects by area (largest first) to find parents first
        # We want to process from largest to smallest
        sorted_objects = sorted(self.objects, key=lambda x: x['area'], reverse=True)
        
        # Map IDs to objects for easy access
        id_to_obj = {obj['id']: obj for obj in self.objects}
        
        # Keep track of which objects are already assigned as children
        assigned_children = set()
        
        # Re-construct the list to be hierarchical (only top-level objects)
        root_objects = []
        
        for i, parent in enumerate(sorted_objects):
            if parent['id'] in assigned_children:
                continue
                
            # Check smaller objects to see if they are contained in this parent
            # We only check objects that haven't been assigned yet
            # And we only check objects that are strictly smaller
            
            px, py, pw, ph = parent['bbox']
            parent_rect = (px, py, px + pw, py + ph)
            
            for child in sorted_objects[i+1:]:
                if child['id'] in assigned_children:
                    continue
                
                cx, cy, cw, ch = child['bbox']
                child_rect = (cx, cy, cx + cw, cy + ch)
                
                # Check containment: child inside parent
                # Relaxed check: center of child inside parent
                child_center_x = cx + cw / 2
                child_center_y = cy + ch / 2
                
                if (px <= child_center_x <= px + pw) and (py <= child_center_y <= py + ph):
                    # It's a match!
                    parent['children'].append(child)
                    assigned_children.add(child['id'])
            
            root_objects.append(parent)
            
        self.objects = root_objects

    def save_graph(self, filename="scene_graph.json"):
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump({'objects': self.objects}, f, indent=4)
        print(f"Saved scene graph to {output_path}")

if __name__ == "__main__":
    # Test stub
    pass
