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
            # Pre-compute all per-Gaussian language features for downstream reassignment
            self.gaussian_features = self._compute_gaussian_features()
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        self.views = self.scene.getTrainCameras()
        
        # Initialize SAM
        # Download the checkpoint if not present
        if os.path.exists(self.sam_checkpoint):
            sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            # Tune SAM for higher granularity (more objects)
            self.mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.88
            )
        else:
            print(f"Warning: SAM checkpoint not found at {self.sam_checkpoint}")

    def _compute_gaussian_features(self):
        """Compute per-Gaussian language feature vectors from the LangSplat codebooks."""
        print("Computing per-Gaussian language features for global reassignment...")
        logits = self.gaussians._language_feature_logits
        codebooks = self.gaussians._language_feature_codebooks
        L, K, D = codebooks.shape
        logits = logits.to(self.device)
        codebooks = codebooks.to(self.device)
        weights_list = []
        for i in range(L):
            level_logits = logits[:, i * K:(i + 1) * K]
            weights_list.append(torch.softmax(level_logits, dim=1))
        weights = torch.cat(weights_list, dim=1)
        codebooks_flat = codebooks.view(-1, D)
        chunk_size = 50000
        N = weights.shape[0]
        features_list = []
        with torch.no_grad():
            for i in range(0, N, chunk_size):
                chunk_w = weights[i:i + chunk_size]
                chunk_f = chunk_w @ codebooks_flat
                chunk_f = chunk_f / (chunk_f.norm(dim=1, keepdim=True) + 1e-10)
                features_list.append(chunk_f.cpu())
        features = torch.cat(features_list, dim=0)  # (N, D)
        print(f"  -> Computed features for {N} Gaussians (dim={D}).")
        return features

    def reassign_points_by_feature(self, spatial_weight=1.5, threshold=0.3):
        """
        Feature-based reassignment within SAM-candidate Gaussians ONLY.

        SAM already depth-filtered the foreground. This function only reassigns
        points that were in at least one SAM mask — it does NOT sweep global
        background Gaussians. This prevents color bleed into empty space.

        Spatial normalization uses the MEDIAN inter-centroid distance (not max),
        so a single far-outlier object can't dilute the penalty for nearby objects.
        """
        if not hasattr(self, 'gaussian_features') or self.gaussian_features is None:
            print("Warning: gaussian_features not available, skipping reassignment.")
            return
        if not self.objects:
            return

        # --- 0. Remove extreme outlier objects before reassignment ---
        # Objects whose centroid is far from the scene median are likely noise.
        all_centroids = np.array([o['centroid'] if o['centroid'] is not None else [0.,0.,0.]
                                  for o in self.objects])
        scene_median = np.median(all_centroids, axis=0)
        centroid_dists_from_median = np.linalg.norm(all_centroids - scene_median, axis=1)
        scene_scale = np.median(centroid_dists_from_median)  # typical spread
        outlier_threshold = max(scene_scale * 5.0, 5.0)       # clip at 5x median or 5m
        valid_obj_mask = centroid_dists_from_median < outlier_threshold
        if not valid_obj_mask.all():
            n_outliers = (~valid_obj_mask).sum()
            for k, obj in enumerate(self.objects):
                if not valid_obj_mask[k]:
                    print(f"  Removing outlier object '{obj.get('metadata', {}).get('name', k)}' "
                          f"(centroid dist={centroid_dists_from_median[k]:.1f}m from scene)")
                    obj['point_indices'] = []

        # --- 1. Radial seed clip + build SAM candidate mask ---
        # BEFORE building the candidate pool, clip each object's SAM seeds to the
        # inner 70th percentile of distances from the object centroid.  This removes
        # the long tail of grazing-angle / projection-noise Gaussians that leaked
        # into the depth-filtered mask but are physically far from the object core.
        N = self.gaussian_features.shape[0]
        for k, obj in enumerate(self.objects):
            if not valid_obj_mask[k]:
                continue
            pts_idx = np.asarray(obj.get('point_indices', []), dtype=np.int64)
            if len(pts_idx) < 5:
                continue
            centroid_np = np.array(obj['centroid'] if obj['centroid'] is not None else [0.0]*3)
            pts_3d = self.xyz[pts_idx]
            dists = np.linalg.norm(pts_3d - centroid_np, axis=1)
            clip_r = np.percentile(dists, 70)  # keep inner 70%
            clip_r = max(clip_r, 0.05)          # never clip tighter than 5cm
            keep_mask = dists <= clip_r
            obj['point_indices'] = pts_idx[keep_mask].tolist()

        candidate_mask = np.zeros(N, dtype=bool)
        for k, obj in enumerate(self.objects):
            if not valid_obj_mask[k]:
                continue  # skip outliers
            pts = obj.get('point_indices', [])
            if len(pts):
                candidate_mask[np.asarray(pts, dtype=np.int64)] = True
        candidate_indices = np.where(candidate_mask)[0]
        print(f"Reassigning {len(candidate_indices)} SAM-candidate Gaussians after radial clip "
              f"(of {N} total; {N - len(candidate_indices)} background kept as-is)...")

        valid_objects = [o for k, o in enumerate(self.objects) if valid_obj_mask[k]]
        if not valid_objects:
            return

        obj_features = np.stack([np.array(o['feature']) for o in valid_objects])
        obj_features = obj_features / (np.linalg.norm(obj_features, axis=1, keepdims=True) + 1e-10)
        obj_centroids = np.array([o['centroid'] if o['centroid'] is not None else [0.0, 0.0, 0.0]
                                  for o in valid_objects])

        device = self.device
        # Only load candidate subset onto GPU
        g_feats_all = self.gaussian_features  # CPU tensor, full
        g_feats_cand = g_feats_all[candidate_indices].float().to(device)
        g_xyz_cand   = torch.from_numpy(self.xyz[candidate_indices]).float().to(device)
        o_feats = torch.from_numpy(obj_features).float().to(device)
        o_cents = torch.from_numpy(obj_centroids).float().to(device)

        chunk = 50000
        K = len(valid_objects)
        cand_assignments_local = np.full(len(candidate_indices), -1, dtype=np.int32)  # local index
        cand_best_sims   = np.zeros(len(candidate_indices), dtype=np.float32)

        # --- KEY FIX: Normalize spatial penalty by MEDIAN inter-centroid distance ---
        # The max would be dominated by any single outlier object far from the scene.
        # Median gives a robust, scene-scale normalization that makes the spatial
        # penalty meaningful for co-located objects.
        centroid_dists_matrix = torch.cdist(o_cents, o_cents)  # (K, K)
        if K > 1:
            # Extract upper triangular (pairwise distances)
            triu_mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
            pairwise = centroid_dists_matrix[triu_mask]
            median_centroid_dist = pairwise.median().clamp(min=0.05)
        else:
            median_centroid_dist = torch.tensor(1.0, device=device)
        print(f"  Spatial normalization: median inter-centroid dist = {median_centroid_dist.item():.3f}m")

        with torch.no_grad():
            for i in range(0, len(candidate_indices), chunk):
                gf = g_feats_cand[i:i + chunk]
                gx = g_xyz_cand[i:i + chunk]
                sim = gf @ o_feats.T                           # (B, K)
                dist = torch.cdist(gx, o_cents)                # (B, K)
                # Score = cosine similarity - spatial penalty
                # Dividing by median dist means dist=1*median_dist gives penalty=spatial_weight
                score = sim - (dist / median_centroid_dist) * spatial_weight
                best_idx = score.argmax(dim=1)
                best_sim = torch.gather(sim, 1, best_idx.unsqueeze(1)).squeeze(1)
                cand_assignments_local[i:i + chunk] = best_idx.cpu().numpy()
                cand_best_sims[i:i + chunk]   = best_sim.cpu().numpy()

        # Apply similarity threshold — low-confidence SAM points become background
        cand_assignments_local[cand_best_sims < threshold] = -1

        # --- 2. Map local valid-object indices back to global object indices ---
        valid_indices_global = [k for k, v in enumerate(valid_obj_mask) if v]
        full_assignments = np.full(N, -1, dtype=np.int32)
        for local_k, global_k in enumerate(valid_indices_global):
            cand_sel = cand_assignments_local == local_k
            full_assignments[candidate_indices[cand_sel]] = global_k

        for k, obj in enumerate(self.objects):
            new_indices = np.where(full_assignments == k)[0]
            obj['point_indices'] = new_indices.tolist()

        assigned = (full_assignments >= 0).sum()
        print(f"Reassignment complete: {assigned} / {N} points assigned "
              f"({N - assigned} background).")
        del g_feats_cand, o_feats, g_xyz_cand, o_cents
        torch.cuda.empty_cache()

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
        is_visible_global = (opacities.squeeze() > 0.05) # Lowered from 0.1

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
                        
                        # 3. Filter: Keep only points very close to the surface (within 1%).
                        # 1.01x is much stricter than the old 1.05x — it suppresses
                        # background splats that are physically behind the object.
                        depth_threshold = surface_depth * 1.01
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
        # Step 0.5: Merge objects whose centroids are nearly identical
        # (same real-world object detected from multiple views)
        self.merge_nearby_objects()
        # Step 1: Reassign SAM-candidate Gaussians by feature similarity (tight, no bleed)
        self.reassign_points_by_feature()
        # Step 2: DBSCAN clean each object's point set
        self.expand_objects_spatially()
        # Step 3: Recompute OBBs using final point assignments
        self.assign_points_and_compute_obbs()

    def merge_nearby_objects(self, radius: float = 0.25):
        """
        Merge objects whose 3D centroids are within `radius` metres of each other.

        This handles the common case where the same physical object is detected
        from multiple camera views and ends up as two separate clusters in feature
        space because the rendered crop angle was slightly different.  After
        merging we keep the highest-area detection's feature/crop as the
        representative and union the point indices.
        """
        if len(self.objects) < 2:
            return

        centroids = np.array([o['centroid'] if o['centroid'] is not None else [0., 0., 0.]
                              for o in self.objects])  # (K, 3)

        # Union-Find to group close objects
        parent = list(range(len(self.objects)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        K = len(self.objects)
        for i in range(K):
            for j in range(i + 1, K):
                d = np.linalg.norm(centroids[i] - centroids[j])
                if d < radius:
                    union(i, j)

        # Collect groups
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(K):
            groups[find(i)].append(i)

        n_merged = sum(1 for g in groups.values() if len(g) > 1)
        if n_merged:
            print(f"merge_nearby_objects: merging {n_merged} group(s) of co-located objects.")

        merged_objects = []
        for root, members in groups.items():
            if len(members) == 1:
                merged_objects.append(self.objects[members[0]])
                continue

            # Pick the highest-area member as the representative
            rep = max(members, key=lambda i: self.objects[i].get('area', 0))
            rep_obj = dict(self.objects[rep])  # shallow copy

            # Union point indices
            all_pts = []
            for m in members:
                pts = self.objects[m].get('point_indices', [])
                if len(pts):
                    all_pts.append(np.asarray(pts, dtype=np.int64))
            if all_pts:
                rep_obj['point_indices'] = np.unique(np.concatenate(all_pts)).tolist()

            # Average centroid
            cs = [self.objects[m]['centroid'] for m in members
                  if self.objects[m].get('centroid') is not None]
            rep_obj['centroid'] = np.mean(cs, axis=0).tolist() if cs else rep_obj.get('centroid')

            merged_objects.append(rep_obj)
            names = [self.objects[m].get('metadata', {}) for m in members]
            print(f"  Merged objects {members} -> 1 object (area rep={rep})")

        print(f"  Objects: {K} -> {len(merged_objects)} after merge.")
        self.objects = merged_objects

    def expand_objects_spatially(self):
        """
        DBSCAN-clean only (no spatial expansion).

        For each object, run DBSCAN on the SAM-reassigned seed points to identify
        the LARGEST spatially-coherent cluster. Discard all disconnected outlier
        sub-clusters (these are typically background/floor Gaussians that leaked
        through depth or projection noise in the SAM step).

        The result is a clean, tight per-object point set suitable for accurate
        OBB computation. No new Gaussians are added from outside the SAM seeds.
        """
        from sklearn.cluster import DBSCAN as _DBSCAN
        from sklearn.neighbors import NearestNeighbors as _NN

        N = len(self.xyz)
        final_assignments = np.full(N, -1, dtype=np.int32)
        # Seed from current tight SAM assignments
        for k, obj in enumerate(self.objects):
            for idx in obj.get('point_indices', []):
                final_assignments[idx] = k

        for k, obj in enumerate(self.objects):
            pts_idx = np.array(obj.get('point_indices', []), dtype=np.int64)
            if len(pts_idx) < 10:
                continue  # too few to clean

            pts = self.xyz[pts_idx]

            # Auto-scale epsilon: use 85th pct of NN distances.
            # The radial seed clip (70th pct from centroid) already enforces
            # tightness; DBSCAN's sole job here is to drop disconnected
            # floating sub-clusters — NOT to discard the bulk of good points.
            try:
                nn_dists, _ = _NN(n_neighbors=2).fit(pts).kneighbors(pts)
                eps_auto = float(np.percentile(nn_dists[:, 1], 85))
            except Exception:
                eps_auto = 0.05
            eps_auto = max(eps_auto, 0.02)

            # min_samples=5: light core density requirement
            dbs = _DBSCAN(eps=eps_auto, min_samples=5).fit(pts)
            labels = dbs.labels_

            unique_labels = [l for l in set(labels) if l >= 0]
            if not unique_labels:
                # No coherent cluster found – keep all (already assigned)
                continue

            # Keep LARGEST cluster only
            largest = max(unique_labels, key=lambda l: (labels == l).sum())
            inlier_mask = labels == largest
            inlier_idx  = pts_idx[inlier_mask]
            outlier_idx = pts_idx[~inlier_mask]

            # Update assignments: outliers go back to background
            final_assignments[outlier_idx] = -1
            # (inliers already assigned to k)

            cluster_size = inlier_mask.sum()
            outlier_size = (~inlier_mask).sum()
            if outlier_size > 0:
                print(f"  Object {k} DBSCAN: kept {cluster_size} / {len(pts_idx)} pts "
                      f"(removed {outlier_size} outliers, eps={eps_auto:.3f})")

        # Write back
        for k, obj in enumerate(self.objects):
            obj['point_indices'] = np.where(final_assignments == k)[0].tolist()

        assigned = (final_assignments >= 0).sum()
        print(f"DBSCAN-clean complete: {assigned} / {N} Gaussians assigned "
              f"({N - assigned} background).")

    def cluster_objects(self, detections, eps=0.1, min_samples=2):
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
        Uses centroid-distance percentile clipping to remove outlier splats.
        """
        if len(points) < 4: return None

        # 1. Centroid-distance percentile clipping (replaces LOF).
        #    Removes the farthest 15% of points from the centroid — the "spider web"
        #    floater splats that bloat bounding boxes. Much faster than LOF.
        centroid = points.mean(axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        dist_thresh = np.percentile(dists, 85)
        points = points[dists <= dist_thresh]
        if len(points) < 4:
            return None

        # 2. Compute 2D Footprint on Floor Plane (XZ)
        # Assuming Y (index 1) is UP.
        points_xz = points[:, [0, 2]]

        # PCA on 2D footprint
        center_xz = points_xz.mean(axis=0)
        centered_xz = points_xz - center_xz
        cov_xz = np.cov(centered_xz, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_xz)

        # Sort eigenvectors by descending variance
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]

        # Project points to 2D PCA axes
        proj_xz = centered_xz @ evecs
        min_xz = np.percentile(proj_xz, 2, axis=0)   # Robust Min
        max_xz = np.percentile(proj_xz, 98, axis=0)  # Robust Max

        # 3. Compute Vertical Extent (Y axis)
        y_vals = points[:, 1]
        y_min = np.percentile(y_vals, 2)
        y_max = np.percentile(y_vals, 98)

        # 4. Reconstruct 8 3D corners
        corners = []
        for x in [min_xz[0], max_xz[0]]:
            for z in [min_xz[1], max_xz[1]]:
                pt_xz = np.dot(np.array([x, z]), evecs.T) + center_xz
                corners.append([pt_xz[0], y_min, pt_xz[1]])  # bottom
                corners.append([pt_xz[0], y_max, pt_xz[1]])  # top

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
            if obj.get('area', 0) < 100: 
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
                    if isinstance(vlm_result, list):
                        if len(vlm_result) > 0:
                            vlm_result = vlm_result[0]
                        else:
                            vlm_result = {}
                            
                    obj_type = vlm_result.get('object_type', 'Unknown')
                    # Fallback for old prompt format
                    if 'is_building' in vlm_result and 'object_type' not in vlm_result:
                         if vlm_result['is_building']: obj_type = "Building"
                    
                    obj['building_info'] = {
                        'object_type': obj_type,
                        'stories_visual': vlm_result.get('stories_visual', 0),
                        'stories_estimated_geo': estimated_stories_geo,
                        'usage': vlm_result.get('usage', 'Unknown'),
                        'estimated_occupants': vlm_result.get('estimated_occupants', 0),
                        'description': vlm_result.get('description', '')
                    }
                    print(f" ID {obj['id']}: {obj_type} ({vlm_result.get('usage', '')})")
            else:
                 pass # No crop found

    def build_hierarchy(self, skip_frames=10):
        """
        This currently isn't used -- but conceptually I think it would be better
        to add back in. 

        Builds a hierarchy based on 3D centroid projection.
        Current default is a flat hierarchy.
        
        If object A's 2D bbox in A's best view contains object B's projected 3D centroid, then B is a child of A.
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
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        output_path = os.path.join(self.output_dir, filename)
        
        data = {
            "metadata": {
                "dataset_path": self.dataset_path,
                "model_path": self.model_path,
                "object_count": len(self.objects)
            },
            "objects": self.objects
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        
        print(f"Scene graph saved to {output_path}")

