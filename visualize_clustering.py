import os
import sys
import json
import argparse
import numpy as np
import torch
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



def load_ply(path):
    plydata = PlyData.read(path)
    xyz = np.stack((plydata.elements[0]["x"],
                    plydata.elements[0]["y"],
                    plydata.elements[0]["z"]), axis=1)
    
    opacities = plydata.elements[0]["opacity"]
    opacities = 1 / (1 + np.exp(-opacities))
    opacities = opacities[:, None]
    
    return xyz, opacities

def main():
    parser = argparse.ArgumentParser(description="Visualize Language Feature Clustering")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--graph_path", required=True, help="Path to the scene graph JSON")
    parser.add_argument("--output_path", required=True, help="Path to save the output plot")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of points to sample per object")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration to load")
    
    args = parser.parse_args()
    
    print(f"Loading scene graph from {args.graph_path}...")
    with open(args.graph_path, 'r') as f:
        graph_data = json.load(f)
    objects = graph_data.get("objects", [])
    print(f"Loaded {len(objects)} objects.")
    
    # Load PLY to get XYZ
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        return

    print(f"Loading PLY from {ply_path}...")
    xyz, opacities = load_ply(ply_path)
    
    # Load Checkpoint for Language Features
    checkpoint_path = os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")
    gaussian_features = None
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
             # Fallback for older torch versions
             checkpoint_data = torch.load(checkpoint_path, map_location="cpu")


        if isinstance(checkpoint_data, tuple):
            if len(checkpoint_data) >= 1:
                model_params = checkpoint_data[0]
            else:
                model_params = []
        else:
            model_params = checkpoint_data
        
        if len(model_params) >= 9:
            logits = model_params[7]
            codebooks = model_params[8]
            
            if logits is not None and codebooks is not None:
                print("Computing per-gaussian language features...")
                L, K, D = codebooks.shape
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logits = logits.to(device)
                codebooks = codebooks.to(device)
                
                weights_list = []
                for i in range(L):
                    level_logits = logits[:, i*K : (i+1)*K]
                    level_weights = torch.softmax(level_logits, dim=1)
                    weights_list.append(level_weights)
                    
                weights = torch.cat(weights_list, dim=1)
                codebooks_flat = codebooks.view(-1, D)
                
                # Compute features
                chunk_size = 100000
                num_points = weights.shape[0]
                features_list = []
                
                for i in range(0, num_points, chunk_size):
                    chunk_weights = weights[i:i+chunk_size]
                    chunk_features = chunk_weights @ codebooks_flat
                    chunk_features = chunk_features / (chunk_features.norm(dim=1, keepdim=True) + 1e-10)
                    features_list.append(chunk_features.detach().cpu())
                
                gaussian_features = torch.cat(features_list, dim=0).numpy()
                print(f"Computed gaussian features: {gaussian_features.shape}")
            else:
                print("Warning: Checkpoint language features are None.")
                return
        else:
            print(f"Warning: Checkpoint tuple length {len(model_params)} unexpected.")
            return
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return

    # Collect samples
    all_features = []
    all_labels = []
    label_names = []
    
    print("Sampling points from objects...")
    
    # Helper to traverse children
    flat_objects = []
    def flatten_objects(objs):
        for obj in objs:
            flat_objects.append(obj)
            flatten_objects(obj.get('children', []))
    flatten_objects(objects)
    
    objects_with_points = 0
    
    for obj in flat_objects:
        indices = obj.get('point_indices')
        if not indices or len(indices) == 0:
            continue
            
        indices = np.array(indices)
        
        # Filter by bounds if needed (sanity check)
        if indices.max() >= gaussian_features.shape[0]:
            print(f"Warning: Indices for obj {obj['id']} out of bounds.")
            continue
            
        # Sample
        if len(indices) > args.num_points:
            selected_indices = np.random.choice(indices, args.num_points, replace=False)
        else:
            selected_indices = indices
            
        feats = gaussian_features[selected_indices]
        
        # Append
        all_features.append(feats)
        all_labels.extend([obj['id']] * len(feats))
        
        # Metadata Name
        name = obj.get('metadata', {}).get('name', f"Obj {obj['id']}")
        # Only add unique mappings? No, we need color map later.
        # Store name map
        
        objects_with_points += 1

    if objects_with_points == 0:
        print("No objects had point indices! Attempting fallback assignment (Nearest Neighbor to centroids)...")
        # Fallback: Assign based on centroids if point_indices missing
        # This is a simplified assignment logic
        
        from sklearn.neighbors import NearestNeighbors
        
        obj_centroids = []
        obj_ids = []
        valid_objs = []
        
        for obj in flat_objects:
            if 'centroid' in obj and obj['centroid']:
                 obj_centroids.append(obj['centroid'])
                 obj_ids.append(obj['id'])
                 valid_objs.append(obj)
                 
        if not obj_centroids:
            print("No centroids found either. Cannot visualize.")
            return

        obj_centroids = np.array(obj_centroids)
        nbrs = NearestNeighbors(n_neighbors=1).fit(obj_centroids)
        
        # We need to assign ALL gaussians? No, that's too slow and noisy.
        # Let's just pick gaussians near the centroids.
        
        # Actually, let's just create a synthetic "perfect" assignment for visualization 
        # based on the assumption that the Scene Graph is correct.
        # But we really want to see the *features*.
        
        # Let's take the Top K points nearest to each centroid.
        print(f"Assigning {args.num_points} nearest points to each of {len(valid_objs)} centroids...")
        
        nbrs_search = NearestNeighbors(n_neighbors=args.num_points).fit(xyz)
        
        distances, indices_list = nbrs_search.kneighbors(obj_centroids)
        
        for i, obj in enumerate(valid_objs):
            point_idxs = indices_list[i]
            feats = gaussian_features[point_idxs]
            
            all_features.append(feats)
            all_labels.extend([obj['id']] * len(feats))
            objects_with_points += 1

    if not all_features:
        print("No features collected.")
        return
        
    X = np.concatenate(all_features, axis=0)
    y = np.array(all_labels)
    
    print(f"Total points to plot: {X.shape[0]}")
    
    # Dimensionality Reduction
    print("Running PCA (50 dims)...")
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    print("Running t-SNE (2 dims)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_pca)
    
    # Plotting
    print(f"Plotting to {args.output_path}...")
    plt.figure(figsize=(12, 10))
    
    # Create colormap
    unique_ids = np.unique(y)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    
    # Map ID -> Name
    id_to_name = {}
    for obj in flat_objects:
        name = obj.get('metadata', {}).get('name', f"Obj {obj['id']}")
        id_to_name[obj['id']] = name

    for i, uid in enumerate(unique_ids):
        mask = y == uid
        name = id_to_name.get(uid, f"Obj {uid}")
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=name, s=5, alpha=0.6)
        
    plt.title(f"Language Feature Clustering (Iter {args.iteration})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150)
    print("Done.")

if __name__ == "__main__":
    main()
