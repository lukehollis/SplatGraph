
import os
import argparse
import json
import numpy as np
import torch
import gc
from plyfile import PlyData
from tqdm import tqdm
import logging

# Local Imports
from generative_fill import GenerativeFiller, MeshAligner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_ply(path):
    plydata = PlyData.read(path)
    x = np.array(plydata['vertex']['x'])
    y = np.array(plydata['vertex']['y'])
    z = np.array(plydata['vertex']['z'])
    return np.stack((x, y, z), axis=1)

def compute_assignments(xyz, model_path, iteration, graph_data):
    """
    Computes point-to-object assignments using LangSplat features.
    Returns: numpy array of object indices (indices into graph_data['objects'])
    """
    checkpoint_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    logging.info(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, tuple): model_params = checkpoint[0]
    else: model_params = checkpoint
        
    logits = model_params[7]
    codebooks = model_params[8]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = logits.to(device)
    codebooks = codebooks.to(device)
    
    if codebooks.ndim == 2:
        K, D = codebooks.shape
        L = 1
        codebooks = codebooks.unsqueeze(0)
    else:
        L, K, D = codebooks.shape
        
    # Weight computation
    weights_list = []
    for i in range(L):
        weights_list.append(torch.softmax(logits[:, i*K : (i+1)*K], dim=1))
    weights = torch.cat(weights_list, dim=1)
    codebooks_flat = codebooks.view(-1, D)
    
    # Pre-extract object features
    obj_feat_list = []
    obj_centroid_list = []
    
    def collect_objs(obj_list):
        for obj in obj_list:
            feat = np.array(obj['feature'])
            feat = feat / (np.linalg.norm(feat) + 1e-10)
            centroid = obj.get('centroid', [0.0, 0.0, 0.0])
            obj_feat_list.append(feat)
            obj_centroid_list.append(centroid)
            collect_objs(obj.get('children', []))
            
    objects = graph_data.get('objects', [])
    collect_objs(objects)
    
    if not obj_feat_list:
        logging.warning("No objects found in graph.")
        return np.full(xyz.shape[0], -1)

    o_feats_th = torch.from_numpy(np.stack(obj_feat_list)).float().to(device)
    o_centroids_th = torch.from_numpy(np.array(obj_centroid_list)).float().to(device)
    g_xyz_th = torch.from_numpy(xyz).float().to(device)
    
    num_points = xyz.shape[0]
    assignments = np.full(num_points, -1, dtype=np.int32)
    chunk_size = 50000
    spatial_weight = 1.0
    
    logging.info("Assigning points (running feature + spatial matching)...")
    with torch.no_grad():
        for i in tqdm(range(0, num_points, chunk_size)):
            chunk_weights = weights[i:i+chunk_size]
            g_chunk = chunk_weights @ codebooks_flat
            g_chunk = g_chunk / (g_chunk.norm(dim=1, keepdim=True) + 1e-10)
            g_xyz_chunk = g_xyz_th[i:i+chunk_size]
            
            sim = g_chunk @ o_feats_th.T
            dist = torch.cdist(g_xyz_chunk, o_centroids_th)
            final_score = sim - (dist * spatial_weight)
            
            _, best_idx = torch.max(final_score, dim=1)
            assignments[i:i+chunk_size] = best_idx.cpu().numpy()
            
    # Cleanup VRAM aggressively
    del logits, codebooks, weights, model_params, checkpoint, o_feats_th, o_centroids_th, g_xyz_th
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("LangSplat model unloaded.")
    
    return assignments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--ply_path", required=True)
    parser.add_argument("--model_path", required=True, help="Path to LangSplat output")
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--output_dir", default="filled_meshes")
    args = parser.parse_args()
    
    # 1. Load Data
    logging.info(f"Loading Graph: {args.graph_path}")
    with open(args.graph_path, 'r') as f:
        graph = json.load(f)
        
    logging.info(f"Loading PLY: {args.ply_path}")
    xyz = load_ply(args.ply_path)
    
    # 2. Compute Assignments (Memory Heavy)
    assignments = compute_assignments(xyz, args.model_path, args.iteration, graph)
    
    # 3. Initialize Generative Fill (Memory Heavy - but separate now)
    filler = GenerativeFiller()
    aligner = MeshAligner()
    
    filled_dir = os.path.join(os.path.dirname(args.graph_path), args.output_dir)
    os.makedirs(filled_dir, exist_ok=True)
    
    # Flatten object list for index mapping (must match compute_assignments order)
    flat_objects = []
    def flatten(obj_list):
        for obj in obj_list:
            flat_objects.append(obj)
            flatten(obj.get('children', []))
    flatten(graph.get('objects', []))
    
    # 4. Processing Loop
    logging.info("Starting Generative Fill...")
    
    for i, obj in enumerate(flat_objects):
        obj_id = obj['id']
        crop_path = obj.get('best_crop_path')
        
        if not crop_path or not os.path.exists(crop_path):
            logging.warning(f"Object {obj_id}: No crop found. Skipping.")
            continue
            
        # Get Points
        mask = (assignments == i)
        if mask.sum() < 50:
            logging.warning(f"Object {obj_id}: Not enough points ({mask.sum()}). Skipping.")
            continue
            
        points = xyz[mask]
        
        try:
            # Generate
            logging.info(f"Object {obj_id}: Generating mesh from {crop_path}...")
            mesh = filler.generate_mesh(crop_path)
            
            # Align
            logging.info(f"Object {obj_id}: Aligning mesh...")
            aligned_mesh = aligner.align(mesh, points)
            
            # Save
            mesh_filename = f"obj_{obj_id}_filled.obj"
            mesh_path = os.path.join(filled_dir, mesh_filename)
            aligned_mesh.export(mesh_path)
            logging.info(f"Saved to {mesh_path}")
            
            # Update Graph Metadata for Physics
            if 'usd_physics' not in obj: obj['usd_physics'] = {}
            obj['usd_physics']['collision_mesh'] = mesh_path
            
        except Exception as e:
            logging.error(f"Object {obj_id}: Failed generative fill: {e}")
            
    # 5. Save Updated Graph
    output_graph_path = args.graph_path.replace(".json", "_filled.json")
    with open(output_graph_path, 'w') as f:
        json.dump(graph, f, indent=4)
    logging.info(f"Saved updated graph to {output_graph_path}")

if __name__ == "__main__":
    main()
