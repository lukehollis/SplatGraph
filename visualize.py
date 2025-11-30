import os
import sys
import json
import argparse
import numpy as np
import torch
import viser
import viser.transforms as vtf
from plyfile import PlyData

# Add LangSplatV2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../LangSplatV2"))

# Mock CUDA modules if necessary (copied from scene_graph.py)
if not torch.cuda.is_available():
    try:
        import simple_knn
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["simple_knn"] = MagicMock()
        sys.modules["diff_gaussian_rasterization"] = MagicMock()
        print("Mocked CUDA modules for CPU execution.")

# Import Scene and Arguments
try:
    from scene import Scene
    from gaussian_renderer import GaussianModel
    from arguments import ModelParams
except ImportError as e:
    print(f"Error importing LangSplat modules: {e}")
    sys.exit(1)

def load_ply(path):
    plydata = PlyData.read(path)
    xyz = np.stack((plydata.elements[0]["x"],
                    plydata.elements[0]["y"],
                    plydata.elements[0]["z"]), axis=1)
    
    opacities = plydata.elements[0]["opacity"]
    opacities = 1 / (1 + np.exp(-opacities))
    opacities = opacities[:, None]
    
    # Colors (f_dc_0, f_dc_1, f_dc_2) are SH coefficients. 
    # For visualization, we can just use the DC component (0th band).
    # SH_C0 = 0.28209479177387814
    # color = 0.5 + SH_C0 * f_dc
    # But usually in PLY from 3DGS they are stored as f_dc_0, etc.
    
    f_dc_0 = plydata.elements[0]["f_dc_0"]
    f_dc_1 = plydata.elements[0]["f_dc_1"]
    f_dc_2 = plydata.elements[0]["f_dc_2"]
    
    SH_C0 = 0.28209479177387814
    colors = np.stack((f_dc_0, f_dc_1, f_dc_2), axis=1) * SH_C0 + 0.5
    colors = np.clip(colors, 0, 1)
    
    # Scales
    scale_0 = plydata.elements[0]["scale_0"]
    scale_1 = plydata.elements[0]["scale_1"]
    scale_2 = plydata.elements[0]["scale_2"]
    scales = np.exp(np.stack((scale_0, scale_1, scale_2), axis=1))
    
    # Rotations (quaternions)
    rot_0 = plydata.elements[0]["rot_0"]
    rot_1 = plydata.elements[0]["rot_1"]
    rot_2 = plydata.elements[0]["rot_2"]
    rot_3 = plydata.elements[0]["rot_3"]
    # 3DGS usually stores as w, x, y, z or x, y, z, w?
    # GaussianModel.py: rot_names = [rot_0, rot_1, rot_2, rot_3]
    # It seems to be w, x, y, z based on common 3DGS implementations.
    quats = np.stack((rot_0, rot_1, rot_2, rot_3), axis=1)
    
    return xyz, colors, scales, quats, opacities

def main():
    parser = argparse.ArgumentParser(description="Visualize SplatGraph results in Viser")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--graph_path", required=True, help="Path to the scene graph JSON")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration to load")
    
    args = parser.parse_args()
    
    # Start Viser server
    server = viser.ViserServer(port=args.port)
    print(f"Viser server started at http://localhost:{args.port}")
    
    # Load Scene Cameras
    print("Loading scene cameras...")
    class MockArgs:
        def __init__(self):
            self.source_path = args.dataset_path
            self.model_path = args.model_path
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.eval = True
            self.sh_degree = 3 
            
    mock_args = MockArgs()
    gaussians = GaussianModel(mock_args.sh_degree)
    scene = Scene(mock_args, gaussians, load_iteration=args.iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()
    print(f"Loaded {len(train_cameras)} training cameras.")
    
    # Load PLY
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        return

    print(f"Loading PLY from {ply_path}...")
    xyz, colors, scales, quats, opacities = load_ply(ply_path)
    
    # Store original opacities for filtering
    original_opacities = opacities.copy()
    
    # Compute covariances
    Rs = vtf.SO3(quats).as_matrix()
    M = Rs * scales[:, None, :]
    covariances = M @ M.transpose(0, 2, 1)
    
    splat_handle = server.scene.add_gaussian_splats(
        "/gaussians",
        centers=xyz,
        rgbs=colors,
        opacities=opacities,
        covariances=covariances
    )
    print("Added Gaussian Splats to Viser.")

    # Load Checkpoint for Language Features
    checkpoint_path = os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")
    gaussian_features = None
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            (model_params, _) = torch.load(checkpoint_path, map_location="cpu")
            # model_params is a tuple.
            # Index 7: _language_feature_logits
            # Index 8: _language_feature_codebooks
            
            if len(model_params) >= 9:
                logits = model_params[7]
                codebooks = model_params[8]
                
                if logits is not None and codebooks is not None:
                    print("Computing per-gaussian language features...")
                    L, K, D = codebooks.shape
                    
                    # Use GPU if available
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
                    # Split into chunks to avoid OOM if necessary
                    chunk_size = 100000
                    num_points = weights.shape[0]
                    features_list = []
                    
                    for i in range(0, num_points, chunk_size):
                        chunk_weights = weights[i:i+chunk_size]
                        chunk_features = chunk_weights @ codebooks_flat
                        chunk_features = chunk_features / (chunk_features.norm(dim=1, keepdim=True) + 1e-10)
                        features_list.append(chunk_features.detach().cpu())
                    
                    gaussian_features = torch.cat(features_list, dim=0).numpy()
                    print("Computed gaussian features.")
                else:
                    print("Warning: Checkpoint language features are None.")
            else:
                print(f"Warning: Checkpoint tuple length {len(model_params)} unexpected.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    # Load OpenCLIP
    print("Loading OpenCLIP...")
    try:
        from eval.openclip_encoder import OpenCLIPNetwork
        clip_model = OpenCLIPNetwork(device="cuda" if torch.cuda.is_available() else "cpu")
        print("OpenCLIP loaded.")
    except ImportError:
        print("Error importing OpenCLIPNetwork. Make sure LangSplatV2 is in path.")
        clip_model = None
    except Exception as e:
        print(f"Error loading OpenCLIP: {e}")
        clip_model = None

    # Load Scene Graph
    print(f"Loading scene graph from {args.graph_path}...")
    with open(args.graph_path, 'r') as f:
        graph_data = json.load(f)
        
    objects = graph_data.get("objects", [])
    print(f"Loaded {len(objects)} objects.")
    
    # GUI Elements
    with server.gui.add_folder("Scene Graph"):
        obj_names = [f"{obj['id']}: {obj.get('physics', {}).get('name', 'Unknown')}" for obj in objects]
        obj_map = {name: obj for name, obj in zip(obj_names, objects)}
        
        obj_dropdown = server.gui.add_dropdown(
            "Select Object",
            options=obj_names,
            initial_value=obj_names[0] if obj_names else None
        )
        
        threshold_slider = server.gui.add_slider(
            "Similarity Threshold",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.75
        )
        
        viz_mode_dropdown = server.gui.add_dropdown(
            "Visualization Mode",
            options=["Filter (Opacity)", "Highlight (Color)", "Original"],
            initial_value="Filter (Opacity)"
        )
        
        info_markdown = server.gui.add_markdown("Select an object to see details.")
        
        view_btn = server.gui.add_button("View Best Crop")
        
    with server.gui.add_folder("Language Query"):
        query_text = server.gui.add_text(
            "Query",
            initial_value="",
        )
        
        language_threshold_slider = server.gui.add_slider(
            "Language Threshold",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.5 # Default to median
        )
        
        language_viz_mode_dropdown = server.gui.add_dropdown(
            "Visualization Mode",
            options=["Hide Non-Matches", "Greyscale Non-Matches", "Heatmap"],
            initial_value="Hide Non-Matches"
        )
        
        query_btn = server.gui.add_button("Run Query")
        
        query_info = server.gui.add_markdown("Enter a text query to filter splats.")

        @server.gui.add_button("Clear Query", hint="Clear text query and reset view").on_click
        def _(_):
            nonlocal current_mode, current_text_feature
            query_text.value = ""
            current_text_feature = None
            current_mode = "object"
            query_info.content = "Enter a text query to filter splats."
            update_visualization()

    # State
    current_mode = "object" # or "text"
    current_text_feature = None

    def update_visualization():
        nonlocal current_mode, splat_handle
        
        if gaussian_features is None:
            return
            
        threshold = 0.5
        
        if current_mode == "object":
            name = obj_dropdown.value
            if not name: return
            obj = obj_map[name]
            
            # Update Info
            physics = obj.get('physics', {})
            md = f"""
            ### {physics.get('name', 'Object')}
            - **Material**: {physics.get('material', 'N/A')}
            - **Mass**: {physics.get('mass_kg', 'N/A')} kg
            - **Friction**: {physics.get('friction_coefficient', 'N/A')}
            - **Elasticity**: {physics.get('elasticity', 'N/A')}
            - **Description**: {physics.get('description', 'N/A')}
            
            **Stats**:
            - Area: {obj.get('area', 0)} px
            - Detections: {obj.get('detection_count', 0)}
            """
            info_markdown.content = md
            
            # Object feature
            target_feature = np.array(obj['feature'])
            threshold = threshold_slider.value
            
        elif current_mode == "text":
            if current_text_feature is None:
                return
            target_feature = current_text_feature
            info_markdown.content = f"### Query: {query_text.value}"
            threshold = language_threshold_slider.value
            
        # Compute similarity
        target_feature = target_feature / (np.linalg.norm(target_feature) + 1e-10)
        
        # gaussian_features: (N, D)
        # target_feature: (D,)
        sim = gaussian_features @ target_feature
        
        print(f"Similarity stats: min={sim.min():.4f}, max={sim.max():.4f}, mean={sim.mean():.4f}")
        
        # Percentile-based thresholding for linear sensitivity
        # Slider value 0.0 -> 0th percentile (Show all)
        # Slider value 1.0 -> 100th percentile (Show none)
        # Actually, if we want "Higher slider = Stricter", then:
        # Threshold = percentile(slider * 100)
        # mask = sim > Threshold
        
        if current_mode == "text":
            slider_val = language_threshold_slider.value
            threshold = np.percentile(sim, slider_val * 100)
            
            info_markdown.content = f"""### Query: {query_text.value}
            **Stats**:
            - Min: {sim.min():.4f}
            - Max: {sim.max():.4f}
            - Mean: {sim.mean():.4f}
            - Threshold ({slider_val*100:.0f}%): {threshold:.4f}
            """
            
            lang_mode = language_viz_mode_dropdown.value
            
            mask = sim > threshold
            print(f"Mask stats: {mask.sum()} / {len(mask)} splats visible ({(mask.sum()/len(mask))*100:.1f}%)")
            
            # Prepare new attributes
            new_opacities = original_opacities.copy()
            new_colors = colors.copy()
            
            if lang_mode == "Hide Non-Matches":
                new_opacities[~mask] = 0
                
            elif lang_mode == "Greyscale Non-Matches":
                # Convert non-matches to greyscale
                # Greyscale = 0.299*R + 0.587*G + 0.114*B
                non_match_colors = new_colors[~mask]
                grey = 0.299 * non_match_colors[:, 0] + 0.587 * non_match_colors[:, 1] + 0.114 * non_match_colors[:, 2]
                new_colors[~mask] = np.stack([grey, grey, grey], axis=1)
                # Optional: reduce opacity of non-matches?
                new_opacities[~mask] *= 0.1
                
            elif lang_mode == "Heatmap":
                # Normalize sim to 0-1 for coloring
                sim_min, sim_max = sim.min(), sim.max()
                if sim_max - sim_min > 1e-6:
                    sim_norm = (sim - sim_min) / (sim_max - sim_min)
                else:
                    sim_norm = np.zeros_like(sim)
                    
                heatmap_colors = np.zeros_like(colors)
                heatmap_colors[:, 0] = (1 - sim_norm) # Red
                heatmap_colors[:, 1] = sim_norm       # Green
                heatmap_colors[:, 2] = 0              # Blue
                new_colors = heatmap_colors
                
        else:
            # Object mode
            # Prepare new attributes
            new_opacities = original_opacities.copy()
            new_colors = colors.copy()
            
            mode = viz_mode_dropdown.value
            
            if mode == "Original":
                pass 
            elif mode == "Filter (Opacity)":
                mask = sim > threshold
                new_opacities[~mask] = 0
            elif mode == "Highlight (Color)":
                # Object mode highlight logic (keep simple for now)
                pass
            
        # Update Splats
        # Remove and re-add to ensure update
        splat_handle.remove()
        splat_handle = server.scene.add_gaussian_splats(
            "/gaussians",
            centers=xyz,
            rgbs=new_colors,
            opacities=new_opacities,
            covariances=covariances
        )

    @obj_dropdown.on_update
    def _(_):
        nonlocal current_mode
        current_mode = "object"
        update_visualization()
        
    @threshold_slider.on_update
    def _(_):
        update_visualization()
        
    @language_threshold_slider.on_update
    def _(_):
        if current_mode == "text":
            update_visualization()
            
    @language_viz_mode_dropdown.on_update
    def _(_):
        if current_mode == "text":
            update_visualization()
        
    @viz_mode_dropdown.on_update
    def _(_):
        update_visualization()
        
    @view_btn.on_click
    def _(_):
        name = obj_dropdown.value
        if not name: return
        obj = obj_map[name]
        
        crop_path = obj['best_crop_path']
        try:
            basename = os.path.basename(crop_path)
            view_idx = int(basename.split('_')[1])
            
            if 0 <= view_idx < len(train_cameras):
                cam = train_cameras[view_idx]
                position = cam.camera_center.cpu().numpy()
                quat = vtf.SO3.from_matrix(cam.R).wxyz
                
                client = server.get_clients()
                for c in client.values():
                    c.camera.position = position
                    c.camera.wxyz = quat
        except Exception as e:
            print(f"Error moving camera: {e}")
            
    @query_btn.on_click
    def _(_):
        nonlocal current_mode, current_text_feature
        text = query_text.value
        if not text: return
        
        if clip_model is None:
            query_info.content = "Error: CLIP model not loaded."
            return
            
        print(f"Encoding query: {text}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            # encode_text expects list
            text_embed = clip_model.encode_text([text], device)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            current_text_feature = text_embed.cpu().numpy()[0]
            
        current_mode = "text"
        update_visualization()
        query_info.content = f"Showing results for: **{text}**"



    # Initial update
    update_visualization()

    # Keep alive
    import time
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()
