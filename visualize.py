import os
import sys
import json
import argparse
import numpy as np
import torch
import viser
import viser.transforms as vtf
import viser.transforms as vtf
from plyfile import PlyData
import time

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
    
    # Splat Click Logic
    def attach_splat_click(handle):
        @handle.on_click
        def _(event):
            nonlocal selected_object, current_mode
            if event.instance_index is None: return
            
            idx = event.instance_index
            
            # Find candidate objects
            candidates = []
            for oid, mask in object_masks.items():
                if idx < len(mask) and mask[idx]:
                    # Find object data
                    # Inefficient search, but N is small
                    # We can cache obj map
                    obj_data = None
                    # Search in objects list (recursive)
                    # Better: build a flat map id->obj
                    pass
                    candidates.append(oid)
            
            if not candidates: return
            
            # Pick the smallest object (fewest splats)
            # This handles hierarchy (child is smaller than parent)
            best_oid = None
            min_count = float('inf')
            
            for oid in candidates:
                count = object_masks[oid].sum()
                if count < min_count:
                    min_count = count
                    best_oid = oid
            
            if best_oid:
                # Find the object dict
                # We need a map
                target_obj = object_map.get(best_oid)
                if target_obj:
                    selected_object = target_obj
                    current_mode = "object"
                    update_info_panel()

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
    # GUI Elements
    selected_object = None
    object_handles = {} # map id -> handle
    object_map = {} # map id -> obj dict
    
    def build_object_map(objs):
        for o in objs:
            object_map[o['id']] = o
            build_object_map(o.get('children', []))
    build_object_map(objects)
    
    # State
    current_mode = "object" # "object" or "text"
    current_text_feature = None
    


    # Scene Graph Tree GUI
    print("Building Scene Graph GUI...")
    with server.gui.add_folder("Scene Graph"):
        def add_obj_to_gui(obj):
            name = obj.get('physics', {}).get('name', f"Object {obj['id']}")
            label = f"{name} (ID: {obj['id']})"
            
            # Use a folder for the object
            with server.gui.add_folder(label, open=False):
                # Physics Info
                physics = obj.get('physics', {})
                md = f"""
                **Physics Properties:**
                - **Material**: {physics.get('material', 'N/A')}
                - **Mass**: {physics.get('mass_kg', 'N/A')} kg
                - **Friction**: {physics.get('friction_coefficient', 'N/A')}
                - **Elasticity**: {physics.get('elasticity', 'N/A')}
                - **Description**: {physics.get('description', 'N/A')}
                """
                server.gui.add_markdown(md)
                
                # Select Button
                @server.gui.add_button("Select Object").on_click
                def _(_):
                    nonlocal selected_object, current_mode
                    selected_object = obj
                    current_mode = "object"
                    # update_info_panel is defined later, but that's fine for callbacks
                    update_info_panel()

                # Recursively add children
                for child in obj.get('children', []):
                    add_obj_to_gui(child)

        for obj in objects:
            add_obj_to_gui(obj)

    # Info Panel
    with server.gui.add_folder("Selected Object Info"):
        info_markdown = server.gui.add_markdown("Click an object in the scene or tree to view details.")
        
        view_btn = server.gui.add_button("View Best Crop")
        
        @server.gui.add_button("Clear Selection", color="red").on_click
        def _(_):
            nonlocal selected_object, current_mode
            selected_object = None
            current_mode = "object"
            update_info_panel()
            update_visualization()

        threshold_slider = server.gui.add_slider(
            "Similarity Threshold",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.6
        )

    # Language Query GUI
    print("Setting up Language Query GUI...")
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
            initial_value=0.5
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

    # Helper to compute 3D bounds
    def compute_object_bounds(obj, threshold=0.6):
        if gaussian_features is None:
            return None
            
        target_feature = np.array(obj['feature'])
        target_feature = target_feature / (np.linalg.norm(target_feature) + 1e-10)
        
        sim = gaussian_features @ target_feature
        mask = sim > threshold
        
        if mask.sum() == 0:
            return None
            
        points = xyz[mask]
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        
        return min_pt, max_pt
    def add_scene_node(obj, parent_path="/SceneGraph"):
        nonlocal selected_object
        
        # Clean name for path
        safe_name = obj.get('physics', {}).get('name', 'Unknown').replace(" ", "_").replace("/", "-")
        node_name = f"{obj['id']}_{safe_name}"
        path = f"{parent_path}/{node_name}"
        
        # Compute bounds
        bounds = compute_object_bounds(obj)
        
        if bounds is not None:
            min_pt, max_pt = bounds
            center = (min_pt + max_pt) / 2
            size = max_pt - min_pt
            
            # Create box lines (12 lines) using absolute coordinates
            # We set the node position to (0,0,0) to avoid transform inheritance issues
            min_x, min_y, min_z = min_pt
            max_x, max_y, max_z = max_pt
            
            # 8 corners
            c0 = np.array([min_x, min_y, min_z])
            c1 = np.array([max_x, min_y, min_z])
            c2 = np.array([max_x, max_y, min_z])
            c3 = np.array([min_x, max_y, min_z])
            c4 = np.array([min_x, min_y, max_z])
            c5 = np.array([max_x, min_y, max_z])
            c6 = np.array([max_x, max_y, max_z])
            c7 = np.array([min_x, max_y, max_z])
            
            # Pairs
            lines = np.array([
                [c0, c1], [c1, c2], [c2, c3], [c3, c0], # Bottom
                [c4, c5], [c5, c6], [c6, c7], [c7, c4], # Top
                [c0, c4], [c1, c5], [c2, c6], [c3, c7]  # Vertical
            ])
            
            # Add lines to Scene Tree
            handle = server.scene.add_line_segments(
                path,
                points=lines,
                colors=(255, 255, 0), # Yellow
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0)
            )
            
            # DEBUG: Add flat wireframe to check coordinates
            server.scene.add_line_segments(
                f"/Debug/{node_name}",
                points=lines,
                colors=(0, 255, 255), # Cyan for debug
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0)
            )
            
            if len(object_handles) < 5:
                print(f"DEBUG: Added {node_name} bounds: min={min_pt}, max={max_pt}")
            
            object_handles[obj['id']] = handle
            
            # Click handler removed as LineSegmentsHandle doesn't support it.
            # We rely on splat clicking.
                
        else:
            server.scene.add_frame(path)
            
        # Recursively add children
        for child in obj.get('children', []):
            add_scene_node(child, path)

    def update_info_panel():
        if selected_object is None: return
        obj = selected_object
        
        physics = obj.get('physics', {})
        md = f"""
        ### {physics.get('name', 'Object')} (ID: {obj['id']})
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

    # Build Scene Tree
    print("Building Scene Tree...")
    for obj in objects:
        add_scene_node(obj)
    print("Scene Tree built.")

    # Visibility Loop
    def visibility_loop():
        while True:
            # Check visibility of handles
            # If a handle is hidden, we should hide its splats?
            # Or rather: We want to show splats for visible objects.
            
            # This is tricky because splats are shared.
            # If multiple objects claim a splat, and one is hidden, what happens?
            # Usually, we want to show splats that belong to ANY visible object.
            
            # Let's construct a mask of visible splats.
            if gaussian_features is not None:
                visible_mask = np.zeros(len(xyz), dtype=bool)
                active_objects = False
                
                for obj_id, handle in object_handles.items():
                    if handle.visible:
                        active_objects = True
                        # Find object (inefficient to search list every time, but okay for now)
                        # Better: store obj in handle? No.
                        # We have obj_id.
                        # We need the obj data to compute bounds/mask again?
                        # Re-computing mask every frame is too slow.
                        # We should cache the masks!
                        pass
                
                # Caching masks
                # Let's do this outside the loop first.
            
    # Pre-compute masks for all objects?
    # Memory usage: N_splats * N_objects bits.
    # N_splats ~ 1M. N_objects ~ 100.
    # 100M bits = 12.5 MB. Totally fine.
    
    print("Pre-computing object masks...")
    object_masks = {}
    for obj in objects:
        def compute_mask_recursive(o):
            mask = compute_object_bounds(o, threshold=0.6) # Re-using logic, but we need the mask, not bounds
            # Wait, compute_object_bounds returns bounds.
            # Let's refactor compute_object_bounds or just duplicate logic.
            
            target_feature = np.array(o['feature'])
            target_feature = target_feature / (np.linalg.norm(target_feature) + 1e-10)
            sim = gaussian_features @ target_feature
            m = sim > 0.6 # Fixed threshold for definition? Or use slider?
            # Using slider value from GUI? 
            # If we use slider, we can't pre-compute.
            # But the bounding boxes are static.
            # Let's use a fixed threshold for "Object Definition" (bounds and tree).
            # And maybe dynamic for "Visualization"?
            # The user wants to toggle visibility.
            
            object_masks[o['id']] = m
            
            for child in o.get('children', []):
                compute_mask_recursive(child)
                
        compute_mask_recursive(obj)
    print("Masks computed.")
    
    # Attach initial click handler
    attach_splat_click(splat_handle)

    # Callbacks for Language Features
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
        query_info.content = f"Showing results for: **{text}**"

    # Visibility / Update Loop
    last_visibility_state = {}
    last_mode = None
    last_text_params = None

    while True:
        # Check if update is needed
        needs_update = False
        
        # 1. Check Mode Change
        if current_mode != last_mode:
            needs_update = True
            last_mode = current_mode
            
        if current_mode == "object":
            # Check Scene Tree visibility
            current_visibility_state = {oid: h.visible for oid, h in object_handles.items()}
            if current_visibility_state != last_visibility_state:
                needs_update = True
                last_visibility_state = current_visibility_state.copy()
                
        elif current_mode == "text":
            # Check text params
            current_text_params = (
                language_threshold_slider.value,
                language_viz_mode_dropdown.value,
                id(current_text_feature) if current_text_feature is not None else 0
            )
            if current_text_params != last_text_params:
                needs_update = True
                last_text_params = current_text_params
        
        if needs_update:
            new_opacities = original_opacities.copy()
            new_colors = colors.copy()
            
            if current_mode == "object":
                # Object Mode Logic
                visible_mask = np.zeros(len(xyz), dtype=bool)
                any_visible = False
                
                for oid, is_visible in last_visibility_state.items():
                    if is_visible:
                        if oid in object_masks:
                            visible_mask |= object_masks[oid]
                            any_visible = True
                
                if any_visible:
                    new_opacities[~visible_mask] = 0
                else:
                    new_opacities[:] = 0
                    
            elif current_mode == "text":
                # Text Mode Logic
                if current_text_feature is not None and gaussian_features is not None:
                    target_feature = current_text_feature
                    
                    # Compute similarity
                    sim = gaussian_features @ target_feature
                    
                    # Threshold
                    slider_val = language_threshold_slider.value
                    threshold = np.percentile(sim, slider_val * 100)
                    
                    mask = sim > threshold
                    
                    lang_mode = language_viz_mode_dropdown.value
                    
                    if lang_mode == "Hide Non-Matches":
                        new_opacities[~mask] = 0
                        
                    elif lang_mode == "Greyscale Non-Matches":
                        non_match_colors = new_colors[~mask]
                        grey = 0.299 * non_match_colors[:, 0] + 0.587 * non_match_colors[:, 1] + 0.114 * non_match_colors[:, 2]
                        new_colors[~mask] = np.stack([grey, grey, grey], axis=1)
                        new_opacities[~mask] *= 0.1
                        
                    elif lang_mode == "Heatmap":
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

            # Update Splats
            splat_handle.remove()
            splat_handle = server.scene.add_gaussian_splats(
                "/gaussians",
                centers=xyz,
                rgbs=new_colors,
                opacities=new_opacities,
                covariances=covariances
            )
            attach_splat_click(splat_handle)
            
if __name__ == "__main__":
    main()
