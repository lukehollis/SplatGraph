import os
import sys
import json
import argparse
import numpy as np
import torch
import viser
import viser.transforms as vtf
from plyfile import PlyData
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.stats import mode
from sklearn.cluster import DBSCAN

# Add LangSplatV2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../LangSplatV2"))

from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams


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

    # 3DGS usually stores as w, x, y, z 
    # GaussianModel.py: rot_names = [rot_0, rot_1, rot_2, rot_3]
    quats = np.stack((rot_0, rot_1, rot_2, rot_3), axis=1)
    
    return xyz, colors, scales, quats, opacities

def main():
    parser = argparse.ArgumentParser(description="Visualize SplatGraph results in Viser")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--graph_path", required=False, default=None, help="Path to the scene graph JSON")
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
    scene = Scene(mock_args, gaussians, load_iteration=args.iteration, shuffle=False, max_train_views=1, max_test_views=1)
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
    
    # Create root frame
    server.scene.add_frame("/world", wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0))


    
    splat_handle = server.scene.add_gaussian_splats(
        "/world/gaussians",
        centers=xyz,
        rgbs=colors,
        opacities=opacities,
        covariances=covariances
    )

    # -- Auto-center camera on the Gaussian cloud --
    # Compute scene centroid and a sensible viewing distance.
    scene_centroid = xyz.mean(axis=0).astype(float)   # (3,)
    scene_std      = xyz.std(axis=0).mean()
    view_dist      = float(max(scene_std * 3.0, 1.0))

    # Look-at vector: position camera behind+above the centroid.
    import math
    cam_position = (
        float(scene_centroid[0]),
        float(scene_centroid[1] - view_dist),
        float(scene_centroid[2] + view_dist * 0.5),
    )
    # Look down the -Y axis (common for 3DGS scenes captured from above/front)
    # wxyz quaternion for looking from cam_position toward scene_centroid
    look_dir = np.array(scene_centroid) - np.array(cam_position)
    look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)

    print(f"Scene centroid: {scene_centroid}  |  cam_position: {cam_position}")

    @server.on_client_connect
    def _on_client(client):
        """Teleport each new client camera to look at the scene."""
        client.camera.look_at(
            position=cam_position,
            target=tuple(float(v) for v in scene_centroid),
            up=(0.0, 0.0, 1.0),
        )

    # Splat Click Logic
    def attach_splat_click(handle):
        @handle.on_click
        def _(event):
            nonlocal current_mode
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
                    # TODO: build a flat map id->obj
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
                    current_mode = "object"

    print("Added Gaussian Splats to Viser.")

    # Load Checkpoint for Language Features
    checkpoint_path = os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")
    gaussian_features = None
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint_data, tuple):
            if len(checkpoint_data) >= 1:
                model_params = checkpoint_data[0]
            else:
                model_params = []
        else:
            model_params = checkpoint_data
        
        if len(model_params) >= 9:
            # Check if index 7 looks like logits (N, L*K) or max_radii2D (N,)
            param7 = model_params[7]
            if isinstance(param7, torch.Tensor) and param7.ndim == 1:
                print("Warning: Checkpoint appears to be a standard Gaussian Splatting model (no language features).")
                print("         Bounding boxes and language queries will be disabled.")
                gaussian_features = None
            else:
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
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    # Load OpenCLIP
    print("Loading OpenCLIP...")
    from eval.openclip_encoder import OpenCLIPNetwork
    clip_model = OpenCLIPNetwork(device="cuda" if torch.cuda.is_available() else "cpu")
    print("OpenCLIP loaded.")

    # Load Scene Graph
    objects = []
    if args.graph_path and os.path.exists(args.graph_path):
        print(f"Loading scene graph from {args.graph_path}...")
        with open(args.graph_path, 'r') as f:
            graph_data = json.load(f)
        objects = graph_data.get("objects", [])
        print(f"Loaded {len(objects)} objects.")
    else:
        print("No scene graph loaded (graph_path not provided or not found).")
        object_masks = {} # Ensure object_masks is defined if derived later or used
        graph_data = {}
    
    # GUI Elements
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

    render_mode_dropdown = server.gui.add_dropdown(
        "Render Mode",
        options=["RGB", "Segmentation"],
        initial_value="RGB"
    )

    @server.gui.add_button("⌖ Center View", hint="Jump camera to scene center").on_click
    def _(_):
        for client in server.get_clients().values():
            client.camera.look_at(
                position=cam_position,
                target=tuple(float(v) for v in scene_centroid),
                up=(0.0, 0.0, 1.0),
            )

    show_boxes_checkbox = server.gui.add_checkbox(
        "Show Bounding Boxes",
        initial_value=True
    )
    
    box_type_dropdown = server.gui.add_dropdown(
        "Box Type",
        options=["AABB", "OBB"],
        initial_value="OBB"
    )

    # Transform GUI
    with server.gui.add_folder("Transform", expand_by_default=False):
        # Position
        with server.gui.add_folder("Position"):
            pos_x = server.gui.add_number("X", initial_value=0.0, step=0.1)
            pos_y = server.gui.add_number("Y", initial_value=0.0, step=0.1)
            pos_z = server.gui.add_number("Z", initial_value=0.0, step=0.1)
        
        # Rotation
        with server.gui.add_folder("Rotation"):
            rot_roll = server.gui.add_slider("Roll", min=-np.pi, max=np.pi, step=0.01, initial_value=0.0)
            rot_pitch = server.gui.add_slider("Pitch", min=-np.pi, max=np.pi, step=0.01, initial_value=0.0)
            rot_yaw = server.gui.add_slider("Yaw", min=-np.pi, max=np.pi, step=0.01, initial_value=0.0)

        @server.gui.add_button("Flip Z (Fix Upside Down)").on_click
        def _(_):
            # Toggle between 0 and pi for roll (or pitch?)
            # Usually "upside down" means rotation around X or Z by 180.
            # Let's try rotating around X by 180 (Roll).
            current = rot_roll.value
            if abs(current) < 0.1:
                rot_roll.value = np.pi
            else:
                rot_roll.value = 0.0

    # Update Transform Logic
    def update_transform():
        # Update /world frame
        world_handle = server.scene.add_frame(
            "/world",
            position=(pos_x.value, pos_y.value, pos_z.value),
            wxyz=vtf.SO3.from_rpy_radians(rot_roll.value, rot_pitch.value, rot_yaw.value).wxyz
        )

    # Bind callbacks
    pos_x.on_update(lambda _: update_transform())
    pos_y.on_update(lambda _: update_transform())
    pos_z.on_update(lambda _: update_transform())
    rot_roll.on_update(lambda _: update_transform())
    rot_pitch.on_update(lambda _: update_transform())
    rot_yaw.on_update(lambda _: update_transform())
    
    with server.gui.add_folder("Scene Graph", expand_by_default=False):
        def add_obj_to_gui(obj):
            # Try to get name from metadata, then usd_physics
            metadata = obj.get('metadata', {})
            usd_physics = obj.get('usd_physics', {})
            
            name = metadata.get('name', f"Object {obj['id']}")
            label = f"{name} (ID: {obj['id']})"
            
            # Use a folder for the object
            with server.gui.add_folder(label, expand_by_default=False):
                def format_dict(d, indent=0):
                    lines = []
                    prefix = "  " * indent
                    for k, v in d.items():
                        if isinstance(v, dict):
                            lines.append(f"{prefix}- **{k}**:")
                            lines.extend(format_dict(v, indent + 1))
                        else:
                            clean_v = str(v).replace("{", "(").replace("}", ")")
                            lines.append(f"{prefix}- **{k}**: {clean_v}")
                    return lines

                # Physics Info
                md_lines = []
                
                # Metadata
                if metadata:
                    md_lines.append("**Metadata:**")
                    md_lines.extend(format_dict(metadata))
                else:
                    md_lines.append("- (No Metadata Found)")
                
                md_lines.append("\n") # Spacer

                # USD Physics
                if usd_physics:
                     md_lines.append("**USD Physics:**")
                     md_lines.extend(format_dict(usd_physics))
                else:
                     md_lines.append("- (No USD Physics Found)")

                formatted = "\n".join(md_lines)
                server.gui.add_markdown(formatted)
                
                # Recursively add children
                for child in obj.get('children', []):
                    add_obj_to_gui(child)

        for obj in objects:
            add_obj_to_gui(obj)


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
        language_query_text = server.gui.add_text(
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
            language_query_text.value = ""
            current_text_feature = None
            current_mode = "object"
            query_info.content = "Enter a text query to filter splats."

    # Pre-compute Object Similarities and Segmentation Colors
    print("Pre-computing object similarities and segmentation colors...")
    
    # Maps for O(1) access
    obj_id_to_idx = {}
    obj_idx_to_id = []
    obj_centroids_list = []
    obj_features_list = []
    
    def collect_obj_features(obj):
        feat = np.array(obj['feature'])
        feat = feat / (np.linalg.norm(feat) + 1e-10)
        
        centroid = obj.get('centroid')
        if centroid is None:
            centroid = [0, 0, 0] 
            
        idx = len(obj_idx_to_id)
        obj_id_to_idx[obj['id']] = idx
        obj_idx_to_id.append(obj['id'])
        obj_features_list.append(feat)
        obj_centroids_list.append(centroid)
        
        for child in obj.get('children', []):
            collect_obj_features(child)
            
    for obj in objects:
        collect_obj_features(obj)
        
    similarity_matrix = None # We will store the BEST match index here instead of full matrix 
    # 1. best_obj_idx (N,) - The index of the winning object
    # 2. best_obj_score (N,) - The score of the winner (for thresholding background)
    
    best_obj_indices = None
    best_obj_scores = None
    
    # Build segmentation colors DIRECTLY from point_indices stored in the JSON.
    # This faithfully reflects the tight assignments computed by the pipeline
    # (reassign_points_by_feature + DBSCAN clean) rather than re-running a loose
    # global cosine-similarity pass over all 1.1M Gaussians.
    N = len(xyz)
    segmentation_colors = np.full((N, 3), 0.35, dtype=np.float32)  # dark grey = unassigned
    best_obj_indices = np.full(N, -1, dtype=np.int32)
    best_obj_scores  = np.zeros(N, dtype=np.float32)

    np.random.seed(42)
    K_vis = len(obj_idx_to_id)
    obj_color_map = np.random.rand(max(K_vis, 1), 3).astype(np.float32)
    # Make colors vivid (avoid dark shades)
    obj_color_map = obj_color_map * 0.7 + 0.3

    def assign_seg_colors_from_json(obj_list):
        """Walk the object list (inc. children) and paint point_indices."""
        for obj in obj_list:
            oid  = obj['id']
            idxs = obj.get('point_indices', [])
            if not idxs:
                assign_seg_colors_from_json(obj.get('children', []))
                continue
            local_k = obj_id_to_idx.get(oid, -1)
            if local_k < 0:
                assign_seg_colors_from_json(obj.get('children', []))
                continue
            arr = np.asarray(idxs, dtype=np.int64)
            arr = arr[arr < N]  # guard against stale indices
            segmentation_colors[arr] = obj_color_map[local_k]
            best_obj_indices[arr] = local_k
            best_obj_scores[arr]  = 1.0  # fully assigned
            assign_seg_colors_from_json(obj.get('children', []))

    assign_seg_colors_from_json(objects)
    assigned_count = (best_obj_indices >= 0).sum()
    print(f"Segmentation colors: {assigned_count} / {N} Gaussians assigned from JSON point_indices.")
    print("Assignment computed.")
    
    

    def build_scene_tree(box_type="OBB"):
        print(f"Building Scene Tree ({box_type})...")
        
        def add_scene_node(obj, parent_path="/SceneGraph"):
            # Clean name for path
            metadata = obj.get('metadata', {})
            safe_name = metadata.get('name', 'Unknown').replace(" ", "_").replace("/", "-")
            node_name = f"{obj['id']}_{safe_name}"

            # Use /world/SceneGraph
            if parent_path == "/SceneGraph":
                parent_path = "/world/SceneGraph"
            
            path = f"{parent_path}/{node_name}"
            
            # Compute bounds
            if 'obb' in obj and obj['obb'] is not None and len(obj['obb']) > 0:
                 # Ensure it's 8 corners
                 corners = np.array(obj['obb'])
                 bounds_result = (corners, None, "OBB")
            else:
                 # bounds_result = compute_object_bounds(obj, box_type=box_type)
                 print(f"Warning: Object {obj['id']} has no OBB in JSON. Skipping box.")
                 bounds_result = None
            
            if bounds_result is not None:
                if bounds_result[2] == "OBB":
                    corners = bounds_result[0]
                    # Corners are [c0, c1, c2, c3, c4, c5, c6, c7]
                    
                    c000 = corners[0]
                    c001 = corners[1]
                    c010 = corners[2]
                    c011 = corners[3]
                    c100 = corners[4]
                    c101 = corners[5]
                    c110 = corners[6]
                    c111 = corners[7]
                    
                    # Lines
                    lines = np.array([
                        [c000, c100], [c010, c110], [c001, c101], [c011, c111], # X-axis lines
                        [c000, c010], [c100, c110], [c001, c011], [c101, c111], # Y-axis lines
                        [c000, c001], [c100, c101], [c010, c011], [c110, c111]  # Z-axis lines
                    ])
                    
                else:
                    # Fallback AABB
                    min_pt, max_pt, _ = bounds_result
                    
                    # Create box lines (12 lines)
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
                    colors=(1.0, 1.0, 0.0), # Yellow (Float)
                    position=(0.0, 0.0, 0.0),
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    visible=show_boxes_checkbox.value
                )
                
                object_handles[obj['id']] = handle
                    
            else:
                server.scene.add_frame(path)
                
            # Recursively add children
            for child in obj.get('children', []):
                add_scene_node(child, path)

        for obj in objects:
            add_scene_node(obj)
        print("Scene Tree built.")

    # Build Scene Tree
    print("Building Scene Tree...")
    build_scene_tree(box_type="OBB")
    print("Scene Tree built.")

    # Visibility Loop
    def visibility_loop():
        pass # Not used
            
    # Pre-compute object masks for visibility logic
    print("Pre-computing object masks...")
    object_masks = {}
    if best_obj_indices is not None:
        for oid, idx in obj_id_to_idx.items():
            # Mask: this Gaussian was assigned to this object in the JSON
            mask = (best_obj_indices == idx) & (best_obj_scores > 0.5)
            object_masks[oid] = mask
    print("Masks computed.")
    
    # Attach initial click handler
    attach_splat_click(splat_handle)

    # Callbacks for Language Features
    @query_btn.on_click
    def _(_):
        nonlocal current_mode, current_text_feature
        text = language_query_text.value
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
    last_render_mode = None
    last_text_params = None
    last_show_boxes = True
    last_box_type = "OBB"

    # Settings GUI
    with server.gui.add_folder("Settings"):
        opacity_threshold_slider = server.gui.add_slider(
            "Opacity Threshold",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.2
        )
        scale_threshold_slider = server.gui.add_slider(
            "Scale Threshold",
            min=0.0,
            max=10.0,
            step=0.1,
            initial_value=0.5
        )

    # Update Loop
    last_text_params = None
    last_render_mode = None
    last_visibility_state = {}
    last_opacity_threshold = -1.0
    last_scale_threshold = -1.0
    
    while True:
        # Check if update needed
        needs_update = False
        
        # Check Box Type
        current_box_type = box_type_dropdown.value
        if current_box_type != last_box_type:
            last_box_type = current_box_type
            build_scene_tree(box_type=current_box_type)
            # Re-apply visibility
            for handle in object_handles.values():
                handle.visible = show_boxes_checkbox.value
        
        # Check Box Visibility
        current_show_boxes = show_boxes_checkbox.value
        if current_show_boxes != last_show_boxes:
            last_show_boxes = current_show_boxes
            # Toggle visibility of all handles
            for handle in object_handles.values():
                handle.visible = current_show_boxes
            # No need to re-render splats for this, just handles
            
        # 1. Check Mode Change
        if current_mode != last_mode:
            needs_update = True
            last_mode = current_mode
        
        # Check Visibility State
        current_visibility_state = {oid: h.visible for oid, h in object_handles.items()}
        if current_visibility_state != last_visibility_state:
            needs_update = True
            last_visibility_state = current_visibility_state
            
        # Check Text Query
        if current_mode == "text":
            current_text_params = (
                language_query_text.value,
                language_threshold_slider.value,
                language_viz_mode_dropdown.value,
                id(current_text_feature) if current_text_feature is not None else 0
            )
            if current_text_params != last_text_params:
                needs_update = True
                last_text_params = current_text_params
                
        # Check Render Mode
        current_render_mode = render_mode_dropdown.value
        if current_render_mode != last_render_mode:
            needs_update = True
            last_render_mode = current_render_mode

        # Check Settings
        current_opacity_threshold = opacity_threshold_slider.value
        if abs(current_opacity_threshold - last_opacity_threshold) > 1e-6:
            needs_update = True
            last_opacity_threshold = current_opacity_threshold

        current_scale_threshold = scale_threshold_slider.value
        if abs(current_scale_threshold - last_scale_threshold) > 1e-6:
            needs_update = True
            last_scale_threshold = current_scale_threshold
        
        if needs_update:
            new_opacities = original_opacities.copy()
            
            # Apply Settings Thresholds
            if current_opacity_threshold > 0:
                new_opacities[original_opacities < current_opacity_threshold] = 0
            
            if current_scale_threshold > 0:
                # scales is (N, 3)
                # We can use max scale dimension
                max_scales = scales.max(axis=1)
                new_opacities[max_scales > current_scale_threshold] = 0
            
            if current_render_mode == "Segmentation":
                new_colors = segmentation_colors.copy()
            else:
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
                
                # Fallback: If no objects detected (handles empty), show everything
                if not object_handles:
                    any_visible = True
                    visible_mask[:] = True
                
                print(f"DEBUG: any_visible: {any_visible}")
                print(f"DEBUG: last_visibility_state len: {len(last_visibility_state)}")
                print(f"DEBUG: object_masks len: {len(object_masks)}")
                
                if any_visible:
                    # Only keep visible objects
                    # But also respect threshold
                    mask = ~visible_mask
                    new_opacities[mask] = 0
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
            # Force re-creation to ensure Viser updates
            if splat_handle is not None:
                splat_handle.remove()
            
            splat_handle = server.scene.add_gaussian_splats(
                "/world/gaussians",
                centers=xyz,
                rgbs=new_colors,
                opacities=new_opacities,
                covariances=covariances
            )
            attach_splat_click(splat_handle)
            
if __name__ == "__main__":
    main()
