import argparse
import numpy as np
import viser
import viser.transforms as vtf
from plyfile import PlyData
import time
import os

def load_ply(path):
    print(f"Loading PLY from {path}...")
    plydata = PlyData.read(path)
    
    # Vertex properties
    xyz = np.stack((plydata.elements[0]["x"],
                    plydata.elements[0]["y"],
                    plydata.elements[0]["z"]), axis=1)
    
    opacities = plydata.elements[0]["opacity"]
    opacities = 1 / (1 + np.exp(-opacities))
    opacities = opacities[:, None]
    
    # Colors (f_dc_0, f_dc_1, f_dc_2)
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
    quats = np.stack((rot_0, rot_1, rot_2, rot_3), axis=1)
    
    return xyz, colors, scales, quats, opacities

def main():
    parser = argparse.ArgumentParser(description="Visualize Base Gaussian Splatting Model")
    parser.add_argument("--ply_path", required=True, help="Path to the point_cloud.ply file")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (e.g. 2 for half points)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ply_path):
        print(f"Error: File not found at {args.ply_path}")
        return

    # Start Viser server
    server = viser.ViserServer(port=args.port)
    print(f"Viser server started at http://localhost:{args.port}")
    
    # Load Data
    xyz, colors, scales, quats, opacities = load_ply(args.ply_path)
    
    if args.downsample > 1:
        print(f"Downsampling by {args.downsample}...")
        xyz = xyz[::args.downsample]
        colors = colors[::args.downsample]
        scales = scales[::args.downsample]
        quats = quats[::args.downsample]
        opacities = opacities[::args.downsample]
    
    print(f"Loaded {len(xyz)} splats.")
    
    # Compute covariances
    # This can be heavy, so we do it in chunks if needed, but for visualization Viser handles it?
    # Viser expects covariances or scales/quats.
    # add_gaussian_splats supports scales and quats directly now in newer versions, 
    # but let's stick to the working `visualize.py` method of computing covariances if we want to be safe,
    # OR check if we can pass scales/quats.
    # The `visualize.py` computed covariances manually. Let's do that to be consistent.
    
    print("Computing covariances...")
    Rs = vtf.SO3(quats).as_matrix()
    M = Rs * scales[:, None, :]
    covariances = M @ M.transpose(0, 2, 1)
    
    print("Adding to Viser...")
    # Create a container for the splats so we can rotate them
    splat_node = server.scene.add_transform_controls("/world_root")
    
    splat_handle = server.scene.add_gaussian_splats(
        "/world_root/gaussians",
        centers=xyz,
        rgbs=colors,
        opacities=opacities,
        covariances=covariances
    )
    
    # Orientation GUI
    with server.gui.add_folder("Orientation"):
        @server.gui.add_button("Flip Upside Down (Rotate 180 X)").on_click
        def _(_):
            # Toggle 180 degree rotation around X
            current_rot = splat_node.wxyz
            # Construct quaternion for 180 deg around X: [0, 1, 0, 0] (w, x, y, z)
            # Actually w=cos(90)=0, x=sin(90)=1. So (0, 1, 0, 0)
            
            # If we are already flipped, reset. 
            # Simple toggle: check if w is close to 0
            if abs(current_rot[0]) < 0.1:
                # Reset to identity
                splat_node.wxyz = (1.0, 0.0, 0.0, 0.0)
            else:
                # Set to 180 X
                splat_node.wxyz = (0.0, 1.0, 0.0, 0.0)
                
        server.gui.add_markdown("Use the transform gizmo on the scene root to adjust manually if needed.")

    # Apply default flip if requested (often Colmap is inverted relative to Viser)
    # Let's default to flipping since the user said it's upside down
    splat_node.wxyz = (0.0, 1.0, 0.0, 0.0)
    
    print("Done. Press Ctrl+C to exit.")
    
    # Keep alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
