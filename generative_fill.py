
import os
import sys
import logging
import time
import numpy as np
import torch
from PIL import Image
import trimesh
import open3d as o3d
import rembg

# Add TripoSR to path
sys.path.append(os.path.join(os.path.dirname(__file__), "TripoSR"))

try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
except ImportError as e:
    logging.warning(f"Could not import TripoSR: {e}. Generative Fill will be disabled.")
    TSR = None

class GenerativeFiller:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.model = None
        self.rembg_session = None
        
        if TSR is not None:
            logging.info("Initializing TripoSR model...")
            self.model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model.to(self.device)
            self.rembg_session = rembg.new_session()
            
    def generate_mesh(self, image_path, output_path=None):
        """
        Generates a 3D mesh from a single image.
        Returns a trimesh.Trimesh object.
        """
        if self.model is None:
            raise RuntimeError("TripoSR model not initialized.")
            
        # 1. Load and Preprocess Image
        logging.info(f"Processing image: {image_path}")
        image = Image.open(image_path)
        
        # Ensure RGBA and remove background if needed
        if image.mode != "RGBA" or self.rembg_session:
            image = remove_background(image, self.rembg_session)
            image = resize_foreground(image, 0.85)
            
        # Prepare for model
        image_np = np.array(image).astype(np.float32) / 255.0
        # Simple composite on white background for the RGB input expected by model
        image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
        image_pil = Image.fromarray((image_np * 255.0).astype(np.uint8))
        
        # 2. Run Inference
        logging.info("Running inference...")
        with torch.no_grad():
            scene_codes = self.model([image_pil], device=self.device)
            
        # 3. Extract Mesh
        logging.info("Extracting mesh...")
        # resolution=256 is default, typically sufficient for low-poly collision proxy
        meshes = self.model.extract_mesh(scene_codes, True, resolution=256)
        mesh = meshes[0]
        
        if output_path:
            mesh.export(output_path)
            
        return mesh

class MeshAligner:
    def align(self, generated_mesh: trimesh.Trimesh, partial_points: np.ndarray):
        """
        Aligns the generated mesh (canonical) to the partial point cloud (world).
        """
        # Convert Trimesh to Open3D
        gen_o3d = o3d.geometry.TriangleMesh()
        gen_o3d.vertices = o3d.utility.Vector3dVector(generated_mesh.vertices)
        gen_o3d.triangles = o3d.utility.Vector3iVector(generated_mesh.faces)
        gen_o3d.compute_vertex_normals()
        
        # Convert Points to Open3D
        partial_o3d = o3d.geometry.PointCloud()
        partial_o3d.points = o3d.utility.Vector3dVector(partial_points)
        partial_o3d.estimate_normals()
        
        # --- 1. Centering ---
        gen_center = gen_o3d.get_center()
        partial_center = partial_o3d.get_center()
        
        # Move Gen to Origin temporarily
        gen_o3d.translate(-gen_center)
        # Move Partial to Origin analysis (conceptual)
        
        # --- 2. Scale Estimation ---
        # Heuristic: Compare bounding box diagonals
        # Note: Generated mesh is usually normalized to unit cube [-0.5, 0.5]
        gen_bbox = gen_o3d.get_axis_aligned_bounding_box()
        partial_bbox = partial_o3d.get_axis_aligned_bounding_box()
        
        gen_diag = np.linalg.norm(gen_bbox.get_max_bound() - gen_bbox.get_min_bound())
        partial_diag = np.linalg.norm(partial_bbox.get_max_bound() - partial_bbox.get_min_bound())
        
        if gen_diag < 1e-6: scale = 1.0
        else: scale = partial_diag / gen_diag
        
        logging.info(f"Estimated scale: {scale:.4f}")
        gen_o3d.scale(scale, center=(0,0,0))
        
        # --- 3. Initial Translation ---
        # Move Gen to Partial Center
        gen_o3d.translate(partial_center)
        
        # --- 4. ICP Alignment (Rigid) ---
        # We assume the generated mesh is roughly upright. 
        # If rotation is wild, we might need Global Registration (RANSAC).
        # For object-centric crops, standard ICP often works if scale is close.
        
        threshold = 0.02 * scale # Adaptive threshold
        trans_init = np.eye(4) # Identity since we already centered
        
        logging.info("Running ICP...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            gen_o3d, partial_o3d, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Apply Transform
        gen_o3d.transform(reg_p2p.transformation)
        logging.info(f"ICP Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")
        
        # Convert back to Trimesh
        aligned_mesh = trimesh.Trimesh(
            vertices=np.asarray(gen_o3d.vertices),
            faces=np.asarray(gen_o3d.triangles)
        )
        
        return aligned_mesh

if __name__ == "__main__":
    # Smoke Test
    logging.basicConfig(level=logging.INFO)
    filler = GenerativeFiller()
    # Mock Processing (Requires an image path)
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        filler.generate_mesh(img_path, "test_output.obj")
