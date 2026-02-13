
import os
import sys
import argparse
import glob
import torch
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Path Setup
# -----------------------------------------------------------------------------
# We assume this script is run from SplatGraph dir or we find paths relative to it.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_3D_REPO = os.path.join(CURRENT_DIR, "sam-3d-objects")
LANGSPLAT_DIR = os.path.join(CURRENT_DIR, "..", "LangSplatV2")

# Add Segment Anything to path
# We need to add the PARENT directory of the 'segment_anything' package
sys.path.append(os.path.join(LANGSPLAT_DIR, "submodules", "segment-anything-langsplat"))

# Add SAM-3D-Objects notebook dir to path for inference imports
sys.path.append(os.path.join(SAM_3D_REPO, "notebook"))

from inference import Inference, load_image, load_single_mask

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    print("Warning: Could not import segment_anything. Check paths.")

# -----------------------------------------------------------------------------
# Pipeline Class
# -----------------------------------------------------------------------------
class SAM3DPipeline:
    def __init__(self, sam_checkpoint, sam3d_config, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95, min_mask_region_area=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Initialize SAM
        print(f"Loading SAM from {sam_checkpoint}...")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        
        # Initialize with tunable parameters
        print(f"Initializing SAM Mask Generator with: points_per_side={points_per_side}, pred_iou_thresh={pred_iou_thresh}, stability_score_thresh={stability_score_thresh}, min_mask_region_area={min_mask_region_area}")
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area
        )
        
        # 2. Initialize SAM 3D
        print(f"Loading SAM 3D with config {sam3d_config}...")
        try:
            self.sam3d_inference = Inference(sam3d_config, compile=False)
        except Exception as e:
            print(f"Failed to load SAM 3D: {e}")
            self.sam3d_inference = None

    def process_image(self, image_path, output_dir, min_area_thresh=500, top_k=None):
        os.makedirs(output_dir, exist_ok=True)
        
        # Load Image
        image_pil = load_image(image_path)
        
        # Resize if too large (to avoid OOM)
        max_dim = 1024
        if image_pil.shape[0] > max_dim or image_pil.shape[1] > max_dim:
            print(f"Resizing image from {image_pil.shape[:2]} to max dimension {max_dim}...")
            # load_image returns numpy array, convert to PIL to resize then back or use cv2
            # load_image implementation in inference.py uses PIL.Image.open -> np.array
            # Let's use PIL again for easy resizing
            pil_img = Image.fromarray(image_pil)
            pil_img.thumbnail((max_dim, max_dim))
            image_pil = np.array(pil_img)
            
        print(f"Processing image with shape: {image_pil.shape}")
        image_np = np.array(image_pil)

        # 1. Segment
        print("Segmenting image...")
        masks_output = self.mask_generator.generate(image_np)
        
        # Handle tuple return (e.g. (masks, iou_preds, ...))
        if isinstance(masks_output, tuple):
             print(f"Mask generator returned tuple of length {len(masks_output)}. Using first element.")
             masks = masks_output[0]
        else:
             masks = masks_output
             
        print(f"Found {len(masks)} raw masks.")
        
        # Filter and Sort Masks
        filtered_masks = []
        image_area = image_np.shape[0] * image_np.shape[1]
        
        for mask_data in masks:
            area = mask_data['area']
            
            # Filter small objects
            if area < min_area_thresh:
                continue
                
            # Filter massive background objects (e.g. > 60% of image)
            if area > (image_area * 0.6):
                print(f"Skipping massive background mask with area {area} ({area/image_area:.1%} of image)")
                continue
                
            filtered_masks.append(mask_data)
            
        # Sort by area descending
        filtered_masks.sort(key=lambda x: x['area'], reverse=True)
        
        print(f"Using {len(filtered_masks)} masks after area filtering.")
        
        # Apply Top K
        if top_k is not None and top_k > 0:
            print(f"Selecting Top {top_k} largest objects...")
            filtered_masks = filtered_masks[:top_k]
            
        masks = filtered_masks

        # 2. Reconstruct each mask
        generated_objects = []

        for i, mask_data in enumerate(masks):
            area = mask_data['area']
            print(f"Processing Object {i} (Area: {area})...")
            
            # Prepare Mask for SAM 3D (it expects specific format/value?)
            # SAM 3D usage: inference(image, mask, ...) where mask is single channel
            
            # SAM output binary mask is boolean, convert to what SAM3D expects
            # Usually SAM3D 'load_single_mask' returns [H, W, 1] or similar
            # We construct it from the boolean mask
            full_mask = mask_data['segmentation'] 
            # SAM 3D inference expects: image (PIL or Tensor), mask (PIL or Tensor?)
            # Looking at blog: mask = load_single_mask(...)
            # We need to bridge this.
            
            if self.sam3d_inference:
                # OPTIMIZATION: Unload SAM model to free VRAM for 3D reconstruction
                print("Unloading SAM model to free memory...")
                if hasattr(self, 'mask_generator'):
                    del self.mask_generator
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    # Run Reconstruction
                    # Note: Inference expects mask as numpy array (impl assumes .astype)
                    mask_np = full_mask.astype(np.uint8) # Binary mask 0/1 or True/False
                    
                    # seed=42 for determinism or random? User asked for randomization of *position*, not necessarily shape.
                    output = self.sam3d_inference(image_pil, mask_np, seed=42) 
                    
                    # Output has "gs" (Gaussian Splat) and likely mesh
                    # Blog says: output["gs"].save_ply(...)
                    # And "export the mesh in formats such as .obj"
                    # We assume 'mesh' key or method on 'gs' or similar.
                    
                    obj_name = f"object_{i}"
                    ply_path = os.path.join(output_dir, f"{obj_name}.ply")
                    # mesh_path = os.path.join(output_dir, f"{obj_name}.obj") # if supported
                    
                    output["gs"].save_ply(ply_path)
                    
                    # Check if glb (trimesh object) is available
                    if "glb" in output and output["glb"] is not None:
                         # This is a trimesh object
                         try:
                             output["glb"].export(os.path.join(output_dir, f"{obj_name}.obj"))
                             generated_objects.append(os.path.join(output_dir, f"{obj_name}.obj"))
                         except Exception as e:
                             print(f"Failed to export GLB/Mesh: {e}")
                             # Fallback to PLY if mesh export fails
                             generated_objects.append(ply_path)
                    elif "mesh" in output:
                        print("Warning: Raw mesh found but no GLB export available.")
                        generated_objects.append(ply_path)
                    else:
                        generated_objects.append(ply_path)

                except Exception as e:
                    print(f"Failed to reconstruct object {i}: {e}")
            
        return generated_objects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="sam3d_output")
    parser.add_argument("--checkpoint", default="../LangSplatV2/ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument("--config", default="sam-3d-objects/checkpoints/hf/checkpoints/pipeline.yaml")
    parser.add_argument("--min_area", type=int, default=2000, help="Minimum mask area to process")
    parser.add_argument("--points_per_side", type=int, default=32, help="SAM points per side")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88, help="SAM prediction IOU threshold")
    parser.add_argument("--stability_score_thresh", type=float, default=0.95, help="SAM stability score threshold")
    parser.add_argument("--min_mask_region_area", type=int, default=100, help="SAM min mask region area") # Filter small disjoint regions internally
    parser.add_argument("--top_k", type=int, default=None, help="Process only the top K largest objects")
    args = parser.parse_args()

    # Resolve paths
    checkpoint_path = os.path.abspath(args.checkpoint)
    config_path = os.path.abspath(args.config)
    image_path = os.path.abspath(args.image)
    output_dir = os.path.abspath(args.output)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = SAM3DPipeline(
        checkpoint_path, 
        config_path,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area
    )
    pipeline.process_image(image_path, output_dir, min_area_thresh=args.min_area, top_k=args.top_k)

if __name__ == "__main__":
    main()
