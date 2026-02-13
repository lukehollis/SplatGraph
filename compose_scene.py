
from omni.isaac.kit import SimulationApp

# Launch headless first
config = {"headless": True}
simulation_app = SimulationApp(config)

import argparse
import os
import random
import glob
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage, save_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from pxr import Usd, UsdGeom, Gf, Sdf

def compose_scene(assets_dir, output_usd_path):
    world = World()
    world.reset()
    
    stage = world.stage
    
    # 1. Add Ground Plane (Optional, or just a desk surface)
    # We can create a simple plane
    UsdGeom.Xform.Define(stage, "/World")
    plane = UsdGeom.Plane.Define(stage, "/World/Ground")
    # Make it visible
    
    # 3. Find Objects
    # Find all generated objects (OBJ or USDZ from SplatGraph)
    asset_files = glob.glob(os.path.join(assets_dir, "*.obj"))
    asset_files += glob.glob(os.path.join(assets_dir, "*_*.usdz")) # Pattern for object_id.usdz
    
    # Filter out invalid or non-object files
    asset_files = [f for f in asset_files if "background" not in os.path.basename(f) and "scene" not in os.path.basename(f)]
        
    print(f"Found {len(asset_files)} assets to place.")
    
    # 3. Place Objects Randomly
    for i, asset_path in enumerate(asset_files):
        prim_path = f"/World/Object_{i}"
        
        # Add Reference
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        
        # Random Pose
        # Desk is roughly at Z=0 if we assume ground
        # Random X, Y in e.g. [-0.5, 0.5] range
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(-0.3, 0.3)
        z = 0.0 # On surface
        
        # Random Rotation Z
        rot = random.uniform(0, 360)
        
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        
        # Reset xform ops just in case
        xform.ClearXformOpOrder()
        
        op_translate = xform.AddTranslateOp()
        op_translate.Set(Gf.Vec3d(x, y, z))
        
        op_rotate = xform.AddRotateZOp()
        op_rotate.Set(rot)
        
        # Scale? Assume SAM3D output is unit or reasonable.
        # Maybe scale down if they are huge 
        # op_scale = xform.AddScaleOp()
        # op_scale.Set(Gf.Vec3d(1.0, 1.0, 1.0))

    # 4. Lighting
    # Add Dome Light
    light_prim_path = "/World/DomeLight"
    light = UsdLux.DomeLight.Define(stage, light_prim_path)
    
    # Randomize Intensity
    intensity = random.uniform(500, 2000)
    light.CreateIntensityAttr(intensity)
    
    # Randomize Color (Subtle tint)
    tint = Gf.Vec3f(
        random.uniform(0.9, 1.0),
        random.uniform(0.9, 1.0),
        random.uniform(0.9, 1.0)
    )
    light.CreateColorAttr(tint)
    
    # Randomize texture rotation (if texture were used, but here just rotation of dome)
    # light.AddRotateYOp().Set(...)

    # 5. Save
    save_stage(output_usd_path)
    print(f"Scene saved to {output_usd_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_dir", required=True)
    parser.add_argument("--output", default="scene.usd")
    args = parser.parse_args()
    
    compose_scene(os.path.abspath(args.assets_dir), os.path.abspath(args.output))
    simulation_app.close()

if __name__ == "__main__":
    from pxr import UsdLux # Import here to ensure context
    main()
