import argparse
import sys
import os
import json
from dotenv import load_dotenv
from scene_graph import SplatSceneGraph
from physics_predictor import PhysicsPredictor

def main():
    # Load environment variables from .env file
    # Load environment variables from .env file (in the same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(script_dir, ".env"))

    parser = argparse.ArgumentParser(description="SplatGraph Pipeline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset (LangSplatV2 format)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained LangSplatV2 model directory (e.g., data/crate1/langsplat_output/crate1_0_3)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (default: dataset_path/graphs)")
    parser.add_argument("--openrouter_key", type=str, help="OpenRouter API Key")
    parser.add_argument("--iteration", type=int, default=10000, help="Model iteration to load")
    parser.add_argument("--level", type=int, default=3, help="LangSplatV2 level (1, 2, or 3)")
    parser.add_argument("--skip_frames", type=int, default=10, help="Number of frames to skip during segmentation")
    parser.add_argument("--cluster_eps", type=float, default=0.1, help="DBSCAN epsilon for object clustering (default 0.1)")
    parser.add_argument("--fill_backsides", action="store_true", help="Enable Generative Fill for object backsides")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.dataset_path, "graphs")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Generate Scene Graph
    print("Initializing Scene Graph Generator...")
    graph_gen = SplatSceneGraph(args.model_path, args.dataset_path, args.output_dir)
    graph_gen.load_model(args.iteration, args.level)
    graph_gen.segment_scene(skip_frames=args.skip_frames, cluster_eps=args.cluster_eps)
    graph_gen.save_graph("scene_graph_initial.json")

    # 2. Predict Physics
    openrouter_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        print("Initializing Physics Predictor...")
        predictor = PhysicsPredictor(openrouter_key)
        
        for obj in graph_gen.objects:
            print(f"Predicting physics for object {obj['id']}...")
            physics_props = predictor.predict(obj)
            if physics_props:
                obj['metadata'] = physics_props.get('metadata', {})
                obj['usd_physics'] = physics_props.get('usd_physics', {})
            else:
                print(f"Failed to predict physics for object {obj['id']}")
    else:
        print("No OpenRouter key provided (checked args and .env). Skipping physics prediction.")

    print("Skipping hierarchy building (Flat Graph)...")
    graph_gen.save_graph("scene_graph_final.json")

    # 3. Generative Fill (Optional)
    if args.fill_backsides:
        print("\n--- Starting Generative Fill ---")
        import subprocess
        
        # Construct PLY path assuming standard LangSplat structure
        ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        graph_path = os.path.join(args.output_dir, "scene_graph_final.json")
        
        cmd = [
            sys.executable, "SplatGraph/fill_backsides.py",
            "--graph_path", graph_path,
            "--ply_path", ply_path,
            "--model_path", args.model_path,
            "--iteration", str(args.iteration)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            print("Generative Fill completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running Generative Fill: {e}")

if __name__ == "__main__":
    main()
