import argparse
import os
import json
from scene_graph import SplatSceneGraph
from physics_predictor import PhysicsPredictor

def main():
    parser = argparse.ArgumentParser(description="SplatGraph Pipeline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset (LangSplatV2 format)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained LangSplatV2 model directory (e.g., data/crate1/langsplat_output/crate1_0_3)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (default: dataset_path/graphs)")
    parser.add_argument("--openrouter_key", type=str, help="OpenRouter API Key")
    parser.add_argument("--iteration", type=int, default=10000, help="Model iteration to load")
    parser.add_argument("--level", type=int, default=3, help="LangSplatV2 level (1, 2, or 3)")
    parser.add_argument("--skip_frames", type=int, default=10, help="Number of frames to skip during segmentation")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.dataset_path, "graphs")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Generate Scene Graph
    print("Initializing Scene Graph Generator...")
    graph_gen = SplatSceneGraph(args.model_path, args.dataset_path, args.output_dir)
    graph_gen.load_model(args.iteration, args.level)
    graph_gen.segment_scene(skip_frames=args.skip_frames)
    graph_gen.build_hierarchy()
    graph_gen.save_graph("scene_graph_initial.json")
    
    # 2. Predict Physics
    if args.openrouter_key:
        print("Initializing Physics Predictor...")
        predictor = PhysicsPredictor(args.openrouter_key)
        
        for obj in graph_gen.objects:
            print(f"Predicting physics for object {obj['id']}...")
            physics_props = predictor.predict(obj)
            if physics_props:
                obj['physics'] = physics_props
            else:
                print(f"Failed to predict physics for object {obj['id']}")
        
        graph_gen.save_graph("scene_graph_final.json")
    else:
        print("No OpenRouter key provided. Skipping physics prediction.")

if __name__ == "__main__":
    main()
