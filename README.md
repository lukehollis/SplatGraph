
[Screencast from 2025-11-30 23-22-26.webm](https://github.com/user-attachments/assets/720aabcb-3981-489d-8dcc-e19c9fcbdd9d)

# SplatGraph 
## [under active development]

**SplatGraph** is a pipeline for generating a hierarchical 3D scene graph with predictive physics properties from a set of input images. It leverages **3D Gaussian Splatting** for scene reconstruction, **LangSplatV2** for open-vocabulary language annotations, and **VLMs (Visual Language Models)** for semantic reasoning and physics property prediction.

Credit to advisement of Heng Yang, Zhutian Chen, Wanhua Li, Hanspeter Pfister, and Alejandro Escontrela who have variously provided guidance to this project on integrating their own projects. Anything of merit is theirs, any shortcomings my own. 

## Development goals 
- 3d gaussian splat to segmented scene graph 
- predictive physics properties in scene graph 
- robotic training in gauss gym with realstic physics for objects, i.e. when picking up a piece of cloth or item wrapped in plastic, segmented 3d gaussians behave as expected in real world 


## Pipeline Overview

The full pipeline consists of the following steps:

1.  **Input Data**: A collection of images of a static scene (e.g., `data/crate1`).
2.  **3D Gaussian Splatting (3DGS) Training**: Reconstructs the scene as a 3D Gaussian Splatting model.
3.  **LangSplatV2 Training**: Trains a language feature field on top of the 3DGS model, enabling text-based querying and segmentation.
4.  **Scene Graph Generation**:
    *   **Visual Analysis**: Renders the scene from multiple views.
    *   **Segmentation**: Uses **SAM (Segment Anything Model)** and LangSplat language features to segment objects in 2D views.
    *   **Object Clustering**: Aggregates 2D detections into unique 3D objects using density-based clustering (DBSCAN) on the language features.
    *   **Physics Prediction**: Uses a VLM (currently **Grok-4.1-fast** via OpenRouter) to analyze the visual appearance of each object (best crop + context) and predict detailed physics properties.
    *   **Graph Construction**: Builds a scene graph (currently flat, with hierarchy support planned) containing all objects and their properties.

## Installation

Ensure you have the `LangSplatV2` environment set up (see `../LangSplatV2/README.md`).

```bash
pip install -r requirements.txt
# You also need the SAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ../LangSplatV2/ckpts/
```

## Usage

### Running the Full Pipeline

Use the provided shell script to run all steps from scratch:

```bash
# Run on a specific scene
./scripts/run_full_pipeline.sh data/crate1
```

### Running SplatGraph Only

If you already have a trained LangSplat model, you can run the scene graph generation directly:

```bash
./scripts/06_splatgraph.sh data/crate1
```

### Visualization

Visualize the generated scene graph and LangSplat model using the interactive Viser GUI:

```bash
python SplatGraph/visualize.py \
  --dataset_path data/crate1 \
  --model_path data/crate1/langsplat_output/crate1_0_3 \
  --graph_path results/crate1_graph/scene_graph_final.json \
  --port 8080
```

**Features:**
*   **Scene Graph Tree**: Explore the list of objects. Expand nodes to view detailed **Physics Properties** (Mass, Friction, Material, etc.).
*   **Bounding Boxes**: Toggle **Show Bounding Boxes** and switch between **AABB** (Axis-Aligned) and **OBB** (Oriented) modes.
*   **Language Query**: Enter text queries (e.g., "wooden crate") to filter splats.
    *   **Visualization Modes**: "Hide Non-Matches", "Greyscale Non-Matches", "Heatmap".
    *   **Threshold**: Adjust the similarity threshold dynamically.
*   **Object Selection**: Click on any splat in the 3D view to select the corresponding object and view its details in the "Selected Object Info" panel.
*   **Render Modes**: Switch between standard **RGB** and **Segmentation** view (objects colored by ID).

## Output

The pipeline produces a `scene_graph_final.json` file containing the scene graph:

```json
{
  "objects": [
    {
      "id": 0,
      "physics": {
        "name": "wooden_crate",
        "material": "wood",
        "mass_kg": 5.0,
        "friction_coefficient": 0.6,
        "elasticity": 0.2,
        "motion_type": "dynamic",
        "collision_primitive": "box",
        "center_of_mass": "center",
        "destructibility": "breakable",
        "health": 50,
        "flammability": 0.8,
        "surface_sound": "wood",
        "roughness": 0.9,
        "metallic": 0.0,
        "dimensions": {
            "length": 0.5,
            "width": 0.5,
            "height": 0.5
        },
        "description": "A sturdy wooden crate used for storage."
      },
      "children": [],
      "best_crop_path": "...",
      "feature": [...]
    }
  ]
}
```
