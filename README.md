# SplatGraph

**SplatGraph** is a pipeline for generating a hierarchical 3D scene graph with predictive physics properties from a set of input images. It leverages **3D Gaussian Splatting** for scene reconstruction, **LangSplatV2** for open-vocabulary language annotations, and **VLMs (Visual Language Models)** for semantic reasoning and physics property prediction.

## Pipeline Overview

The full pipeline consists of the following steps:

1.  **Input Data**: A collection of images of a static scene (e.g., `data/crate1`).
2.  **3D Gaussian Splatting (3DGS) Training**: Reconstructs the scene as a 3D Gaussian Splatting model.
3.  **LangSplatV2 Training**: Trains a language feature field on top of the 3DGS model, enabling text-based querying and segmentation.
4.  **Scene Graph Generation**:
    *   **Visual Analysis**: Renders the scene from multiple views (e.g., every 10th training view).
    *   **Segmentation**: Uses **SAM (Segment Anything Model)** and LangSplat language features to segment objects in 2D views.
    *   **Object Clustering**: Aggregates 2D detections into unique 3D objects using density-based clustering (DBSCAN) on the language features.
    *   **Physics Prediction**: Uses a VLM (e.g., Gemini via OpenRouter) to analyze the visual appearance of each object and predict physics properties (mass, friction, elasticity, material).
    *   **Hierarchy Construction**: Builds a hierarchical relationship between objects (e.g., "apple" inside "bowl") based on spatial containment.

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
./scripts/run_full_pipeline_local.sh data/crate1
```

### Running SplatGraph Only

If you already have a trained LangSplat model, you can run the scene graph generation directly:

```bash
python SplatGraph/pipeline.py \
  --dataset_path data/crate1 \
  --model_path data/crate1/langsplat_output/crate1_0_3 \
  --output_dir results/crate1_graph \
  --openrouter_key YOUR_OPENROUTER_KEY
```

## Output

The pipeline produces a `scene_graph.json` file containing the hierarchical scene graph:

```json
{
  "objects": [
    {
      "id": 0,
      "name": "wooden_crate",
      "physics": {
        "material": "wood",
        "mass_kg": 5.0,
        "friction": 0.6,
        "elasticity": 0.2
      },
      "children": [
        {
          "id": 1,
          "name": "apple",
          ...
        }
      ]
    }
  ]
}
```
