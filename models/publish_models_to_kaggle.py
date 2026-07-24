#!/usr/bin/env python3
"""
Publish trained RHIZO-NET models to Kaggle as public models.
Run this AFTER successful training to share models with the community.

Publishes:
  1. RhizoUNet - Modified U-Net for root segmentation
  2. RhizoFusionNet - Multi-modal PyTorch Frame + PyG fusion classifier
  3. RhizoNet-Full - Complete pipeline weights bundle

Usage:
    python publish_models_to_kaggle.py --weights-dir ./outputs/weights
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path


KAGGLE_USERNAME = "saranboddu"

MODELS_TO_PUBLISH = {
    "rhizo-unet": {
        "title": "RhizoUNet - Root Segmentation Model",
        "slug": "rhizo-unet-root-segmentation",
        "subtitle": "Modified U-Net (ELU/AvgPool/Residual Skip) trained on 6 root imagery datasets",
        "description": (
            "RhizoUNet is a lightweight modified U-Net architecture specifically designed for "
            "plant root segmentation from rhizotron and minirhizotron imagery.\n\n"
            "## Architecture Modifications\n"
            "- Feature maps reduced by 4x (16→4, 32→8, 64→16, 128→32, 256→64)\n"
            "- ELU activation (prevents dying ReLU on thin root structures)\n"
            "- Average pooling (preserves fine single-pixel roots)\n"
            "- Residual skip connections (element-wise sum instead of concatenation)\n\n"
            "## Training Data\n"
            "Trained on 6 diverse root imagery datasets:\n"
            "1. RootNav 2.0 (Wheat)\n"
            "2. PRMI (Peanut/Cotton/Switchgrass/Papaya/Sesame/Sunflower)\n"
            "3. DeepRootLab (11 Herbaceous Species)\n"
            "4. SeminalRootAngle (Spring Barley)\n"
            "5. Chicory Subset\n"
            "6. Grassland (Alpine Minirhizotron)\n\n"
            "## Usage\n"
            "```python\n"
            "import torch\n"
            "from rhizonet.unet.model import RhizoUNet\n\n"
            "model = RhizoUNet(in_channels=3, out_channels=1)\n"
            "model.load_state_dict(torch.load('rhizo_unet_best.pth'))\n"
            "model.eval()\n"
            "```\n\n"
            "Part of RHIZO-NET: Root Health and Integrated Zonal Optimization Network via Edaphic Topology"
        ),
        "weight_files": ["rhizo_unet_best.pth", "rhizo_unet_ensemble_*.pth"],
        "framework": "pytorch",
        "task": "image-segmentation",
    },
    "rhizo-fusionnet": {
        "title": "RhizoFusionNet - Multi-Modal Root Health Classifier",
        "slug": "rhizo-fusionnet-multimodal",
        "subtitle": "PyTorch Frame + PyG 2.0 multi-modal fusion for nutrient deficiency detection",
        "description": (
            "RhizoFusionNet is a novel multi-modal deep learning model that fuses:\n"
            "- Root topology graph embeddings (via GNN/PyG 2.0)\n"
            "- Soil chemistry vectors (from ISRIC SoilGrids)\n"
            "- Crop metadata (species, growth stage, planting date)\n\n"
            "## Output Classes\n"
            "1. Optimal (healthy)\n"
            "2. Nitrogen Deficiency\n"
            "3. Phosphorus Deficiency\n"
            "4. Potassium Deficiency\n"
            "5. Micronutrient (Zn/Fe) Deficiency\n\n"
            "## Architecture\n"
            "- GNN Encoder: Graph Attention Network (GAT) with mean pooling\n"
            "- Tabular Fusion: PyTorch Frame TensorFrame with cross-modal attention\n"
            "- Loss: Focal Loss for imbalanced deficiency states\n\n"
            "Part of RHIZO-NET: Root Health and Integrated Zonal Optimization Network via Edaphic Topology"
        ),
        "weight_files": ["rhizo_fusionnet_best.pth", "gnn_encoder_best.pth"],
        "framework": "pytorch",
        "task": "tabular-classification",
    },
    "rhizo-net-full": {
        "title": "RHIZO-NET Complete Pipeline Weights",
        "slug": "rhizo-net-full-pipeline",
        "subtitle": "All trained weights for the complete RHIZO-NET inference pipeline",
        "description": (
            "Complete bundle of all trained model weights for the RHIZO-NET pipeline:\n\n"
            "1. **RhizoUNet** - Root segmentation model\n"
            "2. **MobileSAM Adapter** - Transformer fallback for complex imagery\n"
            "3. **GNN Encoder** - Root topology graph embeddings\n"
            "4. **RhizoFusionNet** - Multi-modal nutrient deficiency classifier\n\n"
            "## Quick Start\n"
            "```python\n"
            "from rhizonet.pipeline import RhizoNetPipeline\n\n"
            "pipeline = RhizoNetPipeline.from_pretrained('/kaggle/input/rhizo-net-full-pipeline')\n"
            "result = pipeline.predict(image_path='root.png', lat=11.0, lon=76.9, crop='sorghum')\n"
            "print(result.deficiency_class)\n"
            "print(result.fertilizer_recommendation)\n"
            "```\n\n"
            "Part of RHIZO-NET: Root Health and Integrated Zonal Optimization Network via Edaphic Topology"
        ),
        "weight_files": [
            "rhizo_unet_best.pth",
            "rhizo_fusionnet_best.pth",
            "gnn_encoder_best.pth",
            "config.json",
        ],
        "framework": "pytorch",
        "task": "other",
    },
}


def create_model_metadata(model_dir, model_info, username):
    """Create dataset-metadata.json for Kaggle model upload."""
    metadata = {
        "title": model_info["title"],
        "id": f"{username}/{model_info['slug']}",
        "subtitle": model_info["subtitle"],
        "description": model_info["description"],
        "isPrivate": False,  # Public models!
        "licenses": [{"name": "Apache 2.0"}],
        "keywords": [
            "rhizo-net",
            "root-segmentation",
            "plant-phenotyping",
            "precision-agriculture",
            "pytorch",
            "deep-learning",
            "nutrient-deficiency",
            "soil-analysis",
        ],
    }

    metadata_path = model_dir / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def create_model_card(model_dir, model_info):
    """Create a README model card."""
    card = f"""# {model_info['title']}

{model_info['subtitle']}

## Description

{model_info['description']}

## Framework

- **Framework**: PyTorch
- **Task**: {model_info['task']}

## Citation

If you use this model, please cite:

```bibtex
@software{{rhizo_net_2026,
    title = {{RHIZO-NET: Root Health and Integrated Zonal Optimization Network via Edaphic Topology}},
    author = {{Saran Boddu}},
    year = {{2026}},
    url = {{https://kaggle.com/models/{KAGGLE_USERNAME}/{model_info['slug']}}}
}}
```

## License

Apache 2.0
"""
    readme_path = model_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(card)


def publish_model(model_key, model_info, weights_dir, username):
    """Package and publish a model to Kaggle."""
    print(f"\n{'─' * 50}")
    print(f"📤 Publishing: {model_info['title']}")
    print(f"   Slug: {username}/{model_info['slug']}")
    print(f"{'─' * 50}")

    # Create staging directory
    staging_dir = Path(f"/tmp/rhizonet_publish/{model_key}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # Copy weight files
    import glob
    files_found = 0
    for pattern in model_info["weight_files"]:
        matches = glob.glob(str(weights_dir / pattern))
        for match in matches:
            src = Path(match)
            dst = staging_dir / src.name
            shutil.copy2(src, dst)
            print(f"  ✓ Copied: {src.name} ({src.stat().st_size / (1024*1024):.1f} MB)")
            files_found += 1

    if files_found == 0:
        print(f"  ⚠ No weight files found for {model_key}. Skipping.")
        print(f"    Expected files in {weights_dir}:")
        for pattern in model_info["weight_files"]:
            print(f"      - {pattern}")
        return False

    # Create metadata and model card
    create_model_metadata(staging_dir, model_info, username)
    create_model_card(staging_dir, model_info)

    # Upload to Kaggle
    cmd = f"kaggle datasets create -p {staging_dir} --dir-mode zip"
    print(f"  Running: {cmd}")
    exit_code = os.system(cmd)

    if exit_code == 0:
        print(f"  ✓ Published successfully!")
        print(f"  🌐 View at: https://kaggle.com/datasets/{username}/{model_info['slug']}")
        return True
    else:
        # Try as version update
        cmd_update = f"kaggle datasets version -p {staging_dir} -m 'Model update' --dir-mode zip"
        print(f"  Trying version update...")
        exit_code = os.system(cmd_update)
        if exit_code == 0:
            print(f"  ✓ Updated successfully!")
            return True
        else:
            print(f"  ✗ Upload failed")
            return False


def main():
    parser = argparse.ArgumentParser(description="Publish RHIZO-NET models to Kaggle")
    parser.add_argument("--weights-dir", type=str, default="./outputs/weights",
                        help="Directory containing trained model weights")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Specific models to publish (default: all)")
    parser.add_argument("--username", type=str, default=KAGGLE_USERNAME)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)

    # Setup Kaggle credentials
    kaggle_json = Path(__file__).parent.parent / "kaggle.json"
    if kaggle_json.exists():
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_dest = kaggle_dir / "kaggle.json"
        if not kaggle_dest.exists():
            shutil.copy2(kaggle_json, kaggle_dest)
            os.chmod(kaggle_dest, 0o600)

    model_keys = args.models or list(MODELS_TO_PUBLISH.keys())

    print("=" * 60)
    print("RHIZO-NET Model Publisher")
    print("=" * 60)

    results = {}
    for key in model_keys:
        if key not in MODELS_TO_PUBLISH:
            print(f"\n⚠ Unknown model: {key}")
            continue
        success = publish_model(key, MODELS_TO_PUBLISH[key], weights_dir, args.username)
        results[key] = "published" if success else "failed"

    print(f"\n{'=' * 60}")
    print("PUBLISH SUMMARY")
    print(f"{'=' * 60}")
    for key, status in results.items():
        emoji = "✓" if status == "published" else "✗"
        print(f"  {emoji} {key}: {status}")


if __name__ == "__main__":
    main()
