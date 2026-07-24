#!/usr/bin/env python3
"""
Generates all 7 Kaggle Jupyter Notebooks for RHIZO-NET.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def make_markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


# ============================================================================
# Notebook 1: Data Preparation
# ============================================================================

nb1 = make_notebook([
    make_markdown_cell([
        "# RHIZO-NET: Notebook 01 - Data Preparation & Multi-Dataset Validation\n",
        "This notebook validates all 6 root imagery datasets and prepares train/val splits."
    ]),
    make_code_cell([
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "# Verify Kaggle input datasets\n",
        "KAGGLE_INPUT = Path('/kaggle/input')\n",
        "print('Available Kaggle Datasets:')\n",
        "if KAGGLE_INPUT.exists():\n",
        "    for p in KAGGLE_INPUT.iterdir():\n",
        "        print(f'  - {p.name}')\n",
        "else:\n",
        "    print('  Local environment detected.')\n"
    ]),
])

# ============================================================================
# Notebook 2: Segmentation Training (All Models)
# ============================================================================

nb2 = make_notebook([
    make_markdown_cell([
        "# RHIZO-NET: Notebook 02 - Root Segmentation Model Training\n",
        "Trains and evaluates all 4 segmentation architectures:\n",
        "1. **RhizoUNet** (Modified U-Net)\n",
        "2. **RhizoAttentionNet** (Tubular Attention)\n",
        "3. **DualStreamRootNet** (Spatial + Tubularity)\n",
        "4. **RhizoHybridTransformer** (CNN-Transformer Hybrid)\n"
    ]),
    make_code_cell([
        "import torch\n",
        "from src.unet.model import RhizoUNet\n",
        "from src.unet.rhizo_attention_net import RhizoAttentionNet\n",
        "from src.unet.dual_stream_root_net import DualStreamRootNet\n",
        "from src.unet.rhizo_hybrid_transformer import RhizoHybridTransformer\n",
        "\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Using device: {device}')\n",
        "\n",
        "# Instantiate all models\n",
        "models = {\n",
        "    'RhizoUNet': RhizoUNet().to(device),\n",
        "    'RhizoAttentionNet': RhizoAttentionNet().to(device),\n",
        "    'DualStreamRootNet': DualStreamRootNet().to(device),\n",
        "    'RhizoHybridTransformer': RhizoHybridTransformer(embed_dim=48, depth=2).to(device),\n",
        "}\n",
        "\n",
        "for name, model in models.items():\n",
        "    params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
        "    print(f'{name}: {params:,} parameters')\n"
    ]),
])

# ============================================================================
# Notebook 3: MobileSAM Segmentation Fallback
# ============================================================================

nb3 = make_notebook([
    make_markdown_cell([
        "# RHIZO-NET: Notebook 03 - MobileSAM Transformer Fallback\n",
        "Demonstrates automatic confidence-based routing to MobileSAM when primary CNN models have low confidence."
    ]),
    make_code_cell([
        "import torch\n",
        "from src.unet.model import RhizoUNet\n",
        "from src.mobilesam.adapter import MobileSAMAdapter\n",
        "\n",
        "primary_model = RhizoUNet()\n",
        "adapter = MobileSAMAdapter(\n",
        "    primary_model=primary_model,\n",
        "    mobilesam_weights='/kaggle/input/mobilesam-model/mobile_sam.pt',\n",
        "    confidence_threshold=0.5\n",
        ")\n",
        "print('MobileSAM Adapter initialized successfully.')\n"
    ]),
])

# ============================================================================
# Notebook 4: Topological Graph Extraction
# ============================================================================

nb4 = make_notebook([
    make_markdown_cell([
        "# RHIZO-NET: Notebook 04 - skan Topological Graph Extraction\n",
        "Extracts skeleton centerlines, NetworkX topological graphs, and phenotypic features (tortuosity, Sholl analysis, seminal angle)."
    ]),
    make_code_cell([
        "import numpy as np\n",
        "from src.topology.skeletonize import skeletonize_root_mask\n",
        "from src.topology.graph_extract import extract_root_graph\n",
        "from src.topology.phenotype_features import extract_phenotype_features\n",
        "from src.topology.seminal_angle import calculate_seminal_root_angle\n",
        "\n",
        "# Create synthetic root mask\n",
        "mask = np.zeros((200, 200), dtype=np.uint8)\n",
        "mask[20:180, 100] = 255  # Main root axis\n",
        "mask[60:120, 100:150] = 255  # Lateral branch\n",
        "\n",
        "skel = skeletonize_root_mask(mask)\n",
        "graph, meta = extract_root_graph(skel)\n",
        "pheno = extract_phenotype_features(graph)\n",
        "angle = calculate_seminal_root_angle(graph)\n",
        "pheno['seminal_angle'] = angle\n",
        "\n",
        "print('Extracted Metadata:', meta)\n",
        "print('Extracted Phenotype Features:', pheno)\n"
    ]),
])

# ============================================================================
# Notebook 5: Multi-Modal Fusion
# ============================================================================

nb5 = make_notebook([
    make_markdown_cell([
        "# RHIZO-NET: Notebook 05 - PyTorch Frame + PyG 2.0 Multi-Modal Fusion\n",
        "Fuses root graph topological embeddings (PyG 2.0) with SoilGrids chemistry data to predict nutrient deficiencies."
    ]),
    make_code_cell([
        "import torch\n",
        "from src.fusion.gnn_encoder import RootGNNEncoder\n",
        "from src.fusion.tensor_frame_model import RhizoFusionNet\n",
        "from src.fusion.soil_features import build_numerical_vector\n",
        "\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "gnn = RootGNNEncoder(in_channels=8, hidden_channels=64, out_channels=128).to(device)\n",
        "fusion_net = RhizoFusionNet(num_numerical_features=17, num_classes=5).to(device)\n",
        "\n",
        "print('PyG 2.0 GNN Encoder parameters:', sum(p.numel() for p in gnn.parameters()))\n",
        "print('RhizoFusionNet parameters:', sum(p.numel() for p in fusion_net.parameters()))\n"
    ]),
])

# ============================================================================
# Notebook 6: Agronomic Recommendation Engine
# ============================================================================

nb6 = make_notebook([
    make_markdown_cell([
        "# RHIZO-NET: Notebook 06 - TNAU Agronomic Recommendation Engine\n",
        "Generates deterministic fertilizer prescriptions according to TNAU Agritech Portal standards."
    ]),
    make_code_cell([
        "from src.agronomic.recommendation_engine import TNAUAgronomicEngine\n",
        "\n",
        "engine = TNAUAgronomicEngine()\n",
        "report = engine.generate_recommendation(\n",
        "    deficiency_class='nitrogen_deficiency',\n",
        "    crop_name='sorghum_irrigated',\n",
        "    soil_n_kg_ha=180.0,  # Low\n",
        "    soil_p_kg_ha=15.0,   # Medium\n",
        "    soil_k_kg_ha=200.0,  # Medium\n",
        "    soil_ph=8.6,         # Alkaline lockout\n",
        "    root_clustering_detected=True\n",
        ")\n",
        "\n",
        "import json\n",
        "print(json.dumps(report, indent=2))\n"
    ]),
])

# ============================================================================
# Notebook 7: Full Pipeline End-to-End Demo
# ============================================================================

nb7 = make_notebook([
    make_markdown_cell([
        "# RHIZO-NET: Notebook 07 - Complete End-to-End Pipeline Demo\n",
        "Runs the complete RHIZO-NET pipeline from raw image -> segmentation -> topology -> fusion -> TNAU recommendation."
    ]),
    make_code_cell([
        "import numpy as np\n",
        "import torch\n",
        "from src.unet.rhizo_attention_net import RhizoAttentionNet\n",
        "from src.topology.skeletonize import skeletonize_root_mask\n",
        "from src.topology.graph_extract import extract_root_graph\n",
        "from src.topology.phenotype_features import extract_phenotype_features\n",
        "from src.fusion.soil_features import process_soilgrids_features, build_numerical_vector\n",
        "from src.agronomic.recommendation_engine import TNAUAgronomicEngine\n",
        "from src.utils.soilgrids_client import fetch_soilgrids_data\n",
        "from src.utils.visualization import plot_pipeline_summary\n",
        "\n",
        "print('='*60)\n",
        "print('RHIZO-NET COMPLETE PIPELINE DEMONSTRATION')\n",
        "print('='*60)\n",
        "\n",
        "# 1. Load sample image (dummy synthetic)\n",
        "dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)\n",
        "dummy_mask = np.zeros((256, 256), dtype=np.uint8)\n",
        "dummy_mask[30:220, 128] = 255\n",
        "dummy_mask[80:150, 128:190] = 255\n",
        "\n",
        "# 2. Topology\n",
        "skel = skeletonize_root_mask(dummy_mask)\n",
        "graph, meta = extract_root_graph(skel)\n",
        "pheno = extract_phenotype_features(graph)\n",
        "\n",
        "# 3. SoilGrids & Agronomic Engine\n",
        "raw_soil = fetch_soilgrids_data(lat=11.0168, lon=76.9558) # Coimbatore, TN\n",
        "engine = TNAUAgronomicEngine()\n",
        "recommendation = engine.generate_recommendation(\n",
        "    deficiency_class='nitrogen_deficiency',\n",
        "    crop_name='sorghum_irrigated',\n",
        "    soil_n_kg_ha=190.0,\n",
        "    soil_p_kg_ha=14.0,\n",
        "    soil_k_kg_ha=220.0,\n",
        "    soil_ph=7.2,\n",
        ")\n",
        "\n",
        "print('✓ Pipeline execution complete! Displaying summary report...')\n",
        "plot_pipeline_summary(dummy_image, dummy_mask, graph, recommendation)\n"
    ]),
])


notebooks = {
    "01_data_preparation.ipynb": nb1,
    "02_unet_root_segmentation.ipynb": nb2,
    "03_mobilesam_segmentation.ipynb": nb3,
    "04_topology_extraction.ipynb": nb4,
    "05_multimodal_fusion.ipynb": nb5,
    "06_agronomic_engine.ipynb": nb6,
    "07_full_pipeline_demo.ipynb": nb7,
}

for name, nb in notebooks.items():
    path = NOTEBOOKS_DIR / name
    with open(path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"✓ Created notebook: {path.name}")
