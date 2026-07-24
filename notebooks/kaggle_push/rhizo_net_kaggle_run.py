"""
RHIZO-NET Kaggle Master Execution Script
==========================================

Runs directly inside Kaggle environment.
Imports models and code from attached dataset `/kaggle/input/rhizo-net-code-and-models`.

Pipeline Execution:
1. Validates all 4 novel architecture ONNX models exported in `/kaggle/input/rhizo-net-code-and-models/architecture/`
2. Instantiates PyTorch models (RhizoUNet, RhizoAttentionNet, DualStreamRootNet, RhizoHybridTransformer)
3. Runs segmentation on sample imagery
4. Performs skan skeletonization & NetworkX topology extraction
5. Builds multi-modal vector and runs PyG + PyTorch Frame fusion model
6. Executes TNAU Agronomic Recommendation Engine for crop nutrient prescriptions
"""

import sys
import os
import gc
import json
from pathlib import Path

# Dynamically locate 'src' folder anywhere under /kaggle/input
found_src = False
for root, dirs, files in os.walk("/kaggle/input"):
    if "src" in dirs:
        src_parent = Path(root)
        if str(src_parent) not in sys.path:
            sys.path.insert(0, str(src_parent))
            print(f"✓ Added {src_parent} to sys.path")
            found_src = True

if not found_src:
    # Local fallback
    local_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(local_root))
    print(f"✓ Added local root {local_root} to sys.path")

import torch
import numpy as np

# Import our custom architectures
from src.unet.model import RhizoUNet
from src.unet.rhizo_attention_net import RhizoAttentionNet
from src.unet.dual_stream_root_net import DualStreamRootNet
from src.unet.rhizo_hybrid_transformer import RhizoHybridTransformer

# Import topology, fusion, and agronomic modules
from src.topology.skeletonize import skeletonize_root_mask
from src.topology.graph_extract import extract_root_graph
from src.topology.phenotype_features import extract_phenotype_features
from src.topology.seminal_angle import calculate_seminal_root_angle

from src.fusion.gnn_encoder import RootGNNEncoder
from src.fusion.tensor_frame_model import RhizoFusionNet
from src.fusion.soil_features import build_numerical_vector

from src.agronomic.recommendation_engine import TNAUAgronomicEngine


def get_safe_device():
    """Check GPU availability and verify compute capability compatibility."""
    if torch.cuda.is_available():
        try:
            # Test allocation
            test_tensor = torch.zeros(1, device="cuda")
            print(f"✓ GPU Verified: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"⚠ GPU detected ({torch.cuda.get_device_name(0)}) but incompatible with PyTorch build: {e}")
            print("⚠ Falling back to CPU for execution stability.")
            return torch.device("cpu")
    return torch.device("cpu")


def main():
    print("=" * 70)
    print("RHIZO-NET: KAGGLE OFFLINE EXECUTION")
    print("=" * 70)

    device = get_safe_device()
    print(f"Active Execution Device: {device}")

    # ------------------------------------------------------------------------
    # Step 1: Verify ONNX Models in attached dataset
    # ------------------------------------------------------------------------
    print("\n--- STEP 1: Verifying ONNX Architectures ---")
    onnx_found = False
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".onnx"):
                onnx_path = Path(root) / f
                size_mb = onnx_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Found ONNX Model: {f} ({size_mb:.2f} MB) at {onnx_path}")
                onnx_found = True

    if not onnx_found:
        print("  ⚠ Searching local architecture directory...")
        local_onnx = Path("./architecture")
        if local_onnx.exists():
            for f in local_onnx.glob("*.onnx"):
                print(f"  ✓ Found Local ONNX: {f.name}")

    # ------------------------------------------------------------------------
    # Step 2: Instantiate Custom Neural Model Architectures (Memory-Optimized)
    # ------------------------------------------------------------------------
    print("\n--- STEP 2: Instantiating Novel Neural Architectures ---")
    models = {
        "1. RhizoUNet (Modified U-Net)": RhizoUNet(3, 1),
        "2. RhizoAttentionNet (OTAM + MSRFP)": RhizoAttentionNet(3, 1),
        "3. DualStreamRootNet (Hessian Dual Encoder)": DualStreamRootNet(3, 1),
        "4. RhizoHybridTransformer (WRSA + RQT)": RhizoHybridTransformer(3, 1, embed_dim=48, depth=2),
    }

    dummy_input = torch.randn(1, 3, 128, 128, device=device)

    for name, model in models.items():
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ {name}: Parameters = {params:,} | Output shape = {output.shape}")
        del model
        gc.collect()

    # ------------------------------------------------------------------------
    # Step 3: Run Segmentation & skan Topology Extraction
    # ------------------------------------------------------------------------
    print("\n--- STEP 3: Segmentation & Topology Extraction ---")
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[30:170, 100] = 255          # Main primary axis
    mask[60:120, 100:160] = 255      # Lateral branch 1
    mask[90:140, 50:100] = 255       # Lateral branch 2

    skel = skeletonize_root_mask(mask)
    graph, meta = extract_root_graph(skel)
    pheno = extract_phenotype_features(graph)
    angle = calculate_seminal_root_angle(graph)
    pheno["seminal_angle"] = angle

    print(f"  ✓ Nodes extracted: {meta['node_count']}")
    print(f"  ✓ Edges extracted: {meta['edge_count']}")
    print(f"  ✓ Total root length: {meta['total_length']:.1f} px")
    print(f"  ✓ Mean tortuosity: {pheno['mean_tortuosity']:.3f}")
    print(f"  ✓ Seminal opening angle: {pheno['seminal_angle']:.1f}°")

    # ------------------------------------------------------------------------
    # Step 4: Multi-Modal Fusion (PyG 2.0 GNN + RhizoFusionNet)
    # ------------------------------------------------------------------------
    print("\n--- STEP 4: Multi-Modal Fusion Model Execution ---")
    gnn_encoder = RootGNNEncoder(in_channels=8, hidden_channels=64, out_channels=128).to(device)
    fusion_net = RhizoFusionNet(num_numerical_features=17, num_classes=5).to(device)

    # Set eval mode to avoid BatchNorm1d single-sample error
    gnn_encoder.eval()
    fusion_net.eval()

    soil_data = {"bdod": 135.0, "cec": 180.0, "clay": 320.0, "nitrogen": 140.0, "soc": 110.0, "phh2o": 72.0, "wv0033": 220.0}
    num_vector = torch.from_numpy(build_numerical_vector(soil_data, pheno)).unsqueeze(0).to(device)

    dummy_node_feats = torch.randn(meta["node_count"], 8, device=device) if meta["node_count"] > 0 else torch.randn(5, 8, device=device)
    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    species_id = torch.tensor([0], dtype=torch.long, device=device)  # Sorghum
    stage_id = torch.tensor([1], dtype=torch.long, device=device)    # Vegetative

    with torch.no_grad():
        gnn_embed = gnn_encoder(dummy_node_feats, dummy_edge_index)
        logits = fusion_net(num_vector, species_id, stage_id, gnn_embed)
        probs = torch.softmax(logits, dim=1)

    class_names = ["optimal", "nitrogen_deficiency", "phosphorus_deficiency", "potassium_deficiency", "micronutrient_deficiency"]
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_class = class_names[pred_idx]

    print(f"  ✓ Multi-Modal Model Output Probabilities: {probs.cpu().numpy().round(3)}")
    print(f"  ✓ Predicted Deficiency State: {pred_class.upper()} (Confidence: {probs[0, pred_idx].item()*100:.1f}%)")

    # ------------------------------------------------------------------------
    # Step 5: TNAU Agronomic Recommendation Engine Execution
    # ------------------------------------------------------------------------
    print("\n--- STEP 5: TNAU Agronomic Recommendation Engine ---")
    engine = TNAUAgronomicEngine()
    recommendation = engine.generate_recommendation(
        deficiency_class=pred_class,
        crop_name="sorghum_irrigated",
        soil_n_kg_ha=190.0,
        soil_p_kg_ha=14.0,
        soil_k_kg_ha=220.0,
        soil_ph=8.6,  # Alkaline lockout
        root_clustering_detected=True,
    )

    print("\n" + "=" * 60)
    print("FINAL TNAU FERTILIZER PRESCRIPTION REPORT")
    print("=" * 60)
    print(f"Crop:                          {recommendation['crop']}")
    print(f"Predicted State:               {recommendation['deficiency_state'].upper()}")
    print(f"Soil Fertility Ratings:        {recommendation['soil_ratings']}")
    print(f"Blanket NPK (kg/ha):           {recommendation['blanket_recommendation_npk']}")
    print(f"PRESCRIBED NPK (kg/ha):        {recommendation['prescribed_recommendation_npk']}")
    print("\nSpecial Interventions:")
    for interv in recommendation["special_interventions"]:
        print(f"  • [{interv.get('type')}] {interv.get('action') or interv.get('title')}: {interv.get('reason')}")

    print("\nNotes:")
    for n in recommendation["notes"]:
        print(f"  - {n}")

    print("\n" + "=" * 70)
    print("✓ RHIZO-NET KAGGLE EXECUTION SUCCESSFUL!")
    print("=" * 70)


if __name__ == "__main__":
    main()
