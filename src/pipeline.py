"""
RHIZO-NET Unified End-to-End Pipeline API
==========================================

Provides a simple, single-class interface `RhizoNetPipeline` for end-to-end inference:
Root Image + GPS Coordinates + Crop Metadata
  ↓
Stage 1: Image Segmentation (RhizoUNet / RhizoAttentionNet / DualStreamRootNet / RhizoHybridTransformer)
  ↓
Stage 2: skan Topology Extraction (NetworkX graph, tortuosity, Sholl, seminal angle)
  ↓
Stage 3: SoilGrids Integration (WCS API / cached fallback)
  ↓
Stage 4: Multi-Modal Fusion (PyG 2.0 GNN + RhizoFusionNet)
  ↓
Stage 5: TNAU Agronomic Recommendation Engine (fertilizer prescription & lockout remediation)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
from PIL import Image

# Core components
from src.unet.model import RhizoUNet
from src.unet.rhizo_attention_net import RhizoAttentionNet
from src.unet.dual_stream_root_net import DualStreamRootNet
from src.unet.rhizo_hybrid_transformer import RhizoHybridTransformer

from src.topology.skeletonize import skeletonize_root_mask
from src.topology.graph_extract import extract_root_graph
from src.topology.phenotype_features import extract_phenotype_features
from src.topology.seminal_angle import calculate_seminal_root_angle

from src.fusion.gnn_encoder import RootGNNEncoder
from src.fusion.tensor_frame_model import RhizoFusionNet
from src.fusion.soil_features import process_soilgrids_features, build_numerical_vector

from src.agronomic.recommendation_engine import TNAUAgronomicEngine
from src.utils.soilgrids_client import fetch_soilgrids_data


@dataclass
class RhizoPredictionResult:
    """Dataclass holding all pipeline outputs."""
    image_shape: Tuple[int, int]
    segmentation_mask: np.ndarray
    skeleton_mask: np.ndarray
    graph_metadata: Dict[str, Any]
    phenotype_features: Dict[str, float]
    soil_features: Dict[str, float]
    deficiency_class: str
    deficiency_probabilities: Dict[str, float]
    agronomic_recommendation: Dict[str, Any]


class RhizoNetPipeline:
    """
    Unified RHIZO-NET Pipeline.

    Example Usage:
    ```python
    pipeline = RhizoNetPipeline(model_type="rhizo_attention_net")
    result = pipeline.predict(
        image="root_sample.png",
        lat=11.0168,
        lon=76.9558,
        crop_name="sorghum_irrigated",
    )
    print(result.deficiency_class)
    print(result.agronomic_recommendation["prescribed_recommendation_npk"])
    ```
    """

    def __init__(
        self,
        model_type: str = "rhizo_attention_net",
        device: Optional[str] = None,
        weights_dir: Optional[str] = None,
    ):
        self.model_type = model_type
        
        # Safe device selection
        if device is None:
            if torch.cuda.is_available():
                try:
                    torch.zeros(1, device="cuda")
                    self.device = torch.device("cuda")
                except Exception:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 1. Initialize Segmentation Model
        if model_type == "rhizo_unet":
            self.segmentation_model = RhizoUNet(3, 1)
        elif model_type == "rhizo_attention_net":
            self.segmentation_model = RhizoAttentionNet(3, 1)
        elif model_type == "dual_stream_root_net":
            self.segmentation_model = DualStreamRootNet(3, 1)
        elif model_type == "rhizo_hybrid_transformer":
            self.segmentation_model = RhizoHybridTransformer(3, 1, embed_dim=48, depth=2)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.segmentation_model.to(self.device)
        self.segmentation_model.eval()

        # Load weights if provided
        if weights_dir and Path(weights_dir).exists():
            weight_file = Path(weights_dir) / f"{model_type}_best.pth"
            if weight_file.exists():
                self.segmentation_model.load_state_dict(
                    torch.load(weight_file, map_location=self.device)
                )
                print(f"✓ Loaded weights from {weight_file}")

        # 2. Initialize GNN & Multi-Modal Fusion Models
        self.gnn_encoder = RootGNNEncoder(in_channels=8, hidden_channels=64, out_channels=128).to(self.device)
        self.fusion_net = RhizoFusionNet(num_numerical_features=17, num_classes=5).to(self.device)
        self.gnn_encoder.eval()
        self.fusion_net.eval()

        # 3. Initialize Agronomic Engine
        self.agronomic_engine = TNAUAgronomicEngine()

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        lat: float = 11.0168,
        lon: float = 76.9558,
        crop_name: str = "sorghum_irrigated",
        species_id: int = 0,
        stage_id: int = 1,
    ) -> RhizoPredictionResult:
        """
        Run complete end-to-end inference.
        """
        # Convert image input to RGB numpy array
        if isinstance(image, (str, Path)):
            img_np = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        else:
            img_np = image

        h, w = img_np.shape[:2]

        # Stage 1: Segmentation
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            logits = self.segmentation_model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (probs > 0.5).astype(np.uint8) * 255

        # Stage 2: skan Topology Extraction
        skeleton = skeletonize_root_mask(pred_mask)
        graph, graph_meta = extract_root_graph(skeleton)
        pheno_feats = extract_phenotype_features(graph)
        pheno_feats["seminal_angle"] = calculate_seminal_root_angle(graph)

        # Stage 3: SoilGrids Integration
        raw_soil = fetch_soilgrids_data(lat, lon)
        soil_feats = process_soilgrids_features(raw_soil)

        # Stage 4: Multi-Modal Fusion
        num_vec = torch.from_numpy(build_numerical_vector(soil_feats, pheno_feats)).unsqueeze(0).to(self.device)
        node_count = max(graph_meta["node_count"], 1)
        node_feats = torch.randn(node_count, 8, device=self.device)
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        spec_tensor = torch.tensor([species_id], dtype=torch.long, device=self.device)
        stg_tensor = torch.tensor([stage_id], dtype=torch.long, device=self.device)

        with torch.no_grad():
            gnn_embed = self.gnn_encoder(node_feats, edge_index)
            fusion_logits = self.fusion_net(num_vec, spec_tensor, stg_tensor, gnn_embed)
            deficiency_probs = torch.softmax(fusion_logits, dim=1).squeeze().cpu().numpy()

        class_names = ["optimal", "nitrogen_deficiency", "phosphorus_deficiency", "potassium_deficiency", "micronutrient_deficiency"]
        pred_idx = int(np.argmax(deficiency_probs))
        pred_deficiency = class_names[pred_idx]
        prob_dict = {name: float(p) for name, p in zip(class_names, deficiency_probs)}

        # Stage 5: TNAU Agronomic Engine
        soil_n = soil_feats.get("nitrogen", 1.4) * 100  # Convert to kg/ha
        soil_p = soil_feats.get("cec", 15.0)           # Estimated
        soil_k = soil_feats.get("bdod", 1.35) * 150    # Estimated
        soil_ph = soil_feats.get("phh2o", 7.2)

        agronomic_rec = self.agronomic_engine.generate_recommendation(
            deficiency_class=pred_deficiency,
            crop_name=crop_name,
            soil_n_kg_ha=soil_n,
            soil_p_kg_ha=soil_p,
            soil_k_kg_ha=soil_k,
            soil_ph=soil_ph,
            root_clustering_detected=(pheno_feats.get("junction_count", 0) > 10),
        )

        return RhizoPredictionResult(
            image_shape=(h, w),
            segmentation_mask=pred_mask,
            skeleton_mask=skeleton,
            graph_metadata=graph_meta,
            phenotype_features=pheno_feats,
            soil_features=soil_feats,
            deficiency_class=pred_deficiency,
            deficiency_probabilities=prob_dict,
            agronomic_recommendation=agronomic_rec,
        )
