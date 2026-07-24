"""
Multi-Modal Fusion Network (PyTorch Frame + GNN Embeddings)
============================================================

Fuses heterogeneous inputs:
1. Root Graph Embeddings (from PyG 2.0 GNN)
2. Phenotypic Numerical Vector (tortuosity, Sholl, seminal angle)
3. SoilGrids Chemistry Features (bdod, cec, clay, N, SOC, pH, wv0033)
4. Crop Metadata Embeddings (species, growth stage)

Outputs class probabilities across 5 nutrient states:
0: Optimal
1: Nitrogen Deficiency
2: Phosphorus Deficiency
3: Potassium Deficiency
4: Micronutrient (Zn/Fe) Deficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


class CrossModalAttention(nn.Module):
    """Attention mechanism between root topology and soil features."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, root_embed, soil_embed):
        q = self.query(root_embed).unsqueeze(1)  # [B, 1, D]
        k = self.key(soil_embed).unsqueeze(1)    # [B, 1, D]
        v = self.value(soil_embed).unsqueeze(1)  # [B, 1, D]

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).squeeze(1)
        return root_embed + out


class RhizoFusionNet(nn.Module):
    """
    RhizoFusionNet: Multi-Modal Nutrient Deficiency Classifier.

    Args:
        num_numerical_features: Number of soil + phenotype continuous features
        num_classes: Number of target deficiency states (default: 5)
        hidden_dim: Multi-modal fusion dimension
    """

    def __init__(
        self,
        num_numerical_features: int = 17,
        num_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Feature Encoders
        self.num_encoder = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
        )

        # Categorical Embeddings (species & growth stage)
        self.species_embed = nn.Embedding(30, 32)
        self.stage_embed = nn.Embedding(10, 16)

        # GNN embedding projection (128 -> 128)
        self.gnn_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
        )

        # Cross-modal attention
        self.cross_attn = CrossModalAttention(embed_dim=128)

        # Multi-Modal Fusion MLP
        # Combined dim: 128 (num) + 32 (species) + 16 (stage) + 128 (gnn) = 304
        self.fusion_mlp = nn.Sequential(
            nn.Linear(304, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        # Classification Head
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(
        self,
        numerical_feats: torch.Tensor,
        species_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        gnn_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            numerical_feats: Tensor [B, 17] (SoilGrids + phenotype stats)
            species_ids: LongTensor [B]
            stage_ids: LongTensor [B]
            gnn_embeddings: Tensor [B, 128]

        Returns:
            Logits [B, 5]
        """
        # Encode numerical
        num_enc = self.num_encoder(numerical_feats)  # [B, 128]

        # Encode categoricals
        spec_enc = self.species_embed(species_ids)   # [B, 32]
        stg_enc = self.stage_embed(stage_ids)       # [B, 16]

        # Encode GNN
        gnn_enc = self.gnn_proj(gnn_embeddings)    # [B, 128]

        # Cross-modal attention between GNN and numerical soil features
        attended_gnn = self.cross_attn(gnn_enc, num_enc)

        # Concatenate all modalities
        fused = torch.cat([num_enc, spec_enc, stg_enc, attended_gnn], dim=1)  # [B, 304]

        # Fusion MLP
        hidden = self.fusion_mlp(fused)

        # Classifier
        return self.classifier(hidden)
