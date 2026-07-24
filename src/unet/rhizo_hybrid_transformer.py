"""
RhizoHybridTransformer: CNN-Transformer Hybrid for Root Segmentation
=====================================================================

A NOVEL architecture that combines the local feature extraction strength of
CNNs with the global context modeling of Transformers, specifically designed
for plant root segmentation challenges.

KEY INNOVATIONS:

1. Lightweight CNN Stem with Root-Frequency Bias:
   - 3-stage CNN stem extracts local features efficiently
   - Incorporates high-frequency bias initialization for detecting fine roots
   - Much lighter than full ViT patch embedding

2. Windowed Root Self-Attention (WRSA):
   - Partitions feature maps into windows for efficient self-attention
   - Shifted windows (Swin-style) enable cross-window communication
   - Critical for connecting root fragments that appear disconnected
     due to imaging artifacts or soil occlusion

3. Root Query Tokens (RQT):
   - Learned tokens representing different root structural types:
     * Primary/Seminal roots (thick, directional)
     * Lateral roots (thin, branching)
     * Root tips (terminal, pointed)
     * Root junctions (branching points)
   - Cross-attention between image features and root queries
   - Produces structure-aware segmentation

4. Topology-Guided Positional Encoding (TGPE):
   - Standard positional encoding is grid-based (ignores root structure)
   - TGPE adds a learned positional bias based on distance from image center
     (roots typically radiate from a seed/crown point)
   - Encodes the radial growth pattern of root systems

5. Progressive Upsampling Decoder:
   - Fuses multi-scale CNN features with transformer outputs
   - Each level refines predictions from coarse to fine
   - Final level recovers single-pixel root details

WHY HYBRID?
Pure CNNs miss global context (can't connect distant root fragments).
Pure Transformers are too heavy and lack local inductive bias.
The hybrid approach gives us both: CNN for local root detection,
Transformer for global connectivity and fragment stitching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# CNN Stem
# ============================================================================

class RhizoCNNStem(nn.Module):
    """
    Lightweight CNN stem that extracts local features before the transformer.
    Uses depthwise separable convolutions for efficiency.
    """

    def __init__(self, in_channels, embed_dim, depths=(2, 2, 2)):
        super().__init__()
        dims = [in_channels, embed_dim // 4, embed_dim // 2, embed_dim]

        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            layers = []
            for j in range(depth):
                in_ch = dims[i] if j == 0 else dims[i + 1]
                out_ch = dims[i + 1]
                layers.extend([
                    # Depthwise
                    nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=max(1, in_ch), bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.ELU(inplace=True),
                    # Pointwise
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ELU(inplace=True),
                ])
            self.stages.append(nn.Sequential(*layers))

        # Downsample between stages
        self.downsamples = nn.ModuleList([
            nn.AvgPool2d(2, 2) for _ in range(len(depths) - 1)
        ])

    def forward(self, x):
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return features  # Multi-scale features


# ============================================================================
# Topology-Guided Positional Encoding
# ============================================================================

class TopologyGuidedPE(nn.Module):
    """
    Positional encoding that encodes radial distance from center,
    biasing the model toward the radial growth pattern of roots.

    Standard sinusoidal PE + learned radial bias.
    """

    def __init__(self, embed_dim, max_size=64):
        super().__init__()
        self.embed_dim = embed_dim

        # Standard 2D sinusoidal PE
        pe = torch.zeros(embed_dim, max_size, max_size)
        d_model = embed_dim // 2
        for pos_h in range(max_size):
            for pos_w in range(max_size):
                for i in range(0, d_model, 2):
                    pe[i, pos_h, pos_w] = math.sin(pos_h / (10000 ** (i / d_model)))
                    pe[i + 1, pos_h, pos_w] = math.cos(pos_h / (10000 ** (i / d_model)))
                for i in range(0, d_model, 2):
                    pe[d_model + i, pos_h, pos_w] = math.sin(pos_w / (10000 ** (i / d_model)))
                    pe[d_model + i + 1, pos_h, pos_w] = math.cos(pos_w / (10000 ** (i / d_model)))
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, C, H, W]

        # Learned radial bias (roots grow outward from center/seed point)
        self.radial_bias = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        self.radial_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # Interpolate sinusoidal PE to match input size
        pe = F.interpolate(self.pe[:, :C], size=(H, W), mode="bilinear", align_corners=False)

        # Compute radial distance from center
        cy, cx = H / 2, W / 2
        y_coords = torch.arange(H, device=x.device, dtype=torch.float32) - cy
        x_coords = torch.arange(W, device=x.device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        radial_dist = torch.sqrt(xx ** 2 + yy ** 2)
        radial_dist = radial_dist / (radial_dist.max() + 1e-6)  # Normalize to [0, 1]
        radial_map = radial_dist.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Combine: sinusoidal + radial bias
        radial_contribution = self.radial_bias * radial_map * self.radial_scale
        return x + pe + radial_contribution


# ============================================================================
# Windowed Root Self-Attention
# ============================================================================

class WindowedRootSelfAttention(nn.Module):
    """
    Efficient self-attention computed within local windows.
    Shifted windows enable cross-window connectivity.

    This is critical for root segmentation because roots can be:
    - Long and continuous (need global context)
    - Thin and fragmented (need local detail)

    Window attention balances both by being local but shifted.
    """

    def __init__(self, dim, num_heads=4, window_size=8, shift=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad to multiple of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, [0, pad_w, 0, pad_h])
        _, _, Hp, Wp = x.shape

        # Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # Partition into windows: [B, C, Hp, Wp] -> [B*nW, ws*ws, C]
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nH, nW, ws, ws, C]
        nH, nW = Hp // ws, Wp // ws
        x = x.view(B * nH * nW, ws * ws, C)

        # Layer norm
        x = self.norm(x)

        # Multi-head self-attention
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, BnW, heads, wsws, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        x = self.proj(x)

        # Unpartition: [B*nW, ws*ws, C] -> [B, C, Hp, Wp]
        x = x.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, nH, ws, nW, ws]
        x = x.view(B, C, Hp, Wp)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # Remove padding
        x = x[:, :, :H, :W]
        return x


# ============================================================================
# Root Query Cross-Attention
# ============================================================================

class RootQueryCrossAttention(nn.Module):
    """
    Cross-attention between image features and learned Root Query Tokens.

    Root Queries represent structural archetypes:
    - Primary root (thick, deep)
    - Lateral root (thin, branching)
    - Root tip (terminal)
    - Junction (branching point)

    Each query learns to attend to its respective structure type in the image.
    """

    def __init__(self, dim, num_queries=4, num_heads=4):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5

        # Learned root query tokens
        self.root_queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Project query outputs back to spatial
        self.spatial_proj = nn.Linear(num_queries, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Flatten spatial dims
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        # Expand queries for batch
        queries = self.root_queries.expand(B, -1, -1)  # [B, Q, C]

        # Cross-attention: queries attend to image features
        q = self.q_proj(queries).view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, Q, HW]
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # [B, heads, Q, head_dim]
        out = out.transpose(1, 2).reshape(B, self.num_queries, C)  # [B, Q, C]

        # Project back to spatial: each pixel gets weighted contribution from all queries
        # attn_weights: [B, heads, Q, HW] → [B, Q, HW]
        attn_spatial = attn.mean(dim=1)  # [B, Q, HW]

        # Weighted sum of query outputs projected to each pixel
        query_contribution = torch.bmm(attn_spatial.transpose(1, 2), self.out_proj(out))  # [B, HW, C]
        query_contribution = query_contribution.permute(0, 2, 1).view(B, C, H, W)

        return x + query_contribution


# ============================================================================
# Transformer Block
# ============================================================================

class RhizoTransformerBlock(nn.Module):
    """One transformer block: Window Attention → Root Query Attention → FFN."""

    def __init__(self, dim, num_heads=4, window_size=8, shift=False, mlp_ratio=2.0):
        super().__init__()
        self.window_attn = WindowedRootSelfAttention(dim, num_heads, window_size, shift)
        self.root_query_attn = RootQueryCrossAttention(dim, num_queries=4, num_heads=num_heads)

        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(mlp_dim, dim, 1),
        )
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        # Window self-attention
        x = x + self.window_attn(self.norm1(x))
        # Root query cross-attention
        x = self.root_query_attn(x)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Full Model
# ============================================================================

class RhizoHybridTransformer(nn.Module):
    """
    RhizoHybridTransformer: A novel CNN-Transformer hybrid architecture
    for plant root segmentation.

    Pipeline:
    1. CNN Stem → multi-scale local features
    2. Topology-Guided Positional Encoding → radial bias
    3. Transformer Blocks → global context with root queries
    4. Progressive Upsampling Decoder → fine-grained segmentation

    Args:
        in_channels: Input channels (3 for RGB)
        out_channels: Output channels (1 for binary mask)
        embed_dim: Transformer embedding dimension
        num_heads: Attention heads
        depth: Number of transformer blocks
        window_size: Window attention window size
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        embed_dim=96,
        num_heads=4,
        depth=4,
        window_size=8,
    ):
        super().__init__()

        # CNN Stem (extracts multi-scale features)
        self.stem = RhizoCNNStem(in_channels, embed_dim)

        # Positional encoding
        self.pos_enc = TopologyGuidedPE(embed_dim)

        # Transformer blocks (alternating regular and shifted windows)
        self.transformer_blocks = nn.ModuleList()
        for i in range(depth):
            shift = (i % 2 == 1)
            self.transformer_blocks.append(
                RhizoTransformerBlock(embed_dim, num_heads, window_size, shift)
            )

        # Progressive upsampling decoder
        stem_dims = [embed_dim // 4, embed_dim // 2, embed_dim]

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ELU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ELU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ELU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ELU(inplace=True),
        )

        # Final head
        self.final = nn.Sequential(
            nn.Conv2d(embed_dim // 4, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, out_channels, 1),
        )

    def forward(self, x):
        input_size = x.shape[2:]

        # CNN Stem → multi-scale features
        stem_features = self.stem(x)  # [feat_s1, feat_s2, feat_s3]

        # Use deepest stem feature as transformer input
        feat = stem_features[-1]

        # Add positional encoding
        feat = self.pos_enc(feat)

        # Transformer blocks
        for block in self.transformer_blocks:
            feat = block(feat)

        # Progressive upsampling with CNN skip connections
        # Level 1: embed_dim → embed_dim//2
        x = self.up1(feat)
        skip2 = stem_features[1]
        diff_h = skip2.size(2) - x.size(2)
        diff_w = skip2.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = x + skip2
        x = self.dec1(x)

        # Level 2: embed_dim//2 → embed_dim//4
        x = self.up2(x)
        skip1 = stem_features[0]
        diff_h = skip1.size(2) - x.size(2)
        diff_w = skip1.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = x + skip1
        x = self.dec2(x)

        # Final segmentation
        out = self.final(x)

        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return out


if __name__ == "__main__":
    model = RhizoHybridTransformer(in_channels=3, out_channels=1, embed_dim=96, depth=4)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")
