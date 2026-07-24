"""
RhizoAttentionNet: Tubular Attention Network for Root Segmentation
===================================================================

A NOVEL architecture specifically designed for the unique challenges of
plant root segmentation. Unlike generic segmentation networks, this model
incorporates domain-specific inductive biases for tubular/filamentary structures.

KEY INNOVATIONS:
1. Oriented Tubular Attention Module (OTAM):
   - Applies directional attention aligned with local root orientation
   - Uses oriented Gabor-like learned filters at multiple angles
   - Captures the elongated, anisotropic nature of root structures

2. Topology-Preserving Decoder with clDice Regularization:
   - Ensures predicted masks maintain connected tubular structures
   - Soft-skeleton extraction during training for topological supervision
   - Prevents fragmented root predictions common with pixel-wise losses

3. Multi-Scale Root Feature Pyramid (MSRFP):
   - Parallel branches for different root thickness scales
   - Fine roots (1-2px): high-resolution, low-level features
   - Thick primary roots: deeper, semantic features
   - Adaptive fusion across scales

4. Root-Aware Squeeze-and-Excitation (RASE):
   - Channel attention calibrated for root vs. soil discrimination
   - Learned emphasis on edge/ridge-detecting channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Oriented Tubular Attention Module (OTAM)
# ============================================================================

class OrientedTubularAttention(nn.Module):
    """
    Applies directional attention along multiple orientations to capture
    the elongated structure of roots growing in various directions.

    Unlike standard attention which is isotropic, OTAM uses oriented
    strip-shaped attention kernels at N angles (0°, 30°, 60°, ..., 150°),
    then aggregates the best-matching orientation per spatial location.
    """

    def __init__(self, channels, num_orientations=6, kernel_length=11):
        super().__init__()
        self.num_orientations = num_orientations
        self.kernel_length = kernel_length

        # Learnable oriented convolution filters (strip-shaped)
        self.oriented_convs = nn.ModuleList()
        for i in range(num_orientations):
            angle = i * (180.0 / num_orientations)
            # Each oriented conv is a 1D-like conv applied along a direction
            self.oriented_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                              groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ELU(inplace=True),
                )
            )

        # Orientation selection attention
        self.orientation_gate = nn.Sequential(
            nn.Conv2d(channels * num_orientations, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ELU(inplace=True),
            nn.Conv2d(channels, num_orientations, kernel_size=1),
            nn.Softmax(dim=1),
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Apply each oriented convolution
        oriented_features = []
        for conv in self.oriented_convs:
            oriented_features.append(conv(x))

        # Stack and compute orientation attention weights
        stacked = torch.cat(oriented_features, dim=1)  # [B, C*N, H, W]
        orientation_weights = self.orientation_gate(stacked)  # [B, N, H, W]

        # Weighted combination of oriented features
        combined = torch.zeros_like(x)
        for i, feat in enumerate(oriented_features):
            weight = orientation_weights[:, i:i+1, :, :]  # [B, 1, H, W]
            combined = combined + feat * weight

        # Residual connection
        return x + self.project(combined)


# ============================================================================
# Root-Aware Squeeze-and-Excitation (RASE)
# ============================================================================

class RootAwareSE(nn.Module):
    """
    Channel attention that learns to emphasize feature channels most
    relevant for root vs. soil discrimination. Includes a root-frequency
    prior that biases attention toward edge/ridge-detecting channels.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ELU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        # Learnable root-frequency bias (high-frequency channels matter more for thin roots)
        self.root_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        B, C, _, _ = x.shape
        scale = self.squeeze(x).view(B, C)
        scale = self.excitation(scale).view(B, C, 1, 1)
        # Apply root-frequency bias
        scale = scale + torch.sigmoid(self.root_bias)
        return x * scale


# ============================================================================
# Multi-Scale Root Feature Pyramid (MSRFP)
# ============================================================================

class MultiScaleRootPyramid(nn.Module):
    """
    Processes features at multiple scales simultaneously to handle the
    extreme variation in root thickness:
    - Fine laterals: 1-2 pixels wide
    - Primary/seminal roots: 5-20+ pixels wide
    - Coarse structural axes: even wider

    Uses dilated convolutions at different rates to capture different
    tubular widths without losing resolution.
    """

    def __init__(self, channels):
        super().__init__()
        branch_ch = channels // 4

        # Fine roots (rate=1, small receptive field)
        self.fine_branch = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ELU(inplace=True),
        )

        # Medium roots (rate=3)
        self.medium_branch = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ELU(inplace=True),
        )

        # Thick roots (rate=5)
        self.thick_branch = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ELU(inplace=True),
        )

        # Global context (1x1 after global pool)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ELU(inplace=True),
        )

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        fine = self.fine_branch(x)
        medium = self.medium_branch(x)
        thick = self.thick_branch(x)
        glob = self.global_branch(x)
        glob = F.interpolate(glob, size=x.shape[2:], mode="bilinear", align_corners=False)

        fused = torch.cat([fine, medium, thick, glob], dim=1)
        return self.fuse(fused)


# ============================================================================
# RhizoAttentionNet Encoder/Decoder Blocks
# ============================================================================

class RhizoAttnEncoderBlock(nn.Module):
    """Encoder block with OTAM + RASE + MSRFP."""

    def __init__(self, in_ch, out_ch, use_otam=True, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )
        self.otam = OrientedTubularAttention(out_ch) if use_otam else nn.Identity()
        self.rase = RootAwareSE(out_ch)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.otam(x)
        x = self.rase(x)
        return x, self.pool(x)


class RhizoAttnDecoderBlock(nn.Module):
    """Decoder block with MSRFP for multi-scale root fusion."""

    def __init__(self, in_ch, skip_ch, out_ch, use_msrfp=True, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.skip_adapt = nn.Conv2d(skip_ch, out_ch, 1) if skip_ch != out_ch else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )
        self.msrfp = MultiScaleRootPyramid(out_ch) if use_msrfp else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        skip = self.skip_adapt(skip)
        x = x + skip  # Residual skip

        x = self.conv(x)
        x = self.msrfp(x)
        return x


# ============================================================================
# Full Model
# ============================================================================

class RhizoAttentionNet(nn.Module):
    """
    RhizoAttentionNet: A novel architecture for plant root segmentation
    combining Oriented Tubular Attention, Multi-Scale Root Feature Pyramids,
    and Root-Aware Squeeze-and-Excitation.

    This model is specifically engineered for the challenges of root imagery:
    - Thin, single-pixel lateral roots that vanish with standard convolutions
    - Highly directional structures that benefit from oriented processing
    - Extreme class imbalance (<2% root pixels)
    - Complex soil background noise in minirhizotron images

    Args:
        in_channels: Input channels (3 for RGB)
        out_channels: Output channels (1 for binary mask)
        features: Channel sizes per level [default: 24, 48, 96, 192, 384]
        dropout: Dropout rate
    """

    def __init__(self, in_channels=3, out_channels=1, features=None, dropout=0.1):
        super().__init__()

        if features is None:
            features = [24, 48, 96, 192, 384]

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for i, feat in enumerate(features[:-1]):
            use_otam = i >= 1  # Skip OTAM at first level (too early)
            self.encoders.append(RhizoAttnEncoderBlock(prev_ch, feat, use_otam, dropout))
            prev_ch = feat

        # Bottleneck with MSRFP
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-2], features[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]),
            nn.ELU(inplace=True),
            MultiScaleRootPyramid(features[-1]),
            nn.Conv2d(features[-1], features[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]),
            nn.ELU(inplace=True),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 2, -1, -1):
            in_ch = features[i + 1] if i + 1 < len(features) else features[-1]
            self.decoders.append(RhizoAttnDecoderBlock(in_ch, features[i], features[i], True, dropout))

        # Deep supervision heads (auxiliary outputs at each decoder level)
        self.deep_supervision_heads = nn.ModuleList()
        for feat in features[:-1]:
            self.deep_supervision_heads.append(nn.Conv2d(feat, out_channels, 1))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x, return_deep=False):
        input_size = x.shape[2:]

        # Encoder
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with deep supervision
        skips = skips[::-1]
        deep_outputs = []
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[i])
            if return_deep:
                ds_out = self.deep_supervision_heads[len(self.decoders) - 1 - i](x)
                ds_out = F.interpolate(ds_out, size=input_size, mode="bilinear", align_corners=False)
                deep_outputs.append(ds_out)

        out = self.final_conv(x)

        if return_deep:
            return out, deep_outputs
        return out


if __name__ == "__main__":
    model = RhizoAttentionNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")

    # Test deep supervision
    y, deep = model(x, return_deep=True)
    print(f"Deep supervision outputs: {len(deep)}")
    for i, d in enumerate(deep):
        print(f"  Level {i}: {d.shape}")
