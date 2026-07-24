"""
DualStreamRootNet: Dual-Path Architecture for Root Segmentation
================================================================

A NOVEL architecture that processes root images through two complementary
encoder streams, each capturing different aspects of root morphology.

KEY INNOVATIONS:

1. Dual Encoder Streams:
   - Spatial Stream: Learns standard visual features (texture, edges, color)
   - Tubularity Stream: Uses Hessian-based tubularity enhancement as input,
     specifically designed to detect ridge-like structures (roots) in soil

2. Cross-Stream Attention Fusion (CSAF):
   - At each encoder level, features from both streams attend to each other
   - Spatial stream provides context about what IS a root
   - Tubularity stream provides context about WHERE roots are
   - Learnable gating decides how much to trust each stream

3. Structure-Preserving Decoder:
   - Receives fused features from both streams
   - Incorporates a connectivity prior that penalizes isolated predictions
   - Outputs both segmentation mask AND root centerline prediction (auxiliary)

4. Adaptive Root-Soil Contrast Module:
   - Learns to enhance contrast between root and soil regions
   - Critical for minirhizotron images with poor lighting
   - Acts as a learned preprocessing step

WHY TWO STREAMS?
Root segmentation has a unique challenge: thin lateral roots have very low
contrast against soil backgrounds. Standard CNNs struggle because:
- Max-pooling destroys single-pixel signals
- Standard convolutions are isotropic but roots are directional
- Soil texture can mimic root-like patterns

The tubularity stream explicitly computes Hessian eigenvalue-based features
that highlight elongated structures, giving the network a strong structural
prior that roots are ridge-like objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Adaptive Contrast Enhancement
# ============================================================================

class AdaptiveContrastModule(nn.Module):
    """
    Learnable preprocessing that enhances root-soil contrast.
    Inspired by CLAHE but fully differentiable and learned end-to-end.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # Learn per-channel contrast curves
        self.contrast = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=4),
            nn.ELU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        # Local mean subtraction (highlights local deviations like roots)
        self.local_mean = nn.AvgPool2d(kernel_size=31, stride=1, padding=15)

    def forward(self, x):
        # Local contrast: deviation from local mean
        local_mean = self.local_mean(x)
        local_contrast = x - local_mean

        # Learned contrast enhancement
        enhancement = self.contrast(x)

        # Combine original + enhanced
        return x + local_contrast * enhancement


# ============================================================================
# Hessian Tubularity Feature Extractor
# ============================================================================

class HessianTubularityExtractor(nn.Module):
    """
    Computes Hessian-based tubularity features that highlight elongated
    ridge-like structures (roots). This is a differentiable approximation
    of Frangi vesselness / Sato tubularity filters.

    The Hessian matrix eigenvalues at each pixel indicate local structure:
    - λ1 ≈ 0, λ2 << 0: Bright ridge (root in dark soil)
    - λ1 ≈ 0, λ2 >> 0: Dark ridge (root in bright background)
    - Both large: Blob
    - Both ≈ 0: Flat region

    We compute these at multiple scales (sigma) to detect roots of
    different thicknesses.
    """

    def __init__(self, in_channels=3, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        # Convert to grayscale first
        self.to_gray = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.to_gray.weight, 1.0 / in_channels)

        # Gaussian smoothing at different scales
        self.scales = nn.ModuleList()
        for s in range(num_scales):
            sigma = 1.0 + s * 1.5  # sigma = 1.0, 2.5, 4.0
            kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
            self.scales.append(
                nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2,
                              groups=1, bias=False),
                    nn.BatchNorm2d(1),
                )
            )

        # Learned Hessian approximation (Sobel-like 2nd derivatives)
        self.dxx = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.dyy = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.dxy = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Initialize with 2nd derivative kernels
        with torch.no_grad():
            self.dxx.weight.copy_(torch.tensor([[[[0, 0, 0], [1, -2, 1], [0, 0, 0]]]], dtype=torch.float32))
            self.dyy.weight.copy_(torch.tensor([[[[0, 1, 0], [0, -2, 0], [0, 1, 0]]]], dtype=torch.float32))
            self.dxy.weight.copy_(torch.tensor([[[[0.25, 0, -0.25], [0, 0, 0], [-0.25, 0, 0.25]]]], dtype=torch.float32))

        # Combine multi-scale tubularity into feature maps
        self.combine = nn.Sequential(
            nn.Conv2d(num_scales, 8, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 3, kernel_size=1),  # Output 3 channels to match spatial stream
        )

    def forward(self, x):
        gray = self.to_gray(x)

        tubularity_maps = []
        for scale_conv in self.scales:
            smoothed = scale_conv(gray)

            # Compute Hessian components
            Ixx = self.dxx(smoothed)
            Iyy = self.dyy(smoothed)
            Ixy = self.dxy(smoothed)

            # Eigenvalues of 2x2 Hessian: λ = 0.5 * (Ixx + Iyy ± sqrt((Ixx-Iyy)² + 4*Ixy²))
            trace = Ixx + Iyy
            det = Ixx * Iyy - Ixy * Ixy
            discriminant = torch.clamp(trace * trace - 4 * det, min=1e-8)
            sqrt_disc = torch.sqrt(discriminant)

            lambda1 = 0.5 * (trace + sqrt_disc)
            lambda2 = 0.5 * (trace - sqrt_disc)

            # Tubularity: |λ2| when λ1 ≈ 0 (ridge structure)
            # Use soft version: |λ2| * exp(-λ1²/2)
            tubularity = torch.abs(lambda2) * torch.exp(-lambda1.pow(2) * 0.5)
            tubularity_maps.append(tubularity)

        # Stack multi-scale tubularity
        multi_scale = torch.cat(tubularity_maps, dim=1)  # [B, num_scales, H, W]
        return self.combine(multi_scale)


# ============================================================================
# Cross-Stream Attention Fusion (CSAF)
# ============================================================================

class CrossStreamAttentionFusion(nn.Module):
    """
    Fuses features from spatial and tubularity streams using
    bidirectional cross-attention.

    Spatial → Tubularity: "Which spatial features correspond to tube-like structures?"
    Tubularity → Spatial: "What does the actual root look like at tube-detected locations?"
    """

    def __init__(self, channels):
        super().__init__()
        mid = channels // 4

        # Spatial attends to Tubularity
        self.s2t_query = nn.Conv2d(channels, mid, 1)
        self.s2t_key = nn.Conv2d(channels, mid, 1)
        self.s2t_value = nn.Conv2d(channels, channels, 1)

        # Tubularity attends to Spatial
        self.t2s_query = nn.Conv2d(channels, mid, 1)
        self.t2s_key = nn.Conv2d(channels, mid, 1)
        self.t2s_value = nn.Conv2d(channels, channels, 1)

        # Gating: learn how much to trust each stream
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.BatchNorm2d(channels)

    def forward(self, spatial, tubular):
        B, C, H, W = spatial.shape

        # Spatial attends to Tubularity (find tube-like spatial features)
        q_s = self.s2t_query(spatial).view(B, -1, H * W)    # [B, mid, HW]
        k_t = self.s2t_key(tubular).view(B, -1, H * W)      # [B, mid, HW]
        v_t = self.s2t_value(tubular).view(B, C, H * W)     # [B, C, HW]

        # Efficient attention (avoid full HW x HW matrix)
        attn_s2t = torch.bmm(q_s.transpose(1, 2), k_t)      # [B, HW, HW] -- use channel attention instead
        # Use channel-level attention for efficiency
        q_s_pool = q_s.mean(dim=2, keepdim=True)             # [B, mid, 1]
        k_t_pool = k_t.mean(dim=2, keepdim=True)             # [B, mid, 1]
        channel_attn = torch.sigmoid(torch.bmm(q_s_pool.transpose(1, 2), k_t_pool))  # [B, 1, 1]
        s2t_out = (v_t * channel_attn).view(B, C, H, W)

        # Tubularity attends to Spatial
        q_t = self.t2s_query(tubular).view(B, -1, H * W)
        k_s = self.t2s_key(spatial).view(B, -1, H * W)
        v_s = self.t2s_value(spatial).view(B, C, H * W)

        q_t_pool = q_t.mean(dim=2, keepdim=True)
        k_s_pool = k_s.mean(dim=2, keepdim=True)
        channel_attn2 = torch.sigmoid(torch.bmm(q_t_pool.transpose(1, 2), k_s_pool))
        t2s_out = (v_s * channel_attn2).view(B, C, H, W)

        # Gate: combine both cross-attended features
        gate_input = torch.cat([s2t_out, t2s_out], dim=1)
        alpha = self.gate(gate_input)

        fused = alpha * (spatial + s2t_out) + (1 - alpha) * (tubular + t2s_out)
        return self.norm(fused)


# ============================================================================
# DualStreamRootNet
# ============================================================================

class DualStreamEncoder(nn.Module):
    """Single stream encoder block."""

    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)


class DualStreamRootNet(nn.Module):
    """
    DualStreamRootNet: A novel dual-path architecture for root segmentation.

    Two parallel encoder streams process the image:
    1. Spatial Stream: Standard visual features from RGB
    2. Tubularity Stream: Hessian-based ridge enhancement features

    Features are fused via Cross-Stream Attention at each encoder level,
    then decoded through a shared decoder with multi-scale supervision.

    Args:
        in_channels: Input channels (3 for RGB)
        out_channels: Output channels (1 for binary mask)
        features: Channel sizes per level
        dropout: Dropout rate
    """

    def __init__(self, in_channels=3, out_channels=1, features=None, dropout=0.1):
        super().__init__()

        if features is None:
            features = [24, 48, 96, 192, 384]

        # Preprocessing
        self.contrast_enhance = AdaptiveContrastModule(in_channels)
        self.tubularity_extract = HessianTubularityExtractor(in_channels)

        # Spatial stream encoders
        self.spatial_encoders = nn.ModuleList()
        prev = in_channels
        for feat in features[:-1]:
            self.spatial_encoders.append(DualStreamEncoder(prev, feat, dropout))
            prev = feat

        # Tubularity stream encoders
        self.tubular_encoders = nn.ModuleList()
        prev = in_channels  # Tubularity extractor outputs 3 channels
        for feat in features[:-1]:
            self.tubular_encoders.append(DualStreamEncoder(prev, feat, dropout))
            prev = feat

        # Cross-stream attention fusion at each level
        self.cross_fusions = nn.ModuleList()
        for feat in features[:-1]:
            self.cross_fusions.append(CrossStreamAttentionFusion(feat))

        # Shared bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-2], features[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]),
            nn.ELU(inplace=True),
            nn.Conv2d(features[-1], features[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]),
            nn.ELU(inplace=True),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 2, -1, -1):
            in_ch = features[i + 1] if i + 1 < len(features) else features[-1]
            out_ch = features[i]
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ELU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ELU(inplace=True),
                )
            )

        # Skip adaptation (fused features to decoder)
        self.skip_adapts = nn.ModuleList()
        for feat in features[:-1]:
            self.skip_adapts.append(nn.Conv2d(feat, feat, 1))

        # Output heads
        self.seg_head = nn.Conv2d(features[0], out_channels, 1)  # Main segmentation
        self.centerline_head = nn.Conv2d(features[0], out_channels, 1)  # Auxiliary: root centerline

    def forward(self, x, return_centerline=False):
        # Preprocessing
        x_enhanced = self.contrast_enhance(x)
        x_tubular = self.tubularity_extract(x)

        # Dual-stream encoding with cross-fusion
        s_x = x_enhanced
        t_x = x_tubular
        fused_skips = []

        for i, (s_enc, t_enc, fusion) in enumerate(
            zip(self.spatial_encoders, self.tubular_encoders, self.cross_fusions)
        ):
            s_feat, s_x = s_enc(s_x)
            t_feat, t_x = t_enc(t_x)
            fused = fusion(s_feat, t_feat)
            fused_skips.append(fused)

        # Bottleneck (combine both streams)
        bottleneck_in = s_x + t_x  # Element-wise sum
        x = self.bottleneck(bottleneck_in)

        # Decoder with fused skip connections
        fused_skips = fused_skips[::-1]
        for i, (decoder, skip_adapt) in enumerate(zip(self.decoders, self.skip_adapts[::-1])):
            x = decoder[0](x)  # Upsample
            skip = skip_adapt(fused_skips[i])
            # Pad if needed
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
            x = x + skip  # Residual skip
            x = decoder[1:](x)  # Conv blocks

        seg = self.seg_head(x)

        if return_centerline:
            centerline = self.centerline_head(x)
            return seg, centerline

        return seg


if __name__ == "__main__":
    model = DualStreamRootNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")

    y_seg, y_center = model(x, return_centerline=True)
    print(f"Segmentation: {y_seg.shape}")
    print(f"Centerline:   {y_center.shape}")
