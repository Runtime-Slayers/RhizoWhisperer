"""
RhizoUNet: Modified U-Net for Root Segmentation
=================================================

A lightweight U-Net variant optimized for the tubular topology of plant roots.
Modifications from standard U-Net (as per project documents):

1. Feature Map Reduction: 4x fewer channels (16,32,64,128,256 vs 64,128,256,512,1024)
2. ELU Activation: Prevents dying ReLU on thin, narrow root structures
3. Average Pooling: Preserves fine single-pixel roots (max-pool exaggerates noise)
4. Residual Skip Connections: Element-wise sum instead of concatenation (ResUNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with ELU activation."""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock + AvgPool downsampling."""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        # Average pooling instead of max pooling (preserves fine roots)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block with residual skip connections.
    Uses element-wise summation instead of concatenation (ResUNet style).
    """

    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 1x1 conv to match channels for residual addition if needed
        self.skip_adapt = (
            nn.Conv2d(skip_channels, out_channels, kernel_size=1)
            if skip_channels != out_channels
            else nn.Identity()
        )
        self.conv = ConvBlock(out_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad if sizes don't match exactly
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Residual skip: element-wise sum (not concatenation)
        skip_adapted = self.skip_adapt(skip)
        x = x + skip_adapted

        return self.conv(x)


class RhizoUNet(nn.Module):
    """
    Modified U-Net specifically designed for root segmentation.

    Key innovations:
    - 4x reduced feature maps for lightweight inference
    - ELU activations prevent gradient death on thin structures
    - Average pooling preserves fine single-pixel root signals
    - Residual skip connections (additive) for better gradient flow

    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (1 for binary mask)
        features: List of feature map sizes per encoder level
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=None,
        dropout=0.1,
    ):
        super().__init__()

        if features is None:
            # 4x reduced from standard U-Net [64,128,256,512,1024]
            features = [16, 32, 64, 128, 256]

        self.features = features

        # Encoder path
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for feat in features[:-1]:
            self.encoders.append(EncoderBlock(prev_channels, feat, dropout))
            prev_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-2], features[-1], dropout)

        # Decoder path
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 2, -1, -1):
            in_ch = features[i + 1] if i + 1 < len(features) else features[-1]
            skip_ch = features[i]
            out_ch = features[i]
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch, dropout))

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (reverse skip connections)
        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        # Final output
        return self.final_conv(x)

    def get_confidence(self, x):
        """
        Get segmentation mask with per-pixel confidence scores.
        Used to decide whether to route to MobileSAM fallback.
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        # Mean confidence across the predicted root region
        confidence = probs.mean(dim=[1, 2, 3])
        return probs, confidence


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = RhizoUNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Features: {model.features}")
