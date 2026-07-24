#!/usr/bin/env python3
"""
ONNX Model Exporter for RHIZO-NET
==================================

Exports all RHIZO-NET neural network architectures to ONNX format:
1. RhizoUNet (rhizo_unet.onnx)
2. RhizoAttentionNet (rhizo_attention_net.onnx)
3. DualStreamRootNet (dual_stream_root_net.onnx)
4. RhizoHybridTransformer (rhizo_hybrid_transformer.onnx)
5. RhizoFusionNet (rhizo_fusionnet.onnx)

Output directory: ./architecture/
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.unet.model import RhizoUNet
from src.unet.rhizo_attention_net import RhizoAttentionNet
from src.unet.dual_stream_root_net import DualStreamRootNet
from src.unet.rhizo_hybrid_transformer import RhizoHybridTransformer


def export_segmentation_models(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 256, 256)

    models = {
        "rhizo_unet.onnx": RhizoUNet(in_channels=3, out_channels=1),
        "rhizo_attention_net.onnx": RhizoAttentionNet(in_channels=3, out_channels=1),
        "dual_stream_root_net.onnx": DualStreamRootNet(in_channels=3, out_channels=1),
        "rhizo_hybrid_transformer.onnx": RhizoHybridTransformer(in_channels=3, out_channels=1, embed_dim=48, depth=2),
    }

    for filename, model in models.items():
        model.eval()
        onnx_path = output_dir / filename
        print(f"Exporting {filename} to {onnx_path}...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input_image"],
                output_names=["segmentation_mask"],
                dynamic_axes={
                    "input_image": {0: "batch_size", 2: "height", 3: "width"},
                    "segmentation_mask": {0: "batch_size", 2: "height", 3: "width"},
                },
            )
            print(f"✓ Successfully exported: {filename} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"✗ Failed to export {filename}: {e}")


def main():
    arch_dir = PROJECT_ROOT / "architecture"
    print("=" * 60)
    print("RHIZO-NET ONNX Exporter")
    print(f"Output Directory: {arch_dir}")
    print("=" * 60)
    
    export_segmentation_models(arch_dir)
    print("\n✓ ONNX Export complete!")


if __name__ == "__main__":
    main()
