# RhizoWhisperer Model Architectures & ONNX Benchmarks

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime%20v1.16%2B-blue.svg)](https://onnxruntime.ai/)
[![Organization: Runtime Slayers](https://img.shields.io/badge/Organization-Runtime%20Slayers-purple.svg)](https://github.com/Runtime-Slayers)
[![Main Repo](https://img.shields.io/badge/Main-RhizoWhisperer%20Repo-green.svg)](https://github.com/Runtime-Slayers/RhizoWhisperer)

> Dedicated repository containing PyTorch implementations, exported **.onnx** binary weights, layer-by-layer parameter specifications, receptive field calculations, and individual architectural flowcharts for **RHIZO-NET**.

---

## 📑 Table of Contents
- [Architectural Benchmarks Summary](#-architectural-benchmarks-summary)
- [Model 1: RhizoUNet (Modified U-Net)](#1-rhizounet-modified-u-net)
- [Model 2: RhizoAttentionNet (OTAM + MSRFP)](#2-rhizoattentionnet-otam--msrfp)
- [Model 3: DualStreamRootNet (Hessian Dual Stream)](#3-dualstreamrootnet-hessian-dual-stream)
- [Model 4: RhizoHybridTransformer (Swin + RQT)](#4-rhizohybridtransformer-swin--rqt)
- [Model 5: RhizoGraphFormer (Graph Transformer + LPE)](#5-rhizographformer-graph-transformer--lpe)
- [ONNX Runtime Benchmarks & Inference Guide](#-onnx-runtime-benchmarks--inference-guide)
- [License](#-license)

---

## 📊 Architectural Benchmarks Summary

| Model Architecture | Parameters | ONNX File Size | Latency (CPU) | Loss (Curriculum) | IoU Accuracy | Key Innovation |
|---|---|---|---|---|---|---|
| **RhizoUNet** | 1,746,737 | `rhizo_unet.onnx` (6.69 MB) | 4.2 ms | 0.0580 | 94.2% | ELU activations, AvgPool, residual skip connections |
| **RhizoAttentionNet** | 5,892,305 | `rhizo_attention_net.onnx` (22.55 MB) | 12.8 ms | **0.0412** | **97.9%** | Oriented Attention (OTAM) + Receptive Field Pyramid (MSRFP) |
| **DualStreamRootNet** | 4,885,959 | `dual_stream_root_net.onnx` (18.73 MB) | 9.6 ms | 0.0482 | 96.5% | Parallel spatial RGB stream + Frangi Hessian vesselness stream |
| **RhizoHybridTransformer**| **79,749** | `rhizo_hybrid_transformer.onnx` (1.17 MB) | **1.8 ms** | 0.0451 | 95.8% | Shifted-Window Swin Attention + Root Query Tokens (RQT) |
| **RhizoGraphFormer** | 64,128 | N/A (Graph Vector) | 0.8 ms | N/A | N/A | Laplacian Positional Encoding (LPE) graph transformer |

---

## 1. RhizoUNet (Modified U-Net)

`RhizoUNet` enhances standard U-Net with Exponential Linear Unit (ELU) activations, Average Pooling (to preserve continuous thin root boundaries), and Residual Skip Connections.

```mermaid
flowchart TD
    subgraph Encoder ["RhizoUNet Encoder"]
        In["Input Image: 128x128x3"] --> C1["Conv3x3 + ELU: 64 channels"]
        C1 --> P1["AvgPool2d 2x2: 64x64"]
        P1 --> C2["ConvBlock + ResSkip: 128 channels"]
        C2 --> P2["AvgPool2d 2x2: 32x32"]
        P2 --> C3["ConvBlock + ResSkip: 256 channels"]
        C3 --> P3["AvgPool2d 2x2: 16x16"]
    end

    subgraph Bottleneck ["Bridge Bottleneck"]
        P3 --> BN["ConvBlock + ELU + Dropout 0.2: 512 channels"]
    end

    subgraph Decoder ["RhizoUNet Decoder"]
        BN --> UP1["Bilinear Upsample 2x2: 256 channels"]
        UP1 & C3 --> CAT1["Concat Skip Connection: 512 channels"]
        CAT1 --> DC1["ConvBlock + ELU: 256 channels"]
        DC1 --> UP2["Bilinear Upsample 2x2: 128 channels"]
        UP2 & C2 --> CAT2["Concat Skip Connection: 256 channels"]
        CAT2 --> DC2["ConvBlock + ELU: 128 channels"]
        DC2 --> UP3["Bilinear Upsample 2x2: 64 channels"]
        UP3 & C1 --> CAT3["Concat Skip Connection: 128 channels"]
        CAT3 --> DC3["ConvBlock + ELU: 64 channels"]
        DC3 --> HEAD["1x1 Conv Head: 1 channel Logits"]
    end
```

---

## 2. RhizoAttentionNet (OTAM + MSRFP)

`RhizoAttentionNet` incorporates an Oriented Topological Attention Module (OTAM) that computes directional spatial attention maps across 4 cardinal angles (0°, 45°, 90°, 135°) to suppress soil background noise while preserving fine root tips.

```mermaid
flowchart TD
    In["Input RGB Image: 128x128"] --> MSRFP["Multi-Scale Receptive Field Pyramid: Convs 3x3, 5x5, 7x7 Dilated"]
    MSRFP --> Enc["Deep Residual Feature Extractor"]
    Enc --> OTAM["Oriented Topological Attention Module"]
    
    subgraph OTAM_Detail ["OTAM Directional Gating"]
        OTAM --> A0["0 Degree Horizontal Filter"]
        OTAM --> A45["45 Degree Diagonal Filter"]
        OTAM --> A90["90 Degree Vertical Filter"]
        OTAM --> A135["135 Degree Anti-Diagonal Filter"]
        A0 & A45 & A90 & A135 --> SoftmaxGate["Spatial Softmax Gating"]
    end

    SoftmaxGate --> GatedFeat["Gated Feature Fusion"]
    GatedFeat --> Dec["Deep Supervision Decoder"]
    Dec --> Head1["Auxiliary Out Head 1"]
    Dec --> Head2["Auxiliary Out Head 2"]
    Dec --> Final["Final 1x1 Conv Mask: 97.9% IoU"]
```

---

## 3. DualStreamRootNet (Hessian Dual Stream)

`DualStreamRootNet` combines a standard spatial RGB convolutional encoder stream with a dedicated Frangi Hessian vesselness filter matrix stream to detect tubular root structures in dense soil background.

```mermaid
flowchart TD
    In["Input Image: 128x128"] --> Stream1["Spatial RGB Stream: 4-Level Conv Encoder"]
    In --> HessianFilter["Multi-Scale Frangi Hessian Filter: Eigenvalues L1 and L2"]
    HessianFilter --> VesselnessMap["Tubular Vesselness Response Map"]
    VesselnessMap --> Stream2["Hessian Tube Stream: 3-Level Conv Encoder"]
    
    Stream1 & Stream2 --> FusionGate["Cross-Stream Adaptive Fusion Gate: Alpha * Spatial + 1-Alpha * Hessian"]
    FusionGate --> JointDecoder["Joint Feature Decoder"]
    JointDecoder --> HeadMask["Primary Root Mask Head"]
    JointDecoder --> HeadCenterline["Centerline Skeleton Head"]
```

---

## 4. RhizoHybridTransformer (Swin + RQT)

`RhizoHybridTransformer` is an ultra-compressed architecture (79.7K parameters, 1.17 MB ONNX size) designed for mobile and drone edge deployment. It uses Shifted-Window Swin Self-Attention (W-MSA/SW-MSA) combined with Root Query Tokens (RQT).

```mermaid
flowchart TD
    In["Input Image: 128x128"] --> PatchEmbed["Patch Embedding: Patch Size 4x4 to 32x32 Tokens"]
    PatchEmbed --> SwinStage1["Swin Block 1: Window Self-Attention W-MSA"]
    SwinStage1 --> SwinStage2["Swin Block 2: Shifted Window Attention SW-MSA"]
    
    subgraph RQT ["Root Query Tokens RQT"]
        SwinStage2 --> RQT_Module["Root Query Cross-Attention: Learned Tokens for Root Junctions"]
    end

    RQT_Module --> LightDecoder["Lightweight Up-Projection Decoder"]
    LightDecoder --> Head["1x1 Out Conv Head: 1.8 ms Latency"]
```

---

## 5. RhizoGraphFormer (Graph Transformer + LPE)

`RhizoGraphFormer` processes extracted root skeleton graphs. It uses Laplacian Positional Encodings (LPE) derived from normalized graph Laplacian eigenvectors ($L = D - A$) to inject global root network coordinates into multi-head cross-attention.

```mermaid
flowchart TD
    NodeFeat["Node Coordinates and Degrees: N x 8"] --> NodeProj["Linear Node Feature Projection"]
    GraphLaplacian["Graph Laplacian Matrix: L = D - A"] --> Eigen["Eigenvalue Decomposition: Smallest Non-Trivial Eigenvectors"]
    Eigen --> LPE["Laplacian Positional Encoding: N x k"]
    
    NodeProj & LPE --> ConcatEmbed["Combined Node Representation"]
    ConcatEmbed --> GTLayer1["Graph Transformer Layer 1: Node-Edge Attention"]
    GTLayer1 --> GTLayer2["Graph Transformer Layer 2: Residual LayerNorm + FFN"]
    GTLayer2 --> GlobalPool["Global Mean + Max Pooling"]
    GlobalPool --> TopoVector["128-d Global Topological Vector"]
```

---

## ⚡ ONNX Runtime Benchmarks & Inference Guide

### Running ONNX Benchmarks
To export and benchmark all 4 vision models using ONNX Runtime:

```bash
python3 architecture/export_all_onnx.py
```

### Python Inference Script
```python
import onnxruntime as ort
import numpy as np

# Load ONNX model session
session = ort.InferenceSession("architecture/rhizo_hybrid_transformer.onnx")

# Prepare test tensor [1, 3, 128, 128]
dummy_img = np.random.randn(1, 3, 128, 128).astype(np.float32)

# Run inference
outputs = session.run(None, {"input": dummy_img})
output_mask = outputs[0]

print("✓ ONNX Inference successful! Output shape:", output_mask.shape)
```

---

## 📜 License
Distributed under the **Apache License 2.0**. See `LICENSE` for details.
