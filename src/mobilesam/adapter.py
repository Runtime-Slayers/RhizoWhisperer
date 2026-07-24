"""
MobileSAM Adapter for RHIZO-NET
================================

Provides a fallback segmentation path using MobileSAM when the
primary CNN models (RhizoUNet, etc.) have low confidence.

The adapter wraps MobileSAM's prompt-based segmentation with automatic
prompt generation from CNN confidence maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class MobileSAMAdapter(nn.Module):
    """
    Adapter that routes low-confidence predictions to MobileSAM.

    Pipeline:
    1. Primary model (e.g., RhizoUNet) produces segmentation + confidence
    2. If mean confidence < threshold, MobileSAM is invoked
    3. MobileSAM receives automatic prompts from CNN attention maps
    4. Final output is a confidence-weighted blend of both predictions

    Args:
        primary_model: Primary segmentation model (RhizoUNet, etc.)
        mobilesam_weights: Path to mobile_sam.pt weights
        confidence_threshold: Routing threshold (default: 0.5)
    """

    def __init__(self, primary_model, mobilesam_weights=None, confidence_threshold=0.5):
        super().__init__()
        self.primary = primary_model
        self.confidence_threshold = confidence_threshold
        self.sam_loaded = False

        # Try to load MobileSAM
        if mobilesam_weights and Path(mobilesam_weights).exists():
            self._load_mobilesam(mobilesam_weights)

    def _load_mobilesam(self, weights_path):
        """Load MobileSAM model."""
        try:
            # Try ultralytics SAM
            from ultralytics import SAM
            self.sam_model = SAM(weights_path)
            self.sam_loaded = True
            print(f"✓ MobileSAM loaded from: {weights_path}")
        except ImportError:
            try:
                # Try mobile_sam direct import
                from mobile_sam import sam_model_registry, SamPredictor
                model_type = "vit_t"
                sam = sam_model_registry[model_type](checkpoint=weights_path)
                self.sam_predictor = SamPredictor(sam)
                self.sam_loaded = True
                print(f"✓ MobileSAM loaded (direct) from: {weights_path}")
            except ImportError:
                print("⚠ MobileSAM not available. Install with: pip install mobile-sam or ultralytics")
                self.sam_loaded = False

    def generate_point_prompts(self, confidence_map, num_points=5):
        """
        Generate point prompts from regions where CNN is most uncertain.
        These points guide MobileSAM to focus on ambiguous areas.
        """
        # Find regions of high uncertainty (confidence near 0.5)
        uncertainty = 1 - torch.abs(confidence_map - 0.5) * 2  # 1 = most uncertain
        uncertainty_np = uncertainty.squeeze().cpu().numpy()

        # Sample points weighted by uncertainty
        flat = uncertainty_np.flatten()
        flat = flat / (flat.sum() + 1e-8)

        h, w = uncertainty_np.shape
        indices = np.random.choice(len(flat), size=num_points, replace=False, p=flat)
        points = np.array([(idx % w, idx // w) for idx in indices])

        return points

    def forward(self, images, force_sam=False):
        """
        Forward pass with confidence-based routing.

        Args:
            images: [B, 3, H, W] tensor
            force_sam: Force MobileSAM usage regardless of confidence

        Returns:
            segmentation mask [B, 1, H, W]
        """
        # Step 1: Primary model prediction
        with torch.no_grad() if not self.training else torch.enable_grad():
            primary_logits = self.primary(images)
            primary_probs = torch.sigmoid(primary_logits)
            confidence = primary_probs.mean(dim=[1, 2, 3])  # Per-image confidence

        # Step 2: Check confidence and route
        if not force_sam and (not self.sam_loaded or confidence.min() > self.confidence_threshold):
            return primary_logits

        # Step 3: For low-confidence images, use MobileSAM
        outputs = []
        for i in range(images.size(0)):
            if confidence[i] > self.confidence_threshold and not force_sam:
                outputs.append(primary_logits[i:i+1])
            elif self.sam_loaded:
                try:
                    sam_mask = self._run_mobilesam(images[i], primary_probs[i])
                    # Blend: confidence-weighted combination
                    alpha = confidence[i].clamp(0.2, 0.8)
                    blended = alpha * primary_probs[i:i+1] + (1 - alpha) * sam_mask
                    outputs.append(torch.logit(blended.clamp(1e-6, 1 - 1e-6)))
                except Exception:
                    outputs.append(primary_logits[i:i+1])
            else:
                outputs.append(primary_logits[i:i+1])

        return torch.cat(outputs, dim=0)

    def _run_mobilesam(self, image_tensor, confidence_map):
        """Run MobileSAM on a single image with auto-generated prompts."""
        # Convert to numpy
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Generate prompts from uncertain regions
        points = self.generate_point_prompts(confidence_map)
        labels = np.ones(len(points))  # All positive prompts

        if hasattr(self, "sam_predictor"):
            self.sam_predictor.set_image(image_np)
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            # Use highest-scoring mask
            best_idx = scores.argmax()
            mask = masks[best_idx]
        elif hasattr(self, "sam_model"):
            results = self.sam_model.predict(image_np, points=points, labels=labels)
            mask = results[0].masks.data[0].cpu().numpy()
        else:
            return confidence_map.unsqueeze(0)

        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.to(image_tensor.device)

        # Resize to match input
        if mask_tensor.shape[2:] != image_tensor.shape[1:]:
            mask_tensor = F.interpolate(mask_tensor, size=image_tensor.shape[1:],
                                         mode="bilinear", align_corners=False)

        return mask_tensor
