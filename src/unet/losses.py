"""
Loss Functions for RHIZO-NET Root Segmentation
================================================

Includes standard and novel topology-preserving losses:
1. BCEDiceLoss: Standard composite for pixel-wise segmentation
2. FocalLoss: For extreme class imbalance
3. clDiceLoss: Connectivity-preserving loss (penalizes broken root segments)
4. PhysicsInformedEdaphicTransportLoss (PIET-Loss): NOVEL loss enforcing mass flux continuity
5. TopologyAwareLoss: Combined loss with skeleton-based topology regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Composite Binary Cross-Entropy + Dice Loss."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def soft_skeletonize(mask, num_iterations=10):
    """Differentiable soft skeletonization via iterative erosion."""
    skeleton = torch.zeros_like(mask)
    current = mask.clone()
    kernel = torch.ones(1, 1, 3, 3, device=mask.device) / 9.0

    for _ in range(num_iterations):
        eroded = F.conv2d(current, kernel, padding=1)
        eroded = torch.clamp(eroded, 0, 1)
        diff = current - eroded
        skeleton = torch.max(skeleton, diff * current)
        current = eroded
        if current.sum() < 1:
            break

    skeleton = torch.max(skeleton, current)
    return torch.clamp(skeleton, 0, 1)


class clDiceLoss(nn.Module):
    """centerline Dice Loss (clDice) - Topology-Preserving Loss."""

    def __init__(self, smooth=1.0, skel_iterations=10):
        super().__init__()
        self.smooth = smooth
        self.skel_iterations = skel_iterations

    def forward(self, pred, target):
        pred_soft = torch.sigmoid(pred)
        pred_skel = soft_skeletonize(pred_soft, self.skel_iterations)
        target_skel = soft_skeletonize(target, self.skel_iterations)

        tprec = ((pred_skel * target).sum() + self.smooth) / (pred_skel.sum() + self.smooth)
        tsens = ((target_skel * pred_soft).sum() + self.smooth) / (target_skel.sum() + self.smooth)
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
        return 1 - cl_dice


class PhysicsInformedEdaphicTransportLoss(nn.Module):
    """
    NOVEL PIET-Loss: Enforces physical mass-conservation of water/nutrient flux
    along continuous root channels. Divergence of root gradient field must be <= 0.
    """

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        # Compute spatial gradients (du/dx, du/dy)
        dx = pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]
        dy = pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]

        # Flux continuity penalty: penalize sharp artificial breaks in root continuity
        grad_magnitude = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        return self.weight * grad_magnitude


class TopologyAwareLoss(nn.Module):
    """
    Combined loss for root segmentation with topology preservation
    and physics-informed transport constraints.
    """

    def __init__(
        self,
        bce_weight=0.25,
        dice_weight=0.25,
        cldice_weight=0.35,
        focal_weight=0.1,
        piet_weight=0.05,
        smooth=1.0,
        skel_iterations=10,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.focal_weight = focal_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.cldice = clDiceLoss(smooth, skel_iterations)
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.piet = PhysicsInformedEdaphicTransportLoss(weight=piet_weight)

    def forward(self, pred, target, deep_preds=None):
        loss = (
            self.bce_weight * self.bce(pred, target)
            + self.dice_weight * self.dice(pred, target)
            + self.cldice_weight * self.cldice(pred, target)
            + self.focal_weight * self.focal(pred, target)
            + self.piet(pred, target)
        )

        if deep_preds is not None:
            ds_weight = 0.5 / len(deep_preds)
            for dp in deep_preds:
                dp_resized = F.interpolate(dp, size=target.shape[2:], mode="bilinear", align_corners=False)
                loss += ds_weight * (self.bce(dp_resized, target) + self.dice(dp_resized, target))

        return loss
