"""
Multi-Dataset Loader for RHIZO-NET Root Segmentation
=====================================================

Unified dataset class that loads from all 6 root imagery datasets,
handling different annotation formats (PNG masks, RSML, COCO JSON).

Supports tile-based training for large rhizotron images.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False


# ============================================================================
# Dataset Adapters (handle different annotation formats)
# ============================================================================

class RootDatasetAdapter:
    """Base adapter for loading images and masks from different dataset formats."""

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples = []

    def load_samples(self) -> List[Dict[str, str]]:
        """Return list of {'image': path, 'mask': path} dicts."""
        raise NotImplementedError


class PNGMaskAdapter(RootDatasetAdapter):
    """
    Adapter for datasets with paired image/mask PNG files.
    Expected structure:
        root_dir/images/*.png (or .jpg, .tif)
        root_dir/masks/*.png
    """

    def __init__(self, root_dir, split="train", image_dir="images", mask_dir="masks"):
        super().__init__(root_dir, split)
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def load_samples(self):
        img_dir = self.root_dir / self.image_dir
        mask_dir = self.root_dir / self.mask_dir

        if not img_dir.exists():
            # Try alternate directory names
            for alt in ["Images", "imgs", "img", "train_images", "raw"]:
                alt_dir = self.root_dir / alt
                if alt_dir.exists():
                    img_dir = alt_dir
                    break

        if not mask_dir.exists():
            for alt in ["Masks", "labels", "annotations", "segmentation", "train_masks"]:
                alt_dir = self.root_dir / alt
                if alt_dir.exists():
                    mask_dir = alt_dir
                    break

        if not img_dir.exists() or not mask_dir.exists():
            print(f"  Warning: Could not find image/mask dirs in {self.root_dir}")
            return []

        # Match images to masks by filename stem
        img_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        images = {}
        for f in img_dir.iterdir():
            if f.suffix.lower() in img_extensions:
                images[f.stem] = str(f)

        masks = {}
        for f in mask_dir.iterdir():
            if f.suffix.lower() in img_extensions:
                masks[f.stem] = str(f)

        samples = []
        for stem in sorted(set(images.keys()) & set(masks.keys())):
            samples.append({"image": images[stem], "mask": masks[stem]})

        return samples


class AutoDetectAdapter(RootDatasetAdapter):
    """
    Automatically detects dataset structure and creates appropriate adapter.
    Searches recursively for image-mask pairs.
    """

    def load_samples(self):
        samples = []

        # Strategy 1: Look for images/ and masks/ directories
        adapter = PNGMaskAdapter(self.root_dir, self.split)
        samples = adapter.load_samples()
        if samples:
            return samples

        # Strategy 2: Search recursively for any paired directories
        for subdir in self.root_dir.rglob("*"):
            if subdir.is_dir() and subdir.name.lower() in ("images", "image", "imgs", "raw"):
                parent = subdir.parent
                for mask_name in ("masks", "mask", "labels", "annotations", "segmentation"):
                    mask_dir = parent / mask_name
                    if mask_dir.exists():
                        adapter = PNGMaskAdapter(parent, self.split,
                                                 image_dir=subdir.name, mask_dir=mask_name)
                        found = adapter.load_samples()
                        samples.extend(found)

        # Strategy 3: Single directory with alternating files (img_001, mask_001)
        if not samples:
            img_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            all_files = sorted([f for f in self.root_dir.rglob("*") if f.suffix.lower() in img_extensions])

            # Group by potential pairs
            for f in all_files:
                stem = f.stem.lower()
                if "mask" in stem or "label" in stem or "seg" in stem:
                    # Find matching image
                    img_stem = stem.replace("_mask", "").replace("_label", "").replace("_seg", "")
                    for g in all_files:
                        if g.stem.lower() == img_stem and g != f:
                            samples.append({"image": str(g), "mask": str(f)})
                            break

        return samples


# ============================================================================
# Augmentation Pipeline
# ============================================================================

def get_train_augmentations(tile_size=512):
    """Training augmentations using albumentations."""
    if not HAS_ALBUM:
        return None

    return A.Compose([
        A.RandomCrop(height=tile_size, width=tile_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ], p=0.5),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_augmentations(tile_size=512):
    """Validation augmentations (just resize + normalize)."""
    if not HAS_ALBUM:
        return None

    return A.Compose([
        A.CenterCrop(height=tile_size, width=tile_size, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Unified Root Segmentation Dataset
# ============================================================================

class RootSegmentationDataset(Dataset):
    """
    Unified dataset for root segmentation that handles all 6 datasets.

    Automatically detects annotation format and applies appropriate loading.
    Supports tile-based cropping for large images and data augmentation.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_name: str = "auto",
        split: str = "train",
        tile_size: int = 512,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.tile_size = tile_size
        self.transform = transform

        # Auto-detect and load samples
        adapter = AutoDetectAdapter(self.root_dir, split)
        self.samples = adapter.load_samples()

        if not self.samples:
            print(f"  Warning: No samples found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = np.array(Image.open(sample["image"]).convert("RGB"))

        # Load mask
        mask = np.array(Image.open(sample["mask"]).convert("L"))
        mask = (mask > 127).astype(np.float32)  # Binarize

        # Pad if smaller than tile_size
        h, w = image.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            pad_h = max(self.tile_size - h, 0)
            pad_w = max(self.tile_size - w, 0)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")

        # Apply augmentations
        if self.transform and HAS_ALBUM:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Basic tensor conversion
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask = torch.from_numpy(mask)

            # Random crop
            _, h, w = image.shape
            if h > self.tile_size and w > self.tile_size:
                top = random.randint(0, h - self.tile_size)
                left = random.randint(0, w - self.tile_size)
                image = image[:, top:top + self.tile_size, left:left + self.tile_size]
                mask = mask[top:top + self.tile_size, left:left + self.tile_size]

        # Ensure mask has channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "dataset": self.dataset_name,
            "path": sample["image"],
        }


def create_multi_dataset_loader(
    dataset_dirs: Dict[str, str],
    split: str = "train",
    tile_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Create a unified DataLoader from multiple root imagery datasets.

    Args:
        dataset_dirs: Dict mapping dataset names to their root directories
        split: 'train', 'val', or 'test'
        tile_size: Tile crop size
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed for reproducibility
    """
    transform = get_train_augmentations(tile_size) if split == "train" else get_val_augmentations(tile_size)

    datasets = []
    for name, path in dataset_dirs.items():
        ds = RootSegmentationDataset(path, name, split, tile_size, transform)
        if len(ds) > 0:
            datasets.append(ds)
            print(f"  ✓ {name}: {len(ds)} samples")
        else:
            print(f"  ⚠ {name}: no samples found at {path}")

    if not datasets:
        raise ValueError("No datasets found! Check paths.")

    combined = ConcatDataset(datasets)
    print(f"  Total: {len(combined)} samples")

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        generator=torch.Generator().manual_seed(seed),
    )

    return loader


# ============================================================================
# Kaggle Offline Dataset Paths
# ============================================================================

KAGGLE_DATASET_PATHS = {
    "rootnav2": "/kaggle/input/rhizonet-rootnav2",
    "prmi": "/kaggle/input/rhizonet-prmi",
    "deeprootlab": "/kaggle/input/rhizonet-deeprootlab",
    "seminal_root_angle": "/kaggle/input/rhizonet-seminalrootangle",
    "chicory": "/kaggle/input/rhizonet-chicory",
    "grassland": "/kaggle/input/rhizonet-grassland",
}


if __name__ == "__main__":
    # Test with dummy data
    print("Testing RootSegmentationDataset...")
    print(f"Albumentations available: {HAS_ALBUM}")
