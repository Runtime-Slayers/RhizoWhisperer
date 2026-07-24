#!/usr/bin/env python3
"""
Upload downloaded datasets to Kaggle as private datasets.
Each dataset gets its own Kaggle dataset slug under your account.

Usage:
    python upload_to_kaggle.py --input-dir ./datasets/raw
"""

import os
import sys
import json
import argparse
from pathlib import Path


KAGGLE_DATASETS = {
    "rootnav2": {
        "slug": "rhizonet-rootnav2",
        "title": "RHIZO-NET: RootNav 2.0 Wheat Root Images",
        "subtitle": "Wheat seminal root images with dense pixel & topology annotations (Zenodo 3270726)",
    },
    "prmi": {
        "slug": "rhizonet-prmi",
        "title": "RHIZO-NET: PRMI Plant Root Minirhizotron Imagery",
        "subtitle": "72K+ RGB minirhizotron images across 6 species with pixel-level masks",
    },
    "deeprootlab": {
        "slug": "rhizonet-deeprootlab",
        "title": "RHIZO-NET: DeepRootLab 11-Species Rhizotron",
        "subtitle": "Rhizotron images of 11 herbaceous species with segmentation masks (Zenodo 15213661)",
    },
    "seminal_root_angle": {
        "slug": "rhizonet-seminalrootangle",
        "title": "RHIZO-NET: Seminal Root Angle Spring Barley",
        "subtitle": "Spring barley rhizobox images with corrective seed/root masks (Zenodo 7870965)",
    },
    "chicory": {
        "slug": "rhizonet-chicory",
        "title": "RHIZO-NET: Chicory Root Subset",
        "subtitle": "Chicory field soil root images (Zenodo 3527713)",
    },
    "grassland": {
        "slug": "rhizonet-grassland",
        "title": "RHIZO-NET: Grassland Alpine Minirhizotron",
        "subtitle": "Alpine mixed flora field minirhizotron images (Figshare 20440497)",
    },
}


def create_dataset_metadata(dataset_dir, slug, title, subtitle, username="saranboddu"):
    """Create dataset-metadata.json for Kaggle dataset upload."""
    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "subtitle": subtitle,
        "description": f"Part of the RHIZO-NET project: Root Health and Integrated Zonal Optimization Network via Edaphic Topology. {subtitle}",
        "isPrivate": True,
        "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": [
            "rhizo-net",
            "root-segmentation",
            "plant-phenotyping",
            "deep-learning",
            "agriculture",
        ],
    }

    metadata_path = dataset_dir / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Created metadata: {metadata_path}")
    return metadata_path


def upload_dataset(dataset_dir, slug, is_new=True):
    """Upload a dataset directory to Kaggle."""
    if is_new:
        cmd = f"kaggle datasets create -p {dataset_dir} --dir-mode zip"
    else:
        cmd = f"kaggle datasets version -p {dataset_dir} -m 'Updated data' --dir-mode zip"

    print(f"  Running: {cmd}")
    exit_code = os.system(cmd)

    if exit_code == 0:
        print(f"  ✓ Uploaded successfully: {slug}")
    else:
        print(f"  ✗ Upload failed (exit {exit_code}). Trying as new version...")
        if is_new:
            upload_dataset(dataset_dir, slug, is_new=False)

    return exit_code == 0


def main():
    parser = argparse.ArgumentParser(description="Upload RHIZO-NET datasets to Kaggle")
    parser.add_argument("--input-dir", type=str, default="./datasets/raw",
                        help="Directory containing downloaded datasets")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Specific datasets to upload (default: all)")
    parser.add_argument("--username", type=str, default="saranboddu",
                        help="Kaggle username")
    args = parser.parse_args()

    input_base = Path(args.input_dir)

    # Setup Kaggle credentials
    kaggle_json = Path(__file__).parent.parent / "kaggle.json"
    if kaggle_json.exists():
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_dest = kaggle_dir / "kaggle.json"
        if not kaggle_dest.exists():
            import shutil
            shutil.copy2(kaggle_json, kaggle_dest)
            os.chmod(kaggle_dest, 0o600)

    dataset_keys = args.datasets or list(KAGGLE_DATASETS.keys())

    print("=" * 60)
    print("RHIZO-NET Kaggle Dataset Uploader")
    print("=" * 60)

    for key in dataset_keys:
        if key not in KAGGLE_DATASETS:
            print(f"\n⚠ Unknown dataset: {key}")
            continue

        ds = KAGGLE_DATASETS[key]
        dataset_dir = input_base / key

        if not dataset_dir.exists():
            print(f"\n⚠ Directory not found: {dataset_dir} (skipping {key})")
            continue

        print(f"\n{'─' * 50}")
        print(f"📤 Uploading: {ds['title']}")
        print(f"   Slug: {args.username}/{ds['slug']}")
        print(f"{'─' * 50}")

        create_dataset_metadata(dataset_dir, ds["slug"], ds["title"], ds["subtitle"], args.username)
        upload_dataset(dataset_dir, ds["slug"])

    print(f"\n{'=' * 60}")
    print("Upload complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
