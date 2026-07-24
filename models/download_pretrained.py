#!/usr/bin/env python3
"""
Download pretrained model weights for RHIZO-NET.
- MobileSAM (mobile_sam.pt) from GitHub
"""

import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests")
    import requests


MODELS = {
    "mobilesam": {
        "url": "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt",
        "filename": "mobile_sam.pt",
        "description": "MobileSAM ViT-Tiny encoder pretrained weights (~40 MB)",
    },
}


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  ✓ Already exists: {dest_path}")
        return dest_path

    print(f"  Downloading: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r  Progress: {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)

    print(f"\n  ✓ Downloaded: {dest_path.name}")
    return dest_path


def main():
    output_dir = Path(__file__).parent / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("RHIZO-NET Pretrained Model Downloader")
    print("=" * 50)

    for name, model in MODELS.items():
        print(f"\n📦 {name}: {model['description']}")
        download_file(model["url"], output_dir / model["filename"])

    print(f"\n✓ All models downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
