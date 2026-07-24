#!/usr/bin/env python3
"""
RHIZO-NET Dataset Download Script
Downloads all required datasets from their sources (Zenodo, Dryad, Figshare)
and also pulls Kaggle-native datasets for soil/fertilizer data.

Run this ONCE locally (with internet) before using Kaggle offline notebooks.

Usage:
    python download_all_datasets.py --output-dir ./raw
"""

import os
import sys
import json
import time
import zipfile
import tarfile
import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS = {
    # ---- ROOT IMAGERY DATASETS (External - must download & upload to Kaggle) ----
    "rootnav2": {
        "name": "RootNav 2.0 - Wheat Seminal Root Images",
        "source": "zenodo",
        "zenodo_record_id": "3270726",
        "description": "Wheat (Triticum aestivum) root images with dense pixel & topology annotations",
        "species": "wheat",
        "modality": "pouch/flatbed",
        "kaggle_slug": "saranboddu/rhizonet-rootnav2",
    },
    "deeprootlab": {
        "name": "DeepRootLab - 11 Herbaceous Species",
        "source": "zenodo",
        "zenodo_record_id": "15213661",
        "description": "Rhizotron images of 11 species (chicory, fescue, etc.) with segmentation masks",
        "species": "multi-herbaceous",
        "modality": "rhizotron",
        "kaggle_slug": "saranboddu/rhizonet-deeprootlab",
    },
    "seminal_root_angle": {
        "name": "SeminalRootAngle - Spring Barley",
        "source": "zenodo",
        "zenodo_record_id": "7870965",
        "description": "Spring barley (Hordeum vulgare) rhizobox images with corrective seed/root masks",
        "species": "barley",
        "modality": "rhizobox",
        "kaggle_slug": "saranboddu/rhizonet-seminalrootangle",
    },
    "chicory": {
        "name": "Chicory Root Subset",
        "source": "zenodo",
        "zenodo_record_id": "3527713",
        "description": "Chicory (Cichorium intybus) field soil root images",
        "species": "chicory",
        "modality": "field-soil",
        "kaggle_slug": "saranboddu/rhizonet-chicory",
    },
    "grassland": {
        "name": "Grassland - Alpine Minirhizotron",
        "source": "figshare",
        "figshare_article_id": "20440497",
        "figshare_version": "2",
        "description": "Alpine mixed flora field minirhizotron images with natural soil interactions",
        "species": "alpine-mixed",
        "modality": "minirhizotron",
        "kaggle_slug": "saranboddu/rhizonet-grassland",
    },
    "prmi": {
        "name": "PRMI - Plant Root Minirhizotron Imagery",
        "source": "dryad",
        "dryad_doi": "10.5061/dryad.gtht76hkv",
        "description": "72K+ RGB images across 6 species with pixel-level binary masks",
        "species": ["peanut", "cotton", "switchgrass", "papaya", "sesame", "sunflower"],
        "modality": "minirhizotron",
        "kaggle_slug": "saranboddu/rhizonet-prmi",
    },

    # ---- KAGGLE-NATIVE DATASETS (download via Kaggle API) ----
    "soil_nutrients_india": {
        "name": "Soil Nutrient Dataset - Southern Indian States",
        "source": "kaggle",
        "kaggle_dataset": "ravirajsinh45/soil-nutrient-dataset-of-southern-indian-states",
        "description": "N, P, K, pH for Tamil Nadu, Karnataka, Kerala, Goa",
    },
    "fertilizer_recommendation": {
        "name": "Fertilizer Recommendation Dataset",
        "source": "kaggle",
        "kaggle_dataset": "namanmanchanda/fertilizer-prediction",
        "description": "NPK levels + soil type + crop → fertilizer type prediction",
    },
    "crop_recommendation": {
        "name": "Crop Recommendation Dataset",
        "source": "kaggle",
        "kaggle_dataset": "atharvaingle/crop-recommendation-dataset",
        "description": "20K records: NPK, temp, humidity, pH, rainfall → crop type",
    },
    "mobilesam_weights": {
        "name": "MobileSAM Pretrained Weights",
        "source": "kaggle",
        "kaggle_dataset": "tiansz/mobilesam-model",
        "description": "mobile_sam.pt (~40 MB) pretrained ViT-Tiny encoder weights",
    },
}


# ============================================================================
# Download Helpers
# ============================================================================

def download_file(url, dest_path, chunk_size=8192, max_retries=3):
    """Download a file with progress display and retry logic."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            print(f"  Downloading: {url}")
            print(f"  Destination: {dest_path}")
            
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  Progress: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)
            
            print(f"\n  ✓ Downloaded: {dest_path.name} ({downloaded / (1024*1024):.1f} MB)")
            return dest_path
            
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"\n  ✗ Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ✗ FAILED after {max_retries} attempts: {url}")
                return None


def extract_archive(archive_path, extract_to):
    """Extract ZIP or TAR archive."""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        print(f"  Extracting ZIP: {archive_path.name}")
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_to)
    elif archive_path.suffix in ('.tar', '.gz', '.tgz', '.bz2'):
        print(f"  Extracting TAR: {archive_path.name}")
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_to)
    else:
        print(f"  Skipping extraction (unknown format): {archive_path.name}")
        return
    
    print(f"  ✓ Extracted to: {extract_to}")


# ============================================================================
# Source-Specific Downloaders
# ============================================================================

def download_zenodo(record_id, output_dir):
    """Download all files from a Zenodo record."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"  Fetching Zenodo record metadata: {record_id}")
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        record = response.json()
    except Exception as e:
        print(f"  ✗ Failed to fetch Zenodo record {record_id}: {e}")
        return []
    
    files = record.get("files", [])
    if not files:
        print(f"  ✗ No files found in Zenodo record {record_id}")
        return []
    
    downloaded_files = []
    for file_info in files:
        file_url = file_info["links"]["self"]
        filename = file_info["key"]
        file_size = file_info.get("size", 0)
        
        print(f"\n  File: {filename} ({file_size / (1024*1024):.1f} MB)")
        dest = Path(output_dir) / filename
        
        if dest.exists() and dest.stat().st_size == file_size:
            print(f"  ✓ Already downloaded: {filename}")
            downloaded_files.append(dest)
            continue
        
        result = download_file(file_url, dest)
        if result:
            downloaded_files.append(result)
    
    return downloaded_files


def download_figshare(article_id, version, output_dir):
    """Download all files from a Figshare article."""
    api_url = f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}"
    print(f"  Fetching Figshare article metadata: {article_id} (v{version})")
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        article = response.json()
    except Exception as e:
        print(f"  ✗ Failed to fetch Figshare article {article_id}: {e}")
        return []
    
    files = article.get("files", [])
    if not files:
        print(f"  ✗ No files found in Figshare article {article_id}")
        return []
    
    downloaded_files = []
    for file_info in files:
        file_url = file_info["download_url"]
        filename = file_info["name"]
        file_size = file_info.get("size", 0)
        
        print(f"\n  File: {filename} ({file_size / (1024*1024):.1f} MB)")
        dest = Path(output_dir) / filename
        
        if dest.exists() and dest.stat().st_size == file_size:
            print(f"  ✓ Already downloaded: {filename}")
            downloaded_files.append(dest)
            continue
        
        result = download_file(file_url, dest)
        if result:
            downloaded_files.append(result)
    
    return downloaded_files


def download_dryad(doi, output_dir):
    """Download dataset from Dryad by DOI."""
    # Dryad API v2
    api_url = f"https://datadryad.org/api/v2/datasets/doi%3A{doi.replace('/', '%2F')}"
    print(f"  Fetching Dryad dataset metadata: {doi}")
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        dataset = response.json()
    except Exception as e:
        print(f"  ✗ Failed to fetch Dryad dataset {doi}: {e}")
        print(f"  Trying direct download URL...")
        # Fallback: try direct download
        direct_url = f"https://datadryad.org/stash/downloads/file_stream/{doi}"
        print(f"  Note: PRMI is a large dataset (~8GB). Consider downloading manually from:")
        print(f"  https://datadryad.org/stash/dataset/doi:{doi}")
        return []
    
    # Get the version files
    version_url = dataset.get("_links", {}).get("stash:version", {}).get("href", "")
    if version_url:
        try:
            ver_response = requests.get(f"https://datadryad.org{version_url}", timeout=30)
            ver_response.raise_for_status()
            version = ver_response.json()
            
            files_url = version.get("_links", {}).get("stash:files", {}).get("href", "")
            if files_url:
                files_response = requests.get(f"https://datadryad.org{files_url}", timeout=30)
                files_response.raise_for_status()
                files_data = files_response.json()
                
                downloaded_files = []
                for file_info in files_data.get("_embedded", {}).get("stash:files", []):
                    file_url = file_info.get("_links", {}).get("stash:file-download", {}).get("href", "")
                    filename = file_info.get("path", "unknown")
                    file_size = file_info.get("size", 0)
                    
                    print(f"\n  File: {filename} ({file_size / (1024*1024):.1f} MB)")
                    dest = Path(output_dir) / filename
                    
                    if dest.exists() and dest.stat().st_size == file_size:
                        print(f"  ✓ Already downloaded: {filename}")
                        downloaded_files.append(dest)
                        continue
                    
                    if file_url:
                        result = download_file(f"https://datadryad.org{file_url}", dest)
                        if result:
                            downloaded_files.append(result)
                
                return downloaded_files
        except Exception as e:
            print(f"  ✗ Failed to get Dryad files: {e}")
    
    print(f"  Note: PRMI dataset may require manual download from:")
    print(f"  https://datadryad.org/stash/dataset/doi:{doi}")
    return []


def download_kaggle_dataset(dataset_slug, output_dir):
    """Download a Kaggle dataset using the Kaggle API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Downloading Kaggle dataset: {dataset_slug}")
    
    # Check if kaggle CLI is available
    kaggle_cmd = f"kaggle datasets download -d {dataset_slug} -p {output_dir} --unzip"
    print(f"  Running: {kaggle_cmd}")
    
    exit_code = os.system(kaggle_cmd)
    if exit_code == 0:
        print(f"  ✓ Downloaded: {dataset_slug}")
        return True
    else:
        print(f"  ✗ Kaggle CLI failed (exit code {exit_code})")
        print(f"  Make sure kaggle.json is configured and kaggle package is installed")
        return False


# ============================================================================
# Main Download Orchestrator
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RHIZO-NET Dataset Downloader")
    parser.add_argument("--output-dir", type=str, default="./datasets/raw",
                        help="Base output directory for downloaded datasets")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Specific datasets to download (default: all)")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle-native datasets")
    parser.add_argument("--skip-external", action="store_true",
                        help="Skip external (Zenodo/Dryad/Figshare) datasets")
    parser.add_argument("--extract", action="store_true", default=True,
                        help="Extract downloaded archives")
    args = parser.parse_args()
    
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
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
            print(f"✓ Kaggle credentials configured: {kaggle_dest}")
    
    # Determine which datasets to download
    dataset_keys = args.datasets or list(DATASETS.keys())
    
    print("=" * 70)
    print("RHIZO-NET Dataset Downloader")
    print("=" * 70)
    print(f"Output directory: {output_base.resolve()}")
    print(f"Datasets to download: {len(dataset_keys)}")
    print("=" * 70)
    
    results = {}
    
    for key in dataset_keys:
        if key not in DATASETS:
            print(f"\n⚠ Unknown dataset: {key}")
            continue
        
        ds = DATASETS[key]
        source = ds["source"]
        
        if args.skip_kaggle and source == "kaggle":
            continue
        if args.skip_external and source != "kaggle":
            continue
        
        dataset_dir = output_base / key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'─' * 60}")
        print(f"📦 [{key}] {ds['name']}")
        print(f"   Source: {source}")
        print(f"   Description: {ds['description']}")
        print(f"{'─' * 60}")
        
        try:
            if source == "zenodo":
                files = download_zenodo(ds["zenodo_record_id"], dataset_dir)
                if args.extract and files:
                    for f in files:
                        if f.suffix in ('.zip', '.tar', '.gz', '.tgz', '.bz2'):
                            extract_archive(f, dataset_dir / "extracted")
                results[key] = {"status": "success" if files else "failed", "files": len(files)}
                
            elif source == "figshare":
                files = download_figshare(ds["figshare_article_id"], ds["figshare_version"], dataset_dir)
                if args.extract and files:
                    for f in files:
                        if f.suffix in ('.zip', '.tar', '.gz', '.tgz', '.bz2'):
                            extract_archive(f, dataset_dir / "extracted")
                results[key] = {"status": "success" if files else "failed", "files": len(files)}
                
            elif source == "dryad":
                files = download_dryad(ds["dryad_doi"], dataset_dir)
                if args.extract and files:
                    for f in files:
                        if f.suffix in ('.zip', '.tar', '.gz', '.tgz', '.bz2'):
                            extract_archive(f, dataset_dir / "extracted")
                results[key] = {"status": "success" if files else "manual_needed", "files": len(files)}
                
            elif source == "kaggle":
                success = download_kaggle_dataset(ds["kaggle_dataset"], dataset_dir)
                results[key] = {"status": "success" if success else "failed"}
                
        except Exception as e:
            print(f"  ✗ Error downloading {key}: {e}")
            results[key] = {"status": "error", "error": str(e)}
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    for key, result in results.items():
        status_emoji = "✓" if result["status"] == "success" else "⚠" if result["status"] == "manual_needed" else "✗"
        print(f"  {status_emoji} {key}: {result['status']}")
    
    # Save manifest
    manifest_path = output_base / "download_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({"datasets": results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\n✓ Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
