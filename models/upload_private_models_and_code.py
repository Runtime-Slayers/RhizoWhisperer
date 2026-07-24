#!/usr/bin/env python3
"""
Upload RHIZO-NET Source Code and Model Architectures (ONNX) to Kaggle as a PRIVATE Dataset.
"""

import os
import sys
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
UPLOAD_DIR = Path("/tmp/rhizonet_private_dataset")


def prepare_and_upload():
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True)

    print(f"1. Copying source files to {UPLOAD_DIR}...")

    # Copy src, architecture, configs
    shutil.copytree(PROJECT_ROOT / "src", UPLOAD_DIR / "src")
    shutil.copytree(PROJECT_ROOT / "architecture", UPLOAD_DIR / "architecture")
    shutil.copytree(PROJECT_ROOT / "configs", UPLOAD_DIR / "configs")

    # Create dataset-metadata.json
    metadata = {
        "title": "RHIZO-NET Source Code and Model Architectures",
        "id": "saranboddu/rhizo-net-code-and-models",
        "subtitle": "Private dataset containing RHIZO-NET Python packages, configs, and ONNX models.",
        "description": "Source code and pre-exported ONNX models for RHIZO-NET root phenotyping pipeline.",
        "isPrivate": True,
        "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["rhizo-net", "plant-phenotyping", "deep-learning"],
    }

    with open(UPLOAD_DIR / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("2. Uploading PRIVATE dataset version to Kaggle...")
    cmd_version = f"kaggle datasets version -p {UPLOAD_DIR} -m 'Update code and fix typing imports' --dir-mode zip"
    exit_code = os.system(cmd_version)

    if exit_code != 0:
        print("Initial version update failed, trying create...")
        cmd_create = f"kaggle datasets create -p {UPLOAD_DIR} --dir-mode zip"
        os.system(cmd_create)

    print("✓ Upload completed: saranboddu/rhizo-net-code-and-models (Private)")


if __name__ == "__main__":
    prepare_and_upload()
