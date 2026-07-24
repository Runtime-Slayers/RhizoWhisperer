"""
Kaggle Environment Utilities
============================

Helpers for running inside Kaggle environment (offline detection, GPU setup, dataset path resolution).
"""

import os
from pathlib import Path


def is_kaggle_environment() -> bool:
    """Check if code is running inside a Kaggle notebook."""
    return os.path.exists("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def setup_kaggle_env():
    """Setup Kaggle runtime environment settings."""
    if is_kaggle_environment():
        print("✓ Detected Kaggle Runtime Environment")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def resolve_kaggle_input_path(dataset_name: str) -> str:
    """Resolve local vs Kaggle dataset path."""
    if is_kaggle_environment():
        kaggle_path = Path(f"/kaggle/input/{dataset_name}")
        if kaggle_path.exists():
            return str(kaggle_path)
    return str(Path(f"./datasets/raw/{dataset_name}"))
