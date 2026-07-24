"""
Skeletonization Module using skan & skimage
============================================

Performs fast Numba/C-accelerated skeletonization of binary root segmentation masks
and prunes spurious small spur branches.
"""

import numpy as np
from skimage.morphology import skeletonize, remove_small_objects


def skeletonize_root_mask(
    mask: np.ndarray,
    min_size: int = 15,
    prune_spurs: bool = True,
    spur_length_threshold: int = 5,
) -> np.ndarray:
    """
    Skeletonize a binary root mask to 1-pixel wide centerlines.

    Args:
        mask: Binary mask [H, W] (boolean or uint8)
        min_size: Minimum pixel area to keep (filters isolated noise blobs)
        prune_spurs: Whether to remove short artifact spurs
        spur_length_threshold: Max spur length in pixels to remove

    Returns:
        Binary skeleton image [H, W] (uint8 0/255 or bool)
    """
    binary = mask > 0.5 if mask.dtype in (np.float32, np.float64) else mask > 0

    # 1. Remove small noise blobs
    if min_size > 0:
        binary = remove_small_objects(binary, min_size=min_size)

    # 2. Extract skeleton
    skel = skeletonize(binary, method="lee")

    # 3. Optional spur pruning using skan
    if prune_spurs and skel.any():
        try:
            from skan import Skeleton, summarize
            skan_skel = Skeleton(skel)
            summary = summarize(skan_skel)

            # Identify short degree-1 branch spurs
            spurs = summary[(summary["branch-type"] == 1) &
                            (summary["branch-distance"] < spur_length_threshold)]

            # Clear spur pixels from skeleton
            pruned_skel = skel.copy()
            for _, row in spurs.iterrows():
                node_idx = int(row["node-id-1"]) if "node-id-1" in row else None
                if node_idx is not None and node_idx < len(skan_skel.coordinates):
                    coord = skan_skel.coordinates[node_idx].astype(int)
                    pruned_skel[coord[0], coord[1]] = False

            return pruned_skel.astype(np.uint8) * 255
        except Exception:
            pass

    return skel.astype(np.uint8) * 255
