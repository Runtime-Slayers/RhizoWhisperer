"""
Visualization Utilities for RHIZO-NET
======================================

Visualization functions for images, masks, skeletons, topological graphs,
and full pipeline recommendation outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, Dict, Any, Optional


def plot_root_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    title: str = "Root Segmentation",
    save_path: Optional[str] = None,
):
    """Plot RGB image, ground truth mask, and predicted mask side-by-side."""
    num_cols = 3 if pred_mask is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    if pred_mask is not None:
        axes[2].imshow(pred_mask, cmap="magma")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_root_graph(
    graph: nx.Graph,
    background_image: Optional[np.ndarray] = None,
    title: str = "Topological Root Graph",
    save_path: Optional[str] = None,
):
    """Plot extracted NetworkX root graph over background image."""
    fig, ax = plt.subplots(figsize=(8, 8))

    if background_image is not None:
        ax.imshow(background_image, cmap="gray" if background_image.ndim == 2 else None)

    pos = nx.get_node_attributes(graph, "pos")
    if not pos:
        pos = {n: (i, i) for i, n in enumerate(graph.nodes())}

    # Draw edges
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="lime", width=1.5, alpha=0.8)

    # Draw nodes (color by degree: 1=tip [red], 3+=junction [cyan], 2=segment [green])
    node_colors = []
    for n in graph.nodes():
        deg = graph.degree(n)
        if deg == 1:
            node_colors.append("red")      # Tip
        elif deg >= 3:
            node_colors.append("cyan")     # Junction
        else:
            node_colors.append("yellow")   # Intermediate

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=20)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_pipeline_summary(
    image: np.ndarray,
    mask: np.ndarray,
    graph: nx.Graph,
    recommendation: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """Plot end-to-end RHIZO-NET pipeline execution summary."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    # 1. Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title("1. Input Root Image", fontweight="bold")
    ax1.axis("off")

    # 2. Predicted Mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap="magma")
    ax2.set_title("2. Segmentation Mask", fontweight="bold")
    ax2.axis("off")

    # 3. Graph Topology
    ax3 = fig.add_subplot(gs[0, 2])
    plot_root_graph(graph, mask, title="3. Root Graph Topology", save_path=None)

    # 4. Recommendation Report
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")

    report_text = (
        f"RHIZO-NET AGRONOMIC RECOMMENDATION REPORT\n"
        f"{'='*60}\n"
        f"Crop: {recommendation.get('crop', 'N/A')}\n"
        f"Predicted Deficiency State: {recommendation.get('deficiency_state', 'N/A').upper()}\n\n"
        f"Soil Fertility Ratings: {recommendation.get('soil_ratings', {})}\n"
        f"Blanket NPK (kg/ha): {recommendation.get('blanket_recommendation_npk', {})}\n"
        f"PRESCRIBED NPK (kg/ha): {recommendation.get('prescribed_recommendation_npk', {})}\n\n"
        f"Special Interventions: {recommendation.get('special_interventions', [])}\n"
        f"Notes: {recommendation.get('notes', [])}\n"
    )

    ax4.text(
        0.05, 0.95, report_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
