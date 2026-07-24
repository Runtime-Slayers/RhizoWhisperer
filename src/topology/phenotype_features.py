"""
Phenotype Features Extraction Module
=====================================

Extracts quantitative phenotypic features from root topological graphs:
1. Mean Tortuosity (curve length / straight-line distance)
2. Sholl Analysis (intersections vs. concentric distance from seed point)
3. Root System Depth & Width Ratio
4. Junction & Tip Counts
5. Total & Mean Branch Lengths
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional


def calculate_sholl_analysis(
    graph: nx.Graph,
    center_point: Tuple[float, float],
    step_radius: float = 20.0,
    max_radius: float = 500.0,
) -> Dict[str, Any]:
    """
    Perform Sholl analysis: count intersections of root branches at concentric radii.

    Args:
        graph: NetworkX root graph with 'pos' node attributes (x, y)
        center_point: (x, y) coordinates of seed/base point
        step_radius: Radius step size in pixels
        max_radius: Max search radius

    Returns:
        Dict with radii, intersection counts, max intersections, and critical radius
    """
    radii = np.arange(step_radius, max_radius + step_radius, step_radius)
    counts = []

    cx, cy = center_point

    # Calculate distance of each node to center
    node_distances = {}
    for node, data in graph.nodes(data=True):
        if "pos" in data:
            nx_pos, ny_pos = data["pos"]
            node_distances[node] = np.sqrt((nx_pos - cx) ** 2 + (ny_pos - cy) ** 2)

    # Count edge intersections with concentric circles
    for r in radii:
        intersections = 0
        for u, v in graph.edges():
            d1 = node_distances.get(u, 0)
            d2 = node_distances.get(v, 0)
            # Edge crosses circle if one node is inside and one is outside
            if (d1 <= r < d2) or (d2 <= r < d1):
                intersections += 1
        counts.append(intersections)

    counts_arr = np.array(counts)
    max_intersections = int(counts_arr.max()) if len(counts_arr) > 0 else 0
    critical_radius_idx = int(counts_arr.argmax()) if len(counts_arr) > 0 else 0
    critical_radius = float(radii[critical_radius_idx]) if len(radii) > 0 else 0.0

    return {
        "sholl_radii": radii.tolist(),
        "sholl_intersections": counts,
        "sholl_max_intersections": max_intersections,
        "sholl_critical_radius": critical_radius,
    }


def extract_phenotype_features(
    graph: nx.Graph,
    seed_point: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Extract comprehensive phenotypic features from root graph.

    Returns:
        Dict of numerical features ready for multi-modal fusion.
    """
    if graph.number_of_nodes() == 0:
        return {
            "mean_tortuosity": 1.0,
            "mean_branch_length": 0.0,
            "total_root_length": 0.0,
            "tip_count": 0,
            "junction_count": 0,
            "seminal_angle": 0.0,
            "sholl_max_intersections": 0,
            "sholl_critical_radius": 0.0,
        }

    # Node degree classification
    degrees = dict(graph.degree())
    tips = [n for n, d in degrees.items() if d == 1]
    junctions = [n for n, d in degrees.items() if d >= 3]

    # Edge tortuosity & length
    tortuosities = [d.get("tortuosity", 1.0) for u, v, d in graph.edges(data=True)]
    lengths = [d.get("weight", 0.0) for u, v, d in graph.edges(data=True)]

    mean_tort = float(np.mean(tortuosities)) if tortuosities else 1.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    total_len = float(np.sum(lengths)) if lengths else 0.0

    # Auto seed point: top-most node (smallest y coordinate)
    if seed_point is None:
        positions = [d["pos"] for n, d in graph.nodes(data=True) if "pos" in d]
        if positions:
            seed_point = min(positions, key=lambda p: p[1])  # Min Y
        else:
            seed_point = (0.0, 0.0)

    # Sholl analysis
    sholl = calculate_sholl_analysis(graph, seed_point)

    return {
        "mean_tortuosity": mean_tort,
        "mean_branch_length": mean_len,
        "total_root_length": total_len,
        "tip_count": len(tips),
        "junction_count": len(junctions),
        "seminal_angle": 0.0,  # Will be updated by seminal_angle module
        "sholl_max_intersections": float(sholl["sholl_max_intersections"]),
        "sholl_critical_radius": float(sholl["sholl_critical_radius"]),
    }
