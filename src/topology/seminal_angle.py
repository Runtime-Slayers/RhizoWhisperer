"""
Seminal Root Angle Calculation Module
=======================================

Measures the angle between primary seminal root branches originating
from the seed coat/crown (critical phenotype for drought resilience & nutrient foraging).
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Optional


def calculate_seminal_root_angle(
    graph: nx.Graph,
    seed_point: Optional[Tuple[float, float]] = None,
    distance_threshold: float = 50.0,
) -> float:
    """
    Calculate the seminal root opening angle in degrees.

    Args:
        graph: NetworkX root graph
        seed_point: (x, y) seed location
        distance_threshold: Distance along branch to measure vector angle

    Returns:
        Angle in degrees (0 to 180)
    """
    if graph.number_of_nodes() < 3:
        return 0.0

    positions = {n: data["pos"] for n, data in graph.nodes(data=True) if "pos" in data}
    if not positions:
        return 0.0

    # Auto-detect seed point if not given (topmost node)
    if seed_point is None:
        seed_node = min(positions.keys(), key=lambda n: positions[n][1])
        seed_pos = positions[seed_node]
    else:
        seed_pos = seed_point
        # Find nearest graph node to seed_point
        seed_node = min(positions.keys(),
                        key=lambda n: (positions[n][0] - seed_pos[0])**2 + (positions[n][1] - seed_pos[1])**2)

    # Get primary branches extending from seed_node
    neighbors = list(graph.neighbors(seed_node))
    if len(neighbors) < 2:
        return 45.0  # Default nominal angle if single root

    # Calculate direction vectors for primary branches
    vectors = []
    for neighbor in neighbors:
        nx_pos, ny_pos = positions[neighbor]
        dx = nx_pos - seed_pos[0]
        dy = ny_pos - seed_pos[1]
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        vectors.append((dx / norm, dy / norm))

    # Find maximum opening angle between any pair of primary branches
    max_angle_deg = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            v1 = vectors[i]
            v2 = vectors[j]
            dot = np.clip(v1[0] * v2[0] + v1[1] * v2[1], -1.0, 1.0)
            angle_rad = np.arccos(dot)
            angle_deg = float(np.degrees(angle_rad))
            if angle_deg > max_angle_deg:
                max_angle_deg = angle_deg

    return max_angle_deg
