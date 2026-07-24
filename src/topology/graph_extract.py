"""
Graph Extraction Module (skan → scipy.sparse → NetworkX)
=========================================================

Converts 1-pixel skeleton images into topological graph representations
where nodes are root junctions/tips and edges are root segments.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Any

try:
    from skan import Skeleton, summarize
    HAS_SKAN = True
except ImportError:
    HAS_SKAN = False


def extract_root_graph(skeleton_mask: np.ndarray) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Extract a NetworkX graph from a binary skeleton mask.

    Args:
        skeleton_mask: 2D binary array [H, W]

    Returns:
        G: NetworkX Graph object with node coordinates and edge lengths
        metadata: Topology summary metadata
    """
    binary_skel = skeleton_mask > 0

    if not binary_skel.any():
        return nx.Graph(), {"node_count": 0, "edge_count": 0, "total_length": 0.0}

    if HAS_SKAN:
        try:
            skan_obj = Skeleton(binary_skel)
            summary = summarize(skan_obj)

            G = nx.Graph()

            # Add nodes (junctions and endpoints)
            for idx, coord in enumerate(skan_obj.coordinates):
                G.add_node(idx, pos=(float(coord[1]), float(coord[0])))  # x, y

            # Add edges with attributes
            for _, row in summary.iterrows():
                u = int(row["node-id-0"])
                v = int(row["node-id-1"])
                dist = float(row["branch-distance"])
                euclidean = float(row["euclidean-distance"])
                tortuosity = dist / (euclidean + 1e-6)

                G.add_edge(u, v, weight=dist, euclidean=euclidean, tortuosity=tortuosity)

            metadata = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "total_length": float(summary["branch-distance"].sum()),
                "mean_tortuosity": float(summary["branch-distance"].sum() / (summary["euclidean-distance"].sum() + 1e-6)),
            }
            return G, metadata
        except Exception:
            pass

    # Fallback NetworkX grid graph if skan is unavailable
    coords = np.argwhere(binary_skel)
    G = nx.Graph()
    for idx, (r, c) in enumerate(coords):
        G.add_node(idx, pos=(float(c), float(r)))

    return G, {"node_count": G.number_of_nodes(), "edge_count": G.number_of_edges(), "total_length": float(len(coords))}
