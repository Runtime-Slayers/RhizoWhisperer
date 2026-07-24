"""
SoilGrids Feature Preprocessing & Conversion Module
===================================================

Converts SoilGrids mapped unit values into standard agronomic units
and computes calibrated soil vectors for the multi-modal model.
"""

from typing import Dict, Any, List
import numpy as np


# Conversion factors (SoilGrids API mapped units -> standard units)
SOILGRIDS_CONVERSIONS = {
    "bdod": 0.01,       # cg/cm³ -> kg/dm³ (÷100)
    "cec": 0.1,         # mmol(c)/kg -> cmol(c)/kg (÷10)
    "clay": 0.1,        # g/kg -> % (÷10)
    "nitrogen": 0.01,   # cg/kg -> g/kg (÷100)
    "soc": 0.1,         # dg/kg -> g/kg (÷10)
    "phh2o": 0.1,       # pH * 10 -> pH (÷10)
    "wv0033": 0.1,      # v‰ -> v% (÷10)
}


def process_soilgrids_features(raw_soil_data: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw SoilGrids API dictionary to standard agronomic units.

    Args:
        raw_soil_data: Dict of raw SoilGrids property values

    Returns:
        Dict of converted, standard agronomic values
    """
    converted = {}
    for key, val in raw_soil_data.items():
        base_key = key.split("_")[0]  # e.g., 'bdod_0-5cm' -> 'bdod'
        factor = SOILGRIDS_CONVERSIONS.get(base_key, 1.0)
        converted[key] = val * factor

    return converted


def build_numerical_vector(
    soil_features: Dict[str, float],
    phenotype_features: Dict[str, float],
) -> np.ndarray:
    """
    Build unified 17-element numerical feature vector for PyTorch Frame / RhizoFusionNet.

    Features (in exact order):
    0: bulk_density (kg/dm³)
    1: cec (cmol(c)/kg)
    2: clay (%)
    3: nitrogen (g/kg)
    4: soc (g/kg)
    5: ph (pH)
    6: volumetric_water (%)
    7: mean_tortuosity
    8: mean_branch_length
    9: total_root_length
    10: tip_count
    11: junction_count
    12: seminal_angle
    13: sholl_max_intersections
    14: sholl_critical_radius
    15: mean_pixel_intensity
    16: stdev_pixel_intensity
    """
    vector = [
        soil_features.get("bdod", 1.3),
        soil_features.get("cec", 15.0),
        soil_features.get("clay", 25.0),
        soil_features.get("nitrogen", 1.5),
        soil_features.get("soc", 10.0),
        soil_features.get("phh2o", 6.5),
        soil_features.get("wv0033", 20.0),
        phenotype_features.get("mean_tortuosity", 1.1),
        phenotype_features.get("mean_branch_length", 25.0),
        phenotype_features.get("total_root_length", 500.0),
        phenotype_features.get("tip_count", 15.0),
        phenotype_features.get("junction_count", 10.0),
        phenotype_features.get("seminal_angle", 45.0),
        phenotype_features.get("sholl_max_intersections", 8.0),
        phenotype_features.get("sholl_critical_radius", 100.0),
        phenotype_features.get("mean_pixel_intensity", 120.0),
        phenotype_features.get("stdev_pixel_intensity", 35.0),
    ]

    return np.array(vector, dtype=np.float32)
