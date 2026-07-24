"""
TNAU Soil Fertility Rating Module
=================================

Rates soil fertility (Low, Medium, High) based on Tamil Nadu Agricultural University
(TNAU) Soil Rating Chart thresholds for N, P, K, pH, and Organic Carbon.
"""

from typing import Dict, Any


TNAU_THRESHOLDS = {
    "available_n_kg_ha": {"low": 240, "high": 480},
    "available_p_kg_ha": {"low": 11, "high": 22},
    "available_k_kg_ha": {"low": 110, "high": 280},
    "organic_carbon_pct": {"low": 0.5, "high": 0.75},
}


def rate_soil_fertility(
    n_kg_ha: float,
    p_kg_ha: float,
    k_kg_ha: float,
    ph: float,
    oc_pct: float = 0.5,
) -> Dict[str, str]:
    """
    Rate soil parameters into Low/Medium/High according to TNAU Agritech Portal standards.

    Returns:
        Dict with ratings for 'N', 'P', 'K', 'pH', and 'OC'.
    """
    # Nitrogen
    if n_kg_ha < TNAU_THRESHOLDS["available_n_kg_ha"]["low"]:
        n_rating = "low"
    elif n_kg_ha <= TNAU_THRESHOLDS["available_n_kg_ha"]["high"]:
        n_rating = "medium"
    else:
        n_rating = "high"

    # Phosphorus
    if p_kg_ha < TNAU_THRESHOLDS["available_p_kg_ha"]["low"]:
        p_rating = "low"
    elif p_kg_ha <= TNAU_THRESHOLDS["available_p_kg_ha"]["high"]:
        p_rating = "medium"
    else:
        p_rating = "high"

    # Potassium
    if k_kg_ha < TNAU_THRESHOLDS["available_k_kg_ha"]["low"]:
        k_rating = "low"
    elif k_kg_ha <= TNAU_THRESHOLDS["available_k_kg_ha"]["high"]:
        k_rating = "medium"
    else:
        k_rating = "high"

    # pH classification
    if ph < 5.0:
        ph_rating = "highly_acidic"
    elif ph < 6.5:
        ph_rating = "acidic"
    elif ph <= 7.5:
        ph_rating = "neutral"
    elif ph <= 8.5:
        ph_rating = "alkaline"
    else:
        ph_rating = "highly_alkaline"

    # Organic Carbon
    if oc_pct < TNAU_THRESHOLDS["organic_carbon_pct"]["low"]:
        oc_rating = "low"
    elif oc_pct <= TNAU_THRESHOLDS["organic_carbon_pct"]["high"]:
        oc_rating = "medium"
    else:
        oc_rating = "high"

    return {
        "N": n_rating,
        "P": p_rating,
        "K": k_rating,
        "pH": ph_rating,
        "OC": oc_rating,
    }
