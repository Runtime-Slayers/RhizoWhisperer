"""
Real-World Agronomic Economic ROI & Farmer Savings Calculator
================================================================

A NOVEL module that calculates precise financial savings ($/ha) and fertilizer
use reduction (% NPK saved) when executing RHIZO-NET precision recommendations.

REAL-WORLD UTILITY:
Provides actionable financial decision support for smallholder and commercial farmers.
"""

from typing import Dict, Any


# Standard Commercial Fertilizer Costs ($/kg active nutrient)
FERTILIZER_PRICES = {
    "N": 1.20,     # Urea equivalent
    "P2O5": 1.45,  # DAP equivalent
    "K2O": 1.10,   # MOP equivalent
}


def calculate_agronomic_roi(
    crop_name: str,
    blanket_npk: Dict[str, float],
    prescribed_npk: Dict[str, float],
    yield_gain_pct: float = 12.5,
    crop_price_per_ton_usd: float = 280.0,
    base_yield_tons_ha: float = 4.5,
) -> Dict[str, Any]:
    """
    Calculate farmer net savings ($/ha), ROI percentage, and fertilizer saved.
    """
    blanket_cost = sum(blanket_npk[k] * FERTILIZER_PRICES[k] for k in ["N", "P2O5", "K2O"])
    prescribed_cost = sum(prescribed_npk[k] * FERTILIZER_PRICES[k] for k in ["N", "P2O5", "K2O"])

    fertilizer_cost_difference = blanket_cost - prescribed_cost

    # Additional yield revenue from precision timing & deficiency remediation
    yield_increase_tons = base_yield_tons_ha * (yield_gain_pct / 100.0)
    additional_revenue_usd = yield_increase_tons * crop_price_per_ton_usd

    # Total Net Financial Benefit ($/ha)
    total_net_benefit_usd_ha = additional_revenue_usd + fertilizer_cost_difference

    # Percentage fertilizer change
    blanket_tot_kg = sum(blanket_npk.values())
    prescribed_tot_kg = sum(prescribed_npk.values())
    pct_fertilizer_change = ((prescribed_tot_kg - blanket_tot_kg) / (blanket_tot_kg + 1e-5)) * 100.0

    return {
        "crop_name": crop_name,
        "blanket_cost_usd_ha": float(blanket_cost),
        "prescribed_cost_usd_ha": float(prescribed_cost),
        "fertilizer_cost_difference_usd": float(fertilizer_cost_difference),
        "additional_revenue_usd_ha": float(additional_revenue_usd),
        "total_net_benefit_usd_ha": float(total_net_benefit_usd_ha),
        "pct_fertilizer_change": float(pct_fertilizer_change),
        "roi_percentage": float((total_net_benefit_usd_ha / (prescribed_cost + 1e-5)) * 100.0),
    }
