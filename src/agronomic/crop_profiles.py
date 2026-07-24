"""
TNAU Crop Profiles & Blanket Recommendations
===========================================

Defines blanket NPK recommendations for key crops according to TNAU guidelines:
- Turmeric (Curcuma longa)
- Irrigated Sorghum (Sorghum bicolor)
- Hybrid Tomato (Solanum lycopersicum)
- Rainfed Groundnut (Arachis hypogaea)
- African Marigold (Tagetes erecta)
"""

from typing import Dict, Any


CROP_PROFILES = {
    "turmeric": {
        "name": "Turmeric",
        "scientific_name": "Curcuma longa",
        "blanket_npk_kg_ha": {"N": 25, "P2O5": 60, "K2O": 18},
        "fym_t_ha": 25,
        "biofertilizer": "Azospirillum @ 2 kg/ha basally",
        "notes": "Apply 25 t/ha FYM basally; requires early biofertilizer integration.",
    },
    "sorghum_irrigated": {
        "name": "Sorghum (Irrigated)",
        "scientific_name": "Sorghum bicolor",
        "blanket_npk_kg_ha": {"N": 90, "P2O5": 45, "K2O": 45},
        "split_n": "50:25:25% at 0, 15, and 30 Days After Sowing (DAS)",
        "notes": "Split N application @ 50:25:25% at 0, 15, and 30 DAS.",
    },
    "tomato_hybrid": {
        "name": "Tomato (Hybrid)",
        "scientific_name": "Solanum lycopersicum",
        "blanket_npk_kg_ha": {"N": 200, "P2O5": 250, "K2O": 250},
        "fertigation": "Daily water-soluble fertilizers (19:19:19, Urea) for 1 hour/day via drip",
        "notes": "Split via daily water soluble fertilizers.",
    },
    "groundnut_rainfed": {
        "name": "Groundnut (Rainfed)",
        "scientific_name": "Arachis hypogaea",
        "blanket_npk_kg_ha": {"N": 25, "P2O5": 50, "K2O": 75},
        "gypsum_kg_ha": 400,
        "gypsum_das": 45,
        "notes": "Requires 400 kg/ha Gypsum at 45 DAS to encourage pod formation.",
    },
    "african_marigold": {
        "name": "African Marigold",
        "scientific_name": "Tagetes erecta",
        "blanket_npk_kg_ha": {"N": 45, "P2O5": 90, "K2O": 75},
        "top_dress_n_kg_ha": 45,
        "top_dress_das": 45,
        "notes": "Apply basal NPK, followed by 45 kg N/ha top dressing at 45 DAS.",
    },
}


def get_crop_profile(crop_key: str) -> Dict[str, Any]:
    """Retrieve TNAU crop profile."""
    crop_clean = crop_key.lower().replace(" ", "_")
    return CROP_PROFILES.get(crop_clean, CROP_PROFILES["sorghum_irrigated"])
