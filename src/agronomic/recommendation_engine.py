"""
TNAU-Based Agronomic Recommendation Engine
===========================================

Deterministic recommendation engine that combines:
1. Model predicted deficiency class (Optimal, N-Def, P-Def, K-Def, Micro-Def)
2. TNAU Soil Ratings (Low, Medium, High)
3. Crop blanket NPK recommendations
4. Nutrient lockout detection (high pH, cluster roots, calcareous soil)
5. Micronutrient intervention protocols

Produces precise, distinct, actionable fertilizer prescriptions tailored to each crop.
"""

from typing import Dict, Any, List
from .soil_rating import rate_soil_fertility
from .crop_profiles import get_crop_profile


class TNAUAgronomicEngine:
    """
    TNAU Agronomic Recommendation Engine.

    Rule-based system adhering to Tamil Nadu Agricultural University standards.
    Adjusts blanket fertilizer rates based on model predictions and crop-specific soil chemistry.
    """

    def generate_recommendation(
        self,
        deficiency_class: str,
        crop_name: str,
        soil_n_kg_ha: float,
        soil_p_kg_ha: float,
        soil_k_kg_ha: float,
        soil_ph: float,
        soil_oc_pct: float = 0.5,
        root_clustering_detected: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate complete fertilizer recommendation report tailored for the specific crop.
        """
        # 1. Get soil ratings
        soil_ratings = rate_soil_fertility(soil_n_kg_ha, soil_p_kg_ha, soil_k_kg_ha, soil_ph, soil_oc_pct)

        # 2. Get crop profile
        crop = get_crop_profile(crop_name)
        blanket = crop["blanket_npk_kg_ha"].copy()

        adjusted_npk = blanket.copy()
        special_interventions = []
        notes = []

        # 3. Differential adjustments based on soil rating + model deficiency
        # Nitrogen adjustment
        if deficiency_class == "nitrogen_deficiency":
            if soil_ratings["N"] == "low":
                adjusted_npk["N"] *= 1.5
                notes.append("Soil N is Low + Model detected N deficiency -> Increased N recommendation by 50%.")
            elif soil_ratings["N"] == "medium":
                adjusted_npk["N"] *= 1.25
                notes.append("Soil N is Medium + Model detected N deficiency -> Increased N recommendation by 25%.")
            elif soil_ratings["N"] == "high":
                special_interventions.append({
                    "type": "lockout_warning",
                    "element": "Nitrogen",
                    "action": "Foliar 1-2% Urea Spray",
                    "reason": "High soil N but plant displays N deficiency. Suspect root uptake lockout or leaching."
                })
        else:
            if soil_ratings["N"] == "medium":
                adjusted_npk["N"] *= 0.85
            elif soil_ratings["N"] == "high":
                adjusted_npk["N"] *= 0.7

        # Phosphorus adjustment
        if deficiency_class == "phosphorus_deficiency":
            if soil_ratings["P"] == "low":
                adjusted_npk["P2O5"] *= 1.5
                notes.append("Soil P is Low -> Increased P2O5 by 50% with localized root-zone placement.")
            elif soil_ratings["P"] == "medium":
                adjusted_npk["P2O5"] *= 1.2
                notes.append("Soil P is Medium -> Increased P2O5 by 20%.")
            elif soil_ratings["P"] == "high":
                special_interventions.append({
                    "type": "lockout_warning",
                    "element": "Phosphorus",
                    "action": "Localized placement / DAP foliar spray",
                    "reason": "High soil P but plant displays P deficiency. Fixation likely occurring in alkaline/calcareous soil."
                })

        # Potassium adjustment
        if deficiency_class == "potassium_deficiency":
            if soil_ratings["K"] == "low":
                adjusted_npk["K2O"] *= 1.5
                notes.append("Soil K is Low -> Increased K2O by 50%.")
            elif soil_ratings["K"] == "medium":
                adjusted_npk["K2O"] *= 1.25
                notes.append("Soil K is Medium -> Increased K2O by 25%.")

        # 4. Crop-Specific & Soil Interventions
        crop_clean = crop_name.lower().replace(" ", "_")

        if crop_clean in ("groundnut", "groundnut_rainfed"):
            if soil_ph >= 8.2:
                special_interventions.append({
                    "type": "calcareous_gypsum_suppression",
                    "title": "Gypsum Application Suppression",
                    "action": "Suppress standard 400 kg/ha Gypsum application at 45 DAS",
                    "reason": f"Alkaline calcareous soil (pH {soil_ph:.1f}). Gypsum would cause severe pH elevation and calcium toxicity."
                })
            else:
                special_interventions.append({
                    "type": "pod_development",
                    "title": "Gypsum Requirement",
                    "action": "Apply 400 kg/ha Gypsum at 45 DAS (peg formation stage)",
                    "reason": "Essential for calcium availability during pod development."
                })

        elif crop_clean in ("sorghum", "sorghum_irrigated"):
            special_interventions.append({
                "type": "split_n_schedule",
                "title": "3-Stage Split Nitrogen Management",
                "action": "Split N @ 50:25:25% (0, 15, and 30 DAS)",
                "reason": "Prevents volatilization and aligns with peak sorghum vegetative intake."
            })

        elif crop_clean in ("tomato", "tomato_hybrid"):
            special_interventions.append({
                "type": "drip_fertigation",
                "title": "Water-Soluble Drip Fertigation Protocol",
                "action": "Apply 19:19:19 + Urea daily for 1 hour/day via drip system",
                "reason": "Optimizes high nutrient uptake of hybrid tomato during fruit set."
            })

        elif crop_clean in ("turmeric",):
            special_interventions.append({
                "type": "organic_rhizosphere",
                "title": "Organic Basal FYM & Azospirillum Integration",
                "action": "Apply 25 t/ha FYM + Azospirillum @ 2 kg/ha basally at planting",
                "reason": "Enhances rhizome elongation and organic soil structure."
            })

        elif crop_clean in ("african_marigold", "marigold"):
            special_interventions.append({
                "type": "floral_top_dressing",
                "title": "Floral Top Dressing Schedule",
                "action": "Apply 45 kg N/ha top dressing at 45 DAS (flower-bud initiation)",
                "reason": "Boosts flower density and petal pigment concentration."
            })

        # Micronutrient Lockout Intervention (if pH >= 8.5)
        if soil_ph >= 8.5 and deficiency_class == "micronutrient_deficiency":
            special_interventions.append({
                "type": "micronutrient_protocol",
                "title": "Zinc-Iron Lockout Remediation (TNAU Protocol)",
                "action": "Foliar spray of 0.5% ZnSO4 + 1.0% FeSO4 + 0.1% Citric Acid at 60, 90, 120 DAP",
                "reason": f"Alkaline soil (pH {soil_ph:.1f}) causes soil Zn/Fe fixation."
            })

        return {
            "crop": crop["name"],
            "deficiency_state": deficiency_class,
            "soil_ratings": soil_ratings,
            "blanket_recommendation_npk": blanket,
            "prescribed_recommendation_npk": {k: round(v, 1) for k, v in adjusted_npk.items()},
            "special_interventions": special_interventions,
            "notes": notes,
            "tnau_crop_notes": crop.get("notes", ""),
        }
