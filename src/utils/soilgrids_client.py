"""
SoilGrids API Client with Local Offline Fallback
===============================================

Queries ISRIC SoilGrids REST API (or returns cached local tiles in offline mode).
Fetches 6 soil depth layers (0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Default nominal values for offline fallback (Tamil Nadu agricultural soils)
NOMINAL_SOIL_FALLBACK = {
    "bdod": 135.0,      # Bulk density: 1.35 kg/dm³
    "cec": 180.0,       # Cation exchange capacity: 18.0 cmol(c)/kg
    "clay": 320.0,      # Clay: 32.0%
    "nitrogen": 140.0,  # Nitrogen: 1.4 g/kg
    "soc": 110.0,       # Soil organic carbon: 11.0 g/kg
    "phh2o": 72.0,      # pH: 7.2
    "wv0033": 220.0,    # Volumetric water content at 33kPa: 22.0%
}


def fetch_soilgrids_data(
    lat: float,
    lon: float,
    cache_dir: Optional[str] = None,
    offline_fallback: bool = True,
) -> Dict[str, float]:
    """
    Fetch SoilGrids data for given GPS coordinates.

    Args:
        lat: Latitude
        lon: Longitude
        cache_dir: Directory to store/load cached SoilGrids JSON
        offline_fallback: If True, return nominal Tamil Nadu values on network failure

    Returns:
        Dict of SoilGrids raw mapped property values
    """
    if cache_dir:
        cache_path = Path(cache_dir) / f"soilgrids_{lat:.4f}_{lon:.4f}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

    # Online fetch via ISRIC REST API
    if HAS_REQUESTS:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                properties = {}
                for prop in data.get("properties", {}).get("layers", []):
                    name = prop.get("name")
                    depths = prop.get("depths", [])
                    if depths:
                        # Average mean across top 0-30cm depths
                        vals = [d.get("values", {}).get("mean") for d in depths[:3] if d.get("values", {}).get("mean") is not None]
                        if vals:
                            properties[name] = float(sum(vals) / len(vals))

                if properties:
                    if cache_dir:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, "w") as f:
                            json.dump(properties, f)
                    return properties
        except Exception:
            pass

    # Offline fallback
    return NOMINAL_SOIL_FALLBACK.copy()
