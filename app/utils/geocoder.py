"""Address geocoding utilities.

Resolution order:
1. Extract coordinates from Google Maps URLs.
2. Query Cesium ion geocoding endpoints.
"""

import re
import requests

from .app_config import get_cesium_ion_token, get_cesium_timeout_seconds

_GOOGLE_MAPS_HOST_PATTERNS = ("google.com/maps", "goo.gl/maps")


def _extract_coords_from_google_maps_url(url):
    """Extract lat/lon from common Google Maps URL formats."""
    data_match = re.search(r"!3d(-?\d+\.?\d*)!4d(-?\d+\.?\d*)", url)
    if data_match:
        return float(data_match.group(1)), float(data_match.group(2))

    at_match = re.search(r"@(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
    if at_match:
        return float(at_match.group(1)), float(at_match.group(2))

    q_match = re.search(r"[?&]q=(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
    if q_match:
        return float(q_match.group(1)), float(q_match.group(2))

    return None


def _is_google_maps_url(value):
    value = (value or "").lower()
    return any(host in value for host in _GOOGLE_MAPS_HOST_PATTERNS)


def _parse_cesium_response(data, original_query):
    """Parse Cesium geocoder response shapes into a standard dict."""
    if not isinstance(data, dict):
        return None

    features = data.get("features")
    if isinstance(features, list) and features:
        first = features[0]
        props = first.get("properties", {}) if isinstance(first, dict) else {}
        geometry = first.get("geometry", {}) if isinstance(first, dict) else {}
        coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
            label = (
                first.get("place_name")
                or first.get("name")
                or props.get("label")
                or props.get("name")
                or original_query
            )
            return {
                "address": label,
                "lat": lat,
                "lon": lon,
                "raw": data,
            }

        bbox = first.get("bbox") if isinstance(first, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            west, south, east, north = map(float, bbox[:4])
            lon = (west + east) / 2.0
            lat = (south + north) / 2.0
            label = (
                first.get("place_name")
                or first.get("name")
                or props.get("label")
                or props.get("name")
                or original_query
            )
            return {
                "address": label,
                "lat": lat,
                "lon": lon,
                "raw": data,
            }

    results = data.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        destination = first.get("destination") if isinstance(first, dict) else None
        if isinstance(destination, (list, tuple)) and len(destination) >= 2:
            lon, lat = float(destination[0]), float(destination[1])
            label = first.get("displayName") or first.get("name") or original_query
            return {
                "address": label,
                "lat": lat,
                "lon": lon,
                "raw": data,
            }

    return None


def _geocode_with_cesium(address, timeout):
    token = get_cesium_ion_token()
    if not token:
        raise Exception("CESIUM_ION_TOKEN is not configured")

    if timeout is None:
        timeout = get_cesium_timeout_seconds()

    endpoint_candidates = [
        ("https://api.cesium.com/v1/geocode/search", {"text": address, "access_token": token}),
        ("https://api.cesium.com/v1/geocode/search", {"q": address, "access_token": token}),
    ]

    last_error = None
    for url, params in endpoint_candidates:
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in {404, 405}:
                last_error = "Endpoint not available"
                continue
            response.raise_for_status()
            data = response.json()
            parsed = _parse_cesium_response(data, address)
            if parsed:
                return parsed
            last_error = "No geocoding results"
        except Exception as exc:
            last_error = str(exc)

    raise Exception(f"Cesium geocoding failed: {last_error}")


def geocode_address(address, timeout=10):
    """Geocode an address to coordinates using Google URL parsing and Cesium ion."""
    value = (address or "").strip()
    if not value:
        raise Exception("No address provided")

    if _is_google_maps_url(value):
        coords = _extract_coords_from_google_maps_url(value)
        if coords:
            lat, lon = coords
            return {
                "address": f"Coordinates from Google Maps URL ({lat:.6f}, {lon:.6f})",
                "lat": lat,
                "lon": lon,
                "raw": {"source": "google_maps_url"},
            }
        raise Exception("Could not extract coordinates from Google Maps URL")

    return _geocode_with_cesium(value, timeout)
