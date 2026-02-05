"""Cesium terrain source adapter using public Cesium ion endpoints."""

import math
import struct
from urllib.parse import urlencode

import numpy as np
import requests

from .app_config import (
    get_cesium_ion_token,
    get_cesium_terrain_asset_id,
    get_cesium_timeout_seconds,
    get_terrain_max_samples,
)

CESIUM_ION_ENDPOINT_TEMPLATE = "https://api.cesium.com/v1/assets/{asset_id}/endpoint"


def _to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _choose_zoom(north, south, east, west):
    span = max(north - south, east - west)
    if span > 1.0:
        return 9
    if span > 0.5:
        return 10
    if span > 0.1:
        return 11
    if span > 0.05:
        return 12
    return 13


def _extract_min_max_height(tile_bytes):
    # Quantized-mesh header includes minHeight/maxHeight as doubles at bytes 24..40.
    if len(tile_bytes) < 40:
        raise RuntimeError("Terrain tile payload too small")
    min_h, max_h = struct.unpack_from("<dd", tile_bytes, 24)
    return float(min_h), float(max_h)


def _build_url(base_url, path, access_token):
    token_qs = urlencode({"access_token": access_token}) if access_token else ""
    join = "&" if "?" in path else "?"
    return f"{base_url.rstrip('/')}/{path}{join}{token_qs}" if token_qs else f"{base_url.rstrip('/')}/{path}"


def fetch_cesium_terrain_data(north, south, east, west, resolution, preview_mode=False):
    """Fetch terrain heights from Cesium World Terrain endpoints.

    This samples Cesium quantized-mesh tiles and uses tile min/max height averages
    to populate the elevation grid.
    """
    token = get_cesium_ion_token()
    if not token:
        raise RuntimeError("CESIUM_ION_TOKEN is not configured")

    max_samples = get_terrain_max_samples(preview_mode)
    requested_samples = int(resolution) * int(resolution)
    if requested_samples > max_samples:
        raise RuntimeError(
            f"Requested terrain resolution ({requested_samples} samples) exceeds Cesium limit ({max_samples})"
        )

    timeout = get_cesium_timeout_seconds()
    asset_id = get_cesium_terrain_asset_id()
    endpoint_url = CESIUM_ION_ENDPOINT_TEMPLATE.format(asset_id=asset_id)
    headers = {"Authorization": f"Bearer {token}"}
    endpoint_resp = requests.get(endpoint_url, headers=headers, timeout=timeout)
    endpoint_resp.raise_for_status()
    endpoint = endpoint_resp.json()

    base_url = (endpoint.get("url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("Cesium endpoint response missing terrain URL")
    access_token = endpoint.get("accessToken", token)

    # Resolve layer.json to discover tile path template.
    layer_url = _build_url(base_url, "layer.json", access_token)
    layer_resp = requests.get(layer_url, timeout=timeout)
    layer_resp.raise_for_status()
    layer = layer_resp.json()
    templates = layer.get("tiles") or []
    if not templates:
        raise RuntimeError("Cesium terrain layer.json missing tiles template")
    template = templates[0]

    lats = np.linspace(float(south), float(north), int(resolution))
    lons = np.linspace(float(west), float(east), int(resolution))
    grid = np.zeros((int(resolution), int(resolution)), dtype=np.float64)

    zoom = _choose_zoom(float(north), float(south), float(east), float(west))
    tile_height_cache = {}

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            tx, ty = _to_tile(float(lat), float(lon), zoom)
            key = (zoom, tx, ty)
            if key not in tile_height_cache:
                path = (
                    template.replace("{z}", str(zoom))
                    .replace("{x}", str(tx))
                    .replace("{y}", str(ty))
                    .replace("{version}", "1.2.0")
                )
                tile_url = _build_url(base_url, path.lstrip("/"), access_token)
                tile_resp = requests.get(tile_url, timeout=timeout)
                if tile_resp.status_code != 200:
                    raise RuntimeError(f"Failed to fetch terrain tile {zoom}/{tx}/{ty}")
                min_h, max_h = _extract_min_max_height(tile_resp.content)
                tile_height_cache[key] = (min_h + max_h) / 2.0
            grid[i, j] = tile_height_cache[key]

    return {
        "grid": grid.tolist(),
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "bounds": {
            "north": float(north),
            "south": float(south),
            "east": float(east),
            "west": float(west),
        },
        "resolution": int(resolution),
        "min_elevation": float(np.min(grid)),
        "max_elevation": float(np.max(grid)),
        "source": "cesium",
    }
