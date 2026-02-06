"""Cesium terrain source adapter using public Cesium ion endpoints."""

import math
import struct
from collections import defaultdict
from urllib.parse import urlencode

import numpy as np
import requests
from scipy.ndimage import gaussian_filter
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from .app_config import (
    get_cesium_ion_token,
    get_cesium_terrain_asset_id,
    get_cesium_timeout_seconds,
    get_terrain_max_samples,
)

CESIUM_ION_ENDPOINT_TEMPLATE = "https://api.cesium.com/v1/assets/{asset_id}/endpoint"


def _to_tile(lat, lon, zoom, scheme="tms"):
    # Cesium quantized-mesh terrain uses a geographic quadtree:
    # X tiles: 2^(z+1) over longitude [-180, 180]
    # Y tiles: 2^z over latitude.
    x_count = 2.0 ** (zoom + 1)
    y_count = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * x_count)
    # Start from north-origin row index and convert for TMS if required.
    y_north = int((90.0 - lat) / 180.0 * y_count)
    if str(scheme).lower() == "tms":
        y = int(y_count - 1 - y_north)
    else:
        y = y_north
    x = max(0, min(int(x_count - 1), x))
    y = max(0, min(int(y_count - 1), y))
    return x, y


def _choose_zoom(north, south, east, west, resolution=120, preview_mode=False):
    span = max(north - south, east - west)
    base = 13
    if span > 1.0:
        base = 9
    elif span > 0.5:
        base = 10
    elif span > 0.1:
        base = 11
    elif span > 0.05:
        base = 12
    elif span > 0.02:
        base = 13
    elif span > 0.01:
        base = 14
    elif span > 0.005:
        base = 15
    else:
        base = 16

    if not preview_mode:
        # Final output should bias toward higher-detail terrain LOD.
        if int(resolution) >= 450:
            base += 3
        elif int(resolution) >= 300:
            base += 1
    return max(9, min(17, base))


def _extract_min_max_height(tile_bytes):
    # Quantized-mesh header includes minHeight/maxHeight as float32 at bytes 24..32.
    if len(tile_bytes) < 32:
        raise RuntimeError("Terrain tile payload too small")
    min_h, max_h = struct.unpack_from("<ff", tile_bytes, 24)
    return float(min_h), float(max_h)


def _zig_zag_decode(value):
    return (value >> 1) ^ (-(value & 1))


def _tile_lon_lat_bounds(z, x, y, scheme="tms"):
    x_count = 2.0 ** (z + 1)
    y_count = 2.0 ** z
    west = (x / x_count) * 360.0 - 180.0
    east = ((x + 1) / x_count) * 360.0 - 180.0
    if str(scheme).lower() == "tms":
        south = -90.0 + (y / y_count) * 180.0
        north = -90.0 + ((y + 1) / y_count) * 180.0
    else:
        north = 90.0 - (y / y_count) * 180.0
        south = 90.0 - ((y + 1) / y_count) * 180.0
    return west, south, east, north


def _decode_quantized_mesh_tile(tile_bytes, z, x, y, scheme="tms"):
    """Decode quantized-mesh vertex quantization streams to sample terrain relief."""
    if len(tile_bytes) < 100:
        raise RuntimeError("Terrain tile payload too small for quantized-mesh")

    min_h, max_h = _extract_min_max_height(tile_bytes)
    offset = 88  # quantized-mesh header length
    vertex_count = struct.unpack_from("<I", tile_bytes, offset)[0]
    offset += 4
    if vertex_count <= 0:
        raise RuntimeError("Terrain tile has zero vertices")

    arr_len = vertex_count * 2
    u_q = np.frombuffer(tile_bytes, dtype=np.uint16, count=vertex_count, offset=offset).astype(np.int64)
    offset += arr_len
    v_q = np.frombuffer(tile_bytes, dtype=np.uint16, count=vertex_count, offset=offset).astype(np.int64)
    offset += arr_len
    h_q = np.frombuffer(tile_bytes, dtype=np.uint16, count=vertex_count, offset=offset).astype(np.int64)

    def decode_stream(vals):
        out = np.empty_like(vals, dtype=np.int64)
        total = 0
        for i, v in enumerate(vals):
            total += _zig_zag_decode(int(v))
            out[i] = total
        return out

    u_dec = decode_stream(u_q).astype(np.float64) / 32767.0
    v_dec = decode_stream(v_q).astype(np.float64) / 32767.0
    h_dec = decode_stream(h_q).astype(np.float64) / 32767.0
    heights = min_h + h_dec * (max_h - min_h)
    points_uv = np.column_stack((u_dec, v_dec))
    linear_interp = None
    nearest_interp = None
    if len(points_uv) >= 3:
        try:
            linear_interp = LinearNDInterpolator(points_uv, heights, fill_value=np.nan)
        except Exception:
            linear_interp = None
    if len(points_uv) > 0:
        try:
            nearest_interp = NearestNDInterpolator(points_uv, heights)
        except Exception:
            nearest_interp = None

    west, south, east, north = _tile_lon_lat_bounds(z, x, y, scheme=scheme)
    return {
        "u": u_dec,
        "v": v_dec,
        "h": heights,
        "linear_interp": linear_interp,
        "nearest_interp": nearest_interp,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
    }


def _sample_tile_height(tile_data, lat, lon):
    west = tile_data["west"]
    east = tile_data["east"]
    south = tile_data["south"]
    north = tile_data["north"]
    if east == west or north == south:
        return 0.0

    # Clamp to tile bounds (sampling may fall slightly outside due to fallback-to-parent tiles)
    lon_c = min(max(lon, west), east)
    lat_c = min(max(lat, south), north)
    u = (lon_c - west) / (east - west)
    v = (lat_c - south) / (north - south)

    linear_interp = tile_data.get("linear_interp")
    if linear_interp is not None:
        h_val = linear_interp(u, v)
        if np.isfinite(h_val):
            return float(h_val)
    nearest_interp = tile_data.get("nearest_interp")
    if nearest_interp is not None:
        h_val = nearest_interp(u, v)
        if np.isfinite(h_val):
            return float(h_val)

    du = tile_data["u"] - u
    dv = tile_data["v"] - v
    idx = int(np.argmin((du * du) + (dv * dv)))
    return float(tile_data["h"][idx])


def _build_url(base_url, path, access_token):
    token_qs = urlencode({"access_token": access_token}) if access_token else ""
    join = "&" if "?" in path else "?"
    return f"{base_url.rstrip('/')}/{path}{join}{token_qs}" if token_qs else f"{base_url.rstrip('/')}/{path}"


def fetch_cesium_terrain_data(north, south, east, west, resolution, preview_mode=False):
    """Fetch terrain heights from Cesium World Terrain endpoints.

    This samples Cesium quantized-mesh tiles by decoding vertex quantization streams
    and querying nearest terrain vertex heights.
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
    tile_version = layer.get("version") or "1.2.0"
    tile_scheme = str(layer.get("scheme") or "tms").lower()
    lats = np.linspace(float(south), float(north), int(resolution))
    lons = np.linspace(float(west), float(east), int(resolution))
    grid = np.zeros((int(resolution), int(resolution)), dtype=np.float64)

    zoom = _choose_zoom(
        float(north),
        float(south),
        float(east),
        float(west),
        resolution=int(resolution),
        preview_mode=bool(preview_mode),
    )
    tile_decode_cache = {}
    tile_miss_cache = set()
    sampled_zoom_counts = defaultdict(int)
    sampled_unique_tiles = set()

    def fetch_decoded_tile_with_fallback(z, x, y):
        zz, xx, yy = int(z), int(x), int(y)
        while zz >= 0:
            key = (zz, xx, yy)
            if key in tile_decode_cache:
                return key, tile_decode_cache[key]
            if key in tile_miss_cache:
                zz -= 1
                xx //= 2
                yy //= 2
                continue

            path = (
                template.replace("{z}", str(zz))
                .replace("{x}", str(xx))
                .replace("{y}", str(yy))
                .replace("{version}", str(tile_version))
            )
            tile_url = _build_url(base_url, path.lstrip("/"), access_token)
            tile_resp = requests.get(tile_url, timeout=timeout)
            if tile_resp.status_code == 200:
                try:
                    decoded = _decode_quantized_mesh_tile(
                        tile_resp.content, zz, xx, yy, scheme=tile_scheme
                    )
                    tile_decode_cache[key] = decoded
                    return key, decoded
                except Exception as exc:
                    print(f"[WARN] Failed to decode Cesium terrain tile {zz}/{xx}/{yy}: {exc}")
            elif tile_resp.status_code not in {404, 204}:
                print(f"[WARN] Cesium terrain tile error {zz}/{xx}/{yy}: status={tile_resp.status_code}")

            tile_miss_cache.add(key)
            zz -= 1
            xx //= 2
            yy //= 2

        return None, None

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            tx, ty = _to_tile(float(lat), float(lon), zoom, scheme=tile_scheme)
            key, tile_data = fetch_decoded_tile_with_fallback(zoom, tx, ty)
            if tile_data is None:
                grid[i, j] = 0.0
            else:
                sampled_zoom_counts[key[0]] += 1
                sampled_unique_tiles.add(key)
                grid[i, j] = _sample_tile_height(tile_data, float(lat), float(lon))

    # Light smoothing removes quantization/tile-edge stitching while preserving landforms.
    if grid.size > 0:
        grid = gaussian_filter(grid, sigma=0.6)

    if sampled_zoom_counts:
        zoom_summary = ", ".join(
            f"z{z}:{count}" for z, count in sorted(sampled_zoom_counts.items(), reverse=True)
        )
        print(
            f"[INFO] Cesium terrain sampled zooms (requested z{zoom}): {zoom_summary}; "
            f"unique_tiles={len(sampled_unique_tiles)} misses={len(tile_miss_cache)}"
        )
    highest_zoom_used = max(sampled_zoom_counts.keys()) if sampled_zoom_counts else None

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
        "diagnostics": {
            "requested_zoom": int(zoom),
            "highest_zoom_used": int(highest_zoom_used) if highest_zoom_used is not None else None,
            "sampled_zoom_counts": {str(k): int(v) for k, v in sampled_zoom_counts.items()},
            "unique_tiles": int(len(sampled_unique_tiles)),
            "tile_misses": int(len(tile_miss_cache)),
        },
    }
