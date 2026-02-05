"""Elevation data fetching utilities with pluggable terrain sources."""

import math
import numpy as np
import requests
from PIL import Image
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from io import BytesIO
import srtm

from .app_config import get_default_terrain_source_mode
from .cesium_terrain_provider import fetch_cesium_terrain_data

# AWS Terrain Tiles (Mapzen/Tilezen format) - free and globally available
AWS_TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def fetch_elevation_data(north, south, east, west, resolution=200, source_mode=None, preview_mode=False):
    """Fetch elevation data for a bounding box.

    Args:
        north: Northern latitude bound
        south: Southern latitude bound
        east: Eastern longitude bound
        west: Western longitude bound
        resolution: Number of points along each axis
        source_mode: One of default|cesium|hybrid
        preview_mode: Reduce Cesium sampling limits in preview mode

    Returns:
        dict: Elevation grid data with coordinates
    """
    mode = (source_mode or get_default_terrain_source_mode() or "hybrid").strip().lower()
    if mode not in {"default", "cesium", "hybrid"}:
        mode = "hybrid"

    if mode == "default":
        return fetch_default_elevation_data(north, south, east, west, resolution)

    if mode == "cesium":
        payload = fetch_cesium_terrain_data(
            north, south, east, west, resolution, preview_mode=preview_mode
        )
        payload.setdefault("source", "cesium")
        return payload

    try:
        payload = fetch_cesium_terrain_data(
            north, south, east, west, resolution, preview_mode=preview_mode
        )
        payload.setdefault("source", "cesium")
        return payload
    except Exception as exc:
        print(f"[WARN] Cesium terrain source failed, falling back to default: {exc}")

    payload = fetch_default_elevation_data(north, south, east, west, resolution)
    payload.setdefault("source", "default")
    payload["fallback_reason"] = "Cesium source failed"
    return payload


def fetch_default_elevation_data(north, south, east, west, resolution=200):
    """Fetch elevation data using existing Terrain RGB + SRTM fallback flow."""
    try:
        elevations, lats, lons = fetch_terrain_rgb_tiles(north, south, east, west, resolution)
        if elevations is not None:
            elevations = gaussian_filter(elevations, sigma=0.5)
            return _build_payload(elevations, lats, lons, north, south, east, west, resolution, source="default")
    except Exception as e:
        print(f"Terrain RGB tiles failed, falling back to SRTM: {e}")

    payload = fetch_srtm_elevation(north, south, east, west, resolution)
    payload["source"] = "default"
    return payload


def _build_payload(elevations, lats, lons, north, south, east, west, resolution, source):
    return {
        "grid": elevations.tolist(),
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "bounds": {
            "north": north,
            "south": south,
            "east": east,
            "west": west,
        },
        "resolution": resolution,
        "min_elevation": float(np.min(elevations)),
        "max_elevation": float(np.max(elevations)),
        "source": source,
    }


def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates at a given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lat_lon(x, y, zoom):
    """Convert tile coordinates to lat/lon (northwest corner)."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def decode_terrarium(r, g, b):
    """Decode Terrarium format RGB to elevation in meters."""
    return (r * 256.0 + g + b / 256.0) - 32768.0


def fetch_terrain_rgb_tiles(north, south, east, west, resolution):
    """Fetch elevation data from AWS Terrain RGB tiles."""
    lat_span = north - south
    lon_span = east - west

    max_span = max(lat_span, lon_span)
    if max_span > 1.0:
        zoom = 10
    elif max_span > 0.5:
        zoom = 11
    elif max_span > 0.1:
        zoom = 12
    elif max_span > 0.05:
        zoom = 13
    else:
        zoom = 14

    print(f"Using zoom level {zoom} for terrain tiles")

    x_min, y_max = lat_lon_to_tile(south, west, zoom)
    x_max, y_min = lat_lon_to_tile(north, east, zoom)

    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    num_tiles_x = x_max - x_min + 1
    num_tiles_y = y_max - y_min + 1

    if num_tiles_x * num_tiles_y > 25:
        print(f"Too many tiles ({num_tiles_x * num_tiles_y}), reducing zoom")
        return None, None, None

    print(f"Fetching {num_tiles_x}x{num_tiles_y} tiles")

    tile_size = 256
    stitched_width = num_tiles_x * tile_size
    stitched_height = num_tiles_y * tile_size

    stitched_elevations = np.zeros((stitched_height, stitched_width))

    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            url = AWS_TERRAIN_URL.format(z=zoom, x=tx, y=ty)

            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img_array = np.array(img)

                    r = img_array[:, :, 0].astype(np.float64)
                    g = img_array[:, :, 1].astype(np.float64)
                    b = img_array[:, :, 2].astype(np.float64)

                    tile_elevations = decode_terrarium(r, g, b)

                    px = (tx - x_min) * tile_size
                    py = (ty - y_min) * tile_size
                    stitched_elevations[py:py + tile_size, px:px + tile_size] = tile_elevations
                else:
                    print(f"Failed to fetch tile {tx},{ty}: {response.status_code}")
            except Exception as e:
                print(f"Error fetching tile {tx},{ty}: {e}")

    tile_north, tile_west = tile_to_lat_lon(x_min, y_min, zoom)
    tile_south, tile_east = tile_to_lat_lon(x_max + 1, y_max + 1, zoom)

    tile_lats = np.linspace(tile_south, tile_north, stitched_height)
    tile_lons = np.linspace(tile_west, tile_east, stitched_width)

    stitched_elevations = np.flipud(stitched_elevations)

    lat_indices = np.where((tile_lats >= south) & (tile_lats <= north))[0]
    lon_indices = np.where((tile_lons >= west) & (tile_lons <= east))[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        print("Bounding box doesn't overlap with tiles")
        return None, None, None

    cropped_elevations = stitched_elevations[
        lat_indices[0]:lat_indices[-1] + 1,
        lon_indices[0]:lon_indices[-1] + 1,
    ]
    cropped_lats = tile_lats[lat_indices[0]:lat_indices[-1] + 1]
    cropped_lons = tile_lons[lon_indices[0]:lon_indices[-1] + 1]

    if cropped_elevations.shape[0] < 4 or cropped_elevations.shape[1] < 4:
        print("Cropped region too small for cubic interpolation")
        return None, None, None

    try:
        interp_func = interpolate.RectBivariateSpline(
            cropped_lats, cropped_lons, cropped_elevations, kx=3, ky=3
        )

        target_lats = np.linspace(south, north, resolution)
        target_lons = np.linspace(west, east, resolution)

        elevations = interp_func(target_lats, target_lons)
        return elevations, target_lats, target_lons
    except Exception as e:
        print(f"Interpolation error: {e}")
        return None, None, None


def fetch_srtm_elevation(north, south, east, west, resolution):
    """Fetch elevation data using SRTM fallback."""
    try:
        elevation_data = srtm.get_data()

        sample_res = min(resolution * 2, 400)
        lats = np.linspace(south, north, sample_res)
        lons = np.linspace(west, east, sample_res)

        raw_elevations = np.zeros((sample_res, sample_res))

        for i in range(sample_res):
            for j in range(sample_res):
                lat = lats[i]
                lon = lons[j]
                elev = elevation_data.get_elevation(lat, lon)
                raw_elevations[i, j] = elev if elev is not None else 0

        smoothed = gaussian_filter(raw_elevations, sigma=1.5)

        if sample_res != resolution:
            interp_func = interpolate.RectBivariateSpline(
                lats, lons, smoothed, kx=3, ky=3
            )
            target_lats = np.linspace(south, north, resolution)
            target_lons = np.linspace(west, east, resolution)
            elevations = interp_func(target_lats, target_lons)
        else:
            elevations = smoothed
            target_lats = lats
            target_lons = lons

        return {
            "grid": elevations.tolist(),
            "lats": target_lats.tolist(),
            "lons": target_lons.tolist(),
            "bounds": {
                "north": north,
                "south": south,
                "east": east,
                "west": west,
            },
            "resolution": resolution,
            "min_elevation": float(np.min(elevations)),
            "max_elevation": float(np.max(elevations)),
        }

    except Exception as e:
        raise Exception(f"Error fetching SRTM elevation data: {str(e)}")
