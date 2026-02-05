"""Cesium building mesh source adapter using public Cesium ion endpoints."""

import math
from urllib.parse import urlencode

import numpy as np
import requests

from .app_config import (
    get_cesium_ion_token,
    get_cesium_osm_asset_id,
    get_cesium_timeout_seconds,
)

CESIUM_ION_ENDPOINT_TEMPLATE = "https://api.cesium.com/v1/assets/{asset_id}/endpoint"


def _project_lon_lat_to_model(lon, lat, bounds, scale_factor, avg_lat):
    x = (lon - bounds["west"]) * scale_factor * np.cos(np.radians(avg_lat))
    z = (bounds["north"] - lat) * scale_factor
    return x, z


def _build_url(base_url, path, access_token):
    token_qs = urlencode({"access_token": access_token}) if access_token else ""
    join = "&" if "?" in path else "?"
    return f"{base_url.rstrip('/')}/{path}{join}{token_qs}" if token_qs else f"{base_url.rstrip('/')}/{path}"


def _region_intersects_bounds(region, bounds):
    # region = [west, south, east, north, minHeight, maxHeight] in radians/meters
    west = math.degrees(region[0])
    south = math.degrees(region[1])
    east = math.degrees(region[2])
    north = math.degrees(region[3])
    return not (
        east < bounds["west"]
        or west > bounds["east"]
        or north < bounds["south"]
        or south > bounds["north"]
    )


def _bbox_from_region(region):
    return {
        "west": math.degrees(region[0]),
        "south": math.degrees(region[1]),
        "east": math.degrees(region[2]),
        "north": math.degrees(region[3]),
        "min_h": float(region[4]),
        "max_h": float(region[5]),
    }


def _make_box_mesh_for_region(region_bbox, bounds, scale_factor, vertical_scale, elev_params):
    corners_lon_lat = [
        (region_bbox["west"], region_bbox["south"]),
        (region_bbox["east"], region_bbox["south"]),
        (region_bbox["east"], region_bbox["north"]),
        (region_bbox["west"], region_bbox["north"]),
    ]

    bottom_y = (
        ((region_bbox["min_h"] - elev_params["min_elev"]) / elev_params["elev_range"])
        * 20.0
        * elev_params["size_scale"]
        * vertical_scale
    )
    top_y = (
        ((region_bbox["max_h"] - elev_params["min_elev"]) / elev_params["elev_range"])
        * 20.0
        * elev_params["size_scale"]
        * vertical_scale
    )
    if top_y <= bottom_y:
        top_y = bottom_y + (0.6 * elev_params["size_scale"])

    bottom = []
    top = []
    for lon, lat in corners_lon_lat:
        x, z = _project_lon_lat_to_model(lon, lat, bounds, scale_factor, elev_params["avg_lat"])
        bottom.append([x, bottom_y, z])
        top.append([x, top_y, z])

    vertices = bottom + top
    faces = [
        [0, 1, 2], [0, 2, 3],
        [4, 7, 6], [4, 6, 5],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ]
    return vertices, faces


def _simplify_faces(faces, ratio):
    if ratio >= 0.999:
        return faces
    keep = max(1, int(len(faces) * max(0.05, ratio)))
    return faces[:keep]


def _repair_mesh(vertices, faces):
    if len(vertices) == 0 or len(faces) == 0:
        return [], []

    valid_faces = []
    for f in faces:
        if len(f) != 3:
            continue
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a == b or b == c or c == a:
            continue
        if min(a, b, c) < 0 or max(a, b, c) >= len(vertices):
            continue
        valid_faces.append([a, b, c])

    if not valid_faces:
        return [], []

    used_indices = sorted({idx for face in valid_faces for idx in face})
    remap = {old: new for new, old in enumerate(used_indices)}
    compact_vertices = [vertices[idx] for idx in used_indices]
    compact_faces = [[remap[a], remap[b], remap[c]] for a, b, c in valid_faces]
    return compact_vertices, compact_faces


def fetch_cesium_building_meshes(bounds, scale_factor, vertical_scale, elev_params, options, shape_clipper=None):
    """Fetch prebuilt building meshes via a Cesium-backed API.

    Uses Cesium OSM Buildings asset endpoint and tileset metadata.
    Creates printable proxy meshes from tile region bounding volumes.
    """
    token = get_cesium_ion_token()
    if not token:
        raise RuntimeError("CESIUM_ION_TOKEN is not configured")

    timeout = get_cesium_timeout_seconds()
    asset_id = get_cesium_osm_asset_id()
    endpoint_url = CESIUM_ION_ENDPOINT_TEMPLATE.format(asset_id=asset_id)
    headers = {"Authorization": f"Bearer {token}"}

    endpoint_resp = requests.get(endpoint_url, headers=headers, timeout=timeout)
    endpoint_resp.raise_for_status()
    endpoint = endpoint_resp.json()
    base_url = (endpoint.get("url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("Cesium endpoint response missing tileset URL")
    access_token = endpoint.get("accessToken", token)

    tileset_url = _build_url(base_url, "tileset.json", access_token)
    tileset_resp = requests.get(tileset_url, timeout=timeout)
    tileset_resp.raise_for_status()
    tileset = tileset_resp.json()
    root = tileset.get("root") or {}

    simplify_enabled = bool(options.get("building_mesh_simplify", True))
    ratio = options.get(
        "building_mesh_target_ratio_preview" if options.get("preview_mode", False) else "building_mesh_target_ratio_final",
        0.2 if options.get("preview_mode", False) else 0.4,
    )
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = 0.2 if options.get("preview_mode", False) else 0.4

    max_meshes = 200 if not options.get("preview_mode", False) else 80
    stack = [root]
    regions = []
    while stack and len(regions) < max_meshes:
        node = stack.pop()
        bv = (node.get("boundingVolume") or {})
        region = bv.get("region")
        if isinstance(region, list) and len(region) >= 6 and _region_intersects_bounds(region, bounds):
            regions.append(_bbox_from_region(region))
        for child in node.get("children", []):
            if isinstance(child, dict):
                stack.append(child)

    if not regions:
        return []

    meshes = []
    for idx, region_bbox in enumerate(regions):
        vertices, faces = _make_box_mesh_for_region(
            region_bbox, bounds, scale_factor, vertical_scale, elev_params
        )

        if simplify_enabled:
            faces = _simplify_faces(faces, ratio)

        if shape_clipper is not None:
            clipped_faces = []
            for f in faces:
                v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
                inside = shape_clipper.is_inside(
                    np.array([v0[0], v1[0], v2[0]]), np.array([v0[2], v1[2], v2[2]])
                )
                if bool(np.all(inside)):
                    clipped_faces.append(f)
            faces = clipped_faces

        vertices, faces = _repair_mesh(vertices, faces)
        if not vertices or not faces:
            continue

        meshes.append(
            {
                "type": "building",
                "id": f"tile_{idx}",
                "name": f"Cesium Region Building {idx + 1}",
                "building_type": "yes",
                "vertices": vertices,
                "faces": faces,
                "source": "tiles",
                "custom_color": None,
            }
        )

    return meshes
