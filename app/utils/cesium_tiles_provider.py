"""Cesium building mesh source adapter using public Cesium ion endpoints."""

import math
import json
import struct
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

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


def _append_access_token(url, access_token):
    if not access_token:
        return url
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["access_token"] = access_token
    new_query = urlencode(query)
    return urlunparse(parsed._replace(query=new_query))


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


def _simplify_faces(faces, ratio, min_keep=12):
    if ratio >= 0.999:
        return faces
    keep = max(min_keep, int(len(faces) * max(0.05, ratio)))
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


def _mesh_spans(vertices):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return {
        "x": max(xs) - min(xs),
        "y": max(ys) - min(ys),
        "z": max(zs) - min(zs),
        "min_y": min(ys),
        "max_y": max(ys),
    }


def _inflate_mesh_height(vertices, min_height):
    """Stretch low-profile meshes so buildings remain visible/printable."""
    spans = _mesh_spans(vertices)
    current = spans["y"]
    if current >= min_height:
        return vertices
    base = spans["min_y"]
    if current <= 1e-9:
        return [[v[0], base + min_height, v[2]] for v in vertices]
    scale = min_height / current
    inflated = []
    for x, y, z in vertices:
        inflated.append([x, base + (y - base) * scale, z])
    return inflated


def _clamp_mesh_footprint(vertices, max_span):
    spans = _mesh_spans(vertices)
    current = max(spans["x"], spans["z"])
    if current <= max_span or current <= 1e-9:
        return vertices
    cx = float(np.mean([v[0] for v in vertices]))
    cz = float(np.mean([v[2] for v in vertices]))
    scale = max_span / current
    clamped = []
    for x, y, z in vertices:
        clamped.append([cx + ((x - cx) * scale), y, cz + ((z - cz) * scale)])
    return clamped


def _normalize_mesh_aspect(vertices, max_aspect=4.0):
    spans = _mesh_spans(vertices)
    sx = max(spans["x"], 1e-9)
    sz = max(spans["z"], 1e-9)
    aspect = max(sx, sz) / min(sx, sz)
    if aspect <= max_aspect:
        return vertices

    cx = float(np.mean([v[0] for v in vertices]))
    cz = float(np.mean([v[2] for v in vertices]))
    if sx > sz:
        target_x = sz * max_aspect
        scale_x = target_x / sx
        return [[cx + ((x - cx) * scale_x), y, z] for x, y, z in vertices]
    target_z = sx * max_aspect
    scale_z = target_z / sz
    return [[x, y, cz + ((z - cz) * scale_z)] for x, y, z in vertices]


def _enforce_min_mesh_size(vertices, min_span_xz=1.0, min_height=1.0):
    """Ensure a mesh remains visible/printable by enforcing minimum XY footprint and height."""
    spans = _mesh_spans(vertices)
    cx = float(np.mean([v[0] for v in vertices]))
    cz = float(np.mean([v[2] for v in vertices]))
    base_y = spans["min_y"]

    sx = max(spans["x"], 1e-9)
    sz = max(spans["z"], 1e-9)
    sy = max(spans["y"], 1e-9)

    # Preserve horizontal proportions (avoid skewed/distorted buildings).
    scale_h = max(1.0, float(min_span_xz) / sx, float(min_span_xz) / sz)
    scale_y = max(1.0, float(min_height) / sy)

    resized = []
    for x, y, z in vertices:
        resized.append([
            cx + ((x - cx) * scale_h),
            base_y + ((y - base_y) * scale_y),
            cz + ((z - cz) * scale_h),
        ])
    return resized


def _ecef_to_lon_lat(x, y, z):
    # WGS84 ellipsoid
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(5):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)
    sin_lat = math.sin(lat)
    n = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), float(h)


def _read_accessor(gltf, bin_chunk, accessor_index):
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    byte_offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    component_type = accessor["componentType"]
    count = accessor["count"]
    accessor_type = accessor["type"]
    type_map = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
    }
    comp_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    ncomp = type_map.get(accessor_type, 1)
    dtype = comp_map.get(component_type, np.float32)
    byte_stride = view.get("byteStride")
    if byte_stride:
        arr = np.frombuffer(bin_chunk, dtype=np.uint8, count=count * byte_stride, offset=byte_offset)
        arr = arr.reshape((count, byte_stride))
        raw = np.frombuffer(arr[:, : np.dtype(dtype).itemsize * ncomp].tobytes(), dtype=dtype)
        return raw.reshape((count, ncomp))
    raw = np.frombuffer(
        bin_chunk,
        dtype=dtype,
        count=count * ncomp,
        offset=byte_offset,
    )
    return raw.reshape((count, ncomp))


def _parse_b3dm(tile_bytes):
    if tile_bytes[:4] == b"glTF":
        # raw GLB without b3dm wrapper
        return {}, b"", {}, tile_bytes
    if tile_bytes[:4] != b"b3dm":
        raise RuntimeError("Unsupported tile format (expected b3dm or glb)")
    _, version, byte_len, ft_json, ft_bin, bt_json, bt_bin = struct.unpack("<4sIIIIII", tile_bytes[:28])
    header_len = 28
    ft_json_bytes = tile_bytes[header_len : header_len + ft_json]
    ft_bin_bytes = tile_bytes[header_len + ft_json : header_len + ft_json + ft_bin]
    bt_json_bytes = tile_bytes[header_len + ft_json + ft_bin : header_len + ft_json + ft_bin + bt_json]
    # bt_bin_bytes not used
    glb_start = header_len + ft_json + ft_bin + bt_json + bt_bin
    glb = tile_bytes[glb_start:]
    feature_table = json.loads(ft_json_bytes.decode("utf-8")) if ft_json_bytes else {}
    batch_table = json.loads(bt_json_bytes.decode("utf-8")) if bt_json_bytes else {}
    return feature_table, ft_bin_bytes, batch_table, glb


def _parse_glb(glb_bytes):
    if glb_bytes[:4] != b"glTF":
        raise RuntimeError("Invalid GLB header")
    # glb header
    _, _ = struct.unpack("<II", glb_bytes[4:12])
    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset + 8 <= len(glb_bytes):
        chunk_len, chunk_type = struct.unpack("<I4s", glb_bytes[offset : offset + 8])
        offset += 8
        chunk_data = glb_bytes[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            json_chunk = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk_data
    if json_chunk is None or bin_chunk is None:
        raise RuntimeError("GLB missing JSON or BIN chunk")
    return json_chunk, bin_chunk


def _extract_mesh_primitives(gltf, bin_chunk):
    meshes = []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            pos_idx = attrs.get("POSITION")
            if pos_idx is None:
                continue
            positions = _read_accessor(gltf, bin_chunk, pos_idx)
            batch_idx = None
            for key in ("_BATCHID", "BATCHID", "BATCH_ID", "batchId"):
                if key in attrs:
                    batch_idx = attrs[key]
                    break
            batch_ids = _read_accessor(gltf, bin_chunk, batch_idx).astype(np.int64).flatten() if batch_idx is not None else None
            indices = None
            if "indices" in prim:
                idx = _read_accessor(gltf, bin_chunk, prim["indices"]).astype(np.int64).flatten()
                indices = idx
            meshes.append({"positions": positions, "indices": indices, "batch_ids": batch_ids})
    return meshes


def _find_node_for_point(tileset_root, lon, lat):
    stack = [(tileset_root, np.eye(4))]
    best = None
    while stack:
        node, transform = stack.pop()
        if not node:
            continue
        bv = node.get("boundingVolume") or {}
        region = bv.get("region")
        if region:
            if not (
                math.radians(lon) >= region[0]
                and math.radians(lon) <= region[2]
                and math.radians(lat) >= region[1]
                and math.radians(lat) <= region[3]
            ):
                continue
        node_transform = node.get("transform")
        if node_transform:
            node_mat = np.array(node_transform, dtype=np.float64).reshape((4, 4)).T
            transform = transform @ node_mat
        content = node.get("content") or {}
        uri = content.get("uri") or content.get("url")
        if uri:
            best = (uri, transform, bv.get("region"))
        for child in node.get("children", []):
            stack.append((child, transform.copy()))
    return best


def _resolve_content_uri(base_url, access_token, tileset, lon, lat):
    root = tileset.get("root") or {}
    region = None
    # Traverse top-level tileset
    match = _find_node_for_point(root, lon, lat)
    if match:
        uri, transform, region = match
        while uri.endswith(".json"):
            subtree = _fetch_json_with_token(base_url, access_token, uri)
            sub_root = subtree.get("root") or {}
            sub_match = _find_node_for_point(sub_root, lon, lat)
            if not sub_match:
                return None
            uri, transform, region = sub_match
        return (uri, transform, region)
    # Fallback: check child subtrees
    for child in root.get("children", []):
        content = child.get("content") or {}
        uri = content.get("uri") or content.get("url")
        if not uri or not uri.endswith(".json"):
            continue
        while uri.endswith(".json"):
            subtree = _fetch_json_with_token(base_url, access_token, uri)
            sub_root = subtree.get("root") or {}
            sub_match = _find_node_for_point(sub_root, lon, lat)
            if not sub_match:
                break
            uri, transform, region = sub_match
        if uri and not uri.endswith(".json"):
            return (uri, transform, region)
    return None


def _fetch_json_with_token(base_url, access_token, uri):
    url = uri if uri.startswith("http") else f"{base_url.rsplit('/', 1)[0]}/{uri.lstrip('/')}"
    url = _append_access_token(url, access_token)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _uri_to_url(base_url, access_token, uri):
    url = uri if uri.startswith("http") else f"{base_url.rsplit('/', 1)[0]}/{uri.lstrip('/')}"
    return _append_access_token(url, access_token)


def _collect_intersecting_content(base_url, access_token, tileset_root, bounds, max_tiles):
    """Collect non-JSON content URIs whose tile bounds intersect the requested bounds."""
    stack = [(tileset_root, np.eye(4))]
    content_items = []
    seen_tiles = set()
    seen_subtrees = set()

    while stack and len(content_items) < max_tiles:
        node, transform = stack.pop()
        if not node:
            continue

        node_transform = node.get("transform")
        if node_transform:
            node_mat = np.array(node_transform, dtype=np.float64).reshape((4, 4)).T
            transform = transform @ node_mat

        bv = (node.get("boundingVolume") or {})
        region = bv.get("region")
        if isinstance(region, list) and len(region) >= 6:
            if not _region_intersects_bounds(region, bounds):
                continue

        content = node.get("content") or {}
        uri = content.get("uri") or content.get("url")
        if uri:
            if uri.endswith(".json"):
                subtree_url = _uri_to_url(base_url, access_token, uri)
                if subtree_url not in seen_subtrees:
                    seen_subtrees.add(subtree_url)
                    subtree = requests.get(subtree_url, timeout=30).json()
                    stack.append((subtree.get("root") or {}, transform.copy()))
            else:
                tile_url = _uri_to_url(base_url, access_token, uri)
                if tile_url not in seen_tiles:
                    seen_tiles.add(tile_url)
                    region_bbox = _bbox_from_region(region) if isinstance(region, list) and len(region) >= 6 else None
                    content_items.append((uri, transform.copy(), region_bbox))

        for child in node.get("children", []):
            if isinstance(child, dict):
                stack.append((child, transform.copy()))

    return content_items


def _build_feature_meshes_from_tile(
    uri,
    tile_bytes,
    transform,
    region_bbox,
    bounds,
    scale_factor,
    vertical_scale,
    elev_params,
    shape_clipper,
):
    feature_table, _ft_bin, _batch_table, glb = _parse_b3dm(tile_bytes)
    rtc_center = feature_table.get("RTC_CENTER")
    rtc_center = np.array(rtc_center, dtype=np.float64) if rtc_center is not None else np.zeros(3)
    gltf, bin_chunk = _parse_glb(glb)
    primitives = _extract_mesh_primitives(gltf, bin_chunk)
    if not primitives:
        return []

    min_elev = elev_params["min_elev"]
    elev_range = elev_params["elev_range"]
    avg_lat = elev_params["avg_lat"]
    size_scale = elev_params.get("size_scale", 1.0)
    model_width = max(
        (bounds["east"] - bounds["west"]) * scale_factor * math.cos(math.radians(avg_lat)),
        1.0,
    )
    model_depth = max((bounds["north"] - bounds["south"]) * scale_factor, 1.0)
    margin = max(model_width, model_depth) * 0.7
    max_building_span = max(model_width, model_depth) * 0.08
    min_building_height = 0.8 * size_scale

    def mesh_in_model_extent(vertices):
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        x_overlap = max_x >= (-margin) and min_x <= (model_width + margin)
        z_overlap = max_z >= (-margin) and min_z <= (model_depth + margin)
        y_reasonable = (min_y > -500.0) and (max_y < 3000.0)
        return x_overlap and z_overlap and y_reasonable

    def mesh_has_reasonable_footprint(vertices):
        spans = _mesh_spans(vertices)
        return max(spans["x"], spans["z"]) <= max_building_span

    bounds_pad = {
        "west": bounds["west"] - 0.5,
        "east": bounds["east"] + 0.5,
        "south": bounds["south"] - 0.5,
        "north": bounds["north"] + 0.5,
    }

    def _apply_mat(vertex, mat, use_rtc):
        local = np.array(vertex, dtype=np.float64)
        if use_rtc:
            local = local + rtc_center
        if mat is None:
            world = np.array([local[0], local[1], local[2], 1.0], dtype=np.float64)
        else:
            world = mat @ np.array([local[0], local[1], local[2], 1.0], dtype=np.float64)
        return world

    def _transform_score(sample_vertices, mat, use_rtc):
        inside = 0
        for vertex in sample_vertices:
            world = _apply_mat(vertex, mat, use_rtc)
            lat, lon, h = _ecef_to_lon_lat(world[0], world[1], world[2])
            if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(h)):
                continue
            if abs(h) > 20000:
                continue
            if (
                bounds_pad["west"] <= lon <= bounds_pad["east"]
                and bounds_pad["south"] <= lat <= bounds_pad["north"]
            ):
                inside += 1
        return inside

    sample_vertices = []
    for prim in primitives:
        if prim["positions"] is None or len(prim["positions"]) == 0:
            continue
        step = max(1, len(prim["positions"]) // 64)
        sample_vertices.extend(prim["positions"][::step][:64].tolist())
        if len(sample_vertices) >= 64:
            break
    if not sample_vertices:
        return []

    fallback_bbox = None
    if region_bbox is not None:
        mins = []
        maxs = []
        for prim in primitives:
            positions = prim.get("positions")
            if positions is None or len(positions) == 0:
                continue
            mins.append(np.min(positions, axis=0))
            maxs.append(np.max(positions, axis=0))
        if mins and maxs:
            min_all = np.min(np.array(mins), axis=0)
            max_all = np.max(np.array(maxs), axis=0)
            fallback_bbox = {
                "min_x": float(min_all[0]),
                "max_x": float(max_all[0]),
                "min_y": float(min_all[1]),
                "max_y": float(max_all[1]),
                "min_z": float(min_all[2]),
                "max_z": float(max_all[2]),
            }

    candidate_mats = [("identity", None), ("node", transform), ("node_t", transform.T)]
    if transform is not None:
        try:
            candidate_mats.append(("node_inv", np.linalg.inv(transform)))
        except np.linalg.LinAlgError:
            pass
        try:
            candidate_mats.append(("node_t_inv", np.linalg.inv(transform.T)))
        except np.linalg.LinAlgError:
            pass
    candidate_modes = []
    for name, mat in candidate_mats:
        candidate_modes.append((f"{name}+rtc", mat, True))
        candidate_modes.append((f"{name}-rtc", mat, False))

    best_name, best_mat, best_use_rtc, best_score = "identity+rtc", None, True, -1
    for name, mat, use_rtc in candidate_modes:
        score = _transform_score(sample_vertices, mat, use_rtc)
        if score > best_score:
            best_name, best_mat, best_use_rtc, best_score = name, mat, use_rtc, score
    print(f"[INFO] Cesium tile transform strategy uri={uri} mode={best_name} score={best_score}")

    def to_model(vertex, primitive=None):
        if best_score > 0:
            world = _apply_mat(vertex, best_mat, best_use_rtc)
            lat, lon, h = _ecef_to_lon_lat(world[0], world[1], world[2])
            x, z = _project_lon_lat_to_model(lon, lat, bounds, scale_factor, avg_lat)
            y = ((h - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale
            return x, y, z
        elif region_bbox is not None and fallback_bbox is not None:
            pb = fallback_bbox
            dx = max(pb["max_x"] - pb["min_x"], 1e-9)
            dy = max(pb["max_y"] - pb["min_y"], 1e-9)
            dz = max(pb["max_z"] - pb["min_z"], 1e-9)
            x_norm = (float(vertex[0]) - pb["min_x"]) / dx
            y_norm = (float(vertex[1]) - pb["min_y"]) / dy
            z_norm = (float(vertex[2]) - pb["min_z"]) / dz

            lon = region_bbox["west"] + x_norm * (region_bbox["east"] - region_bbox["west"])
            lat = region_bbox["south"] + z_norm * (region_bbox["north"] - region_bbox["south"])
            x, z = _project_lon_lat_to_model(lon, lat, bounds, scale_factor, avg_lat)
            # Fallback Y profile: keep plausible low-rise heights when world transforms are unavailable.
            y = (0.5 + (y_norm * 2.5)) * size_scale * vertical_scale
            return x, y, z
        else:
            return None

    feature_meshes = []
    mesh_counter = 0

    for prim in primitives:
        positions = prim["positions"]
        indices = prim["indices"]
        batch_ids = prim["batch_ids"]
        if indices is None or len(indices) < 3:
            continue

        if batch_ids is not None and len(batch_ids) == len(positions):
            grouped = {}
            for i in range(0, len(indices), 3):
                a, b, c = int(indices[i]), int(indices[i + 1]), int(indices[i + 2])
                ba, bb, bc = int(batch_ids[a]), int(batch_ids[b]), int(batch_ids[c])
                if ba != bb or bb != bc:
                    continue
                grouped.setdefault(ba, []).append((a, b, c))

            for batch_id, tris in grouped.items():
                vertex_map = {}
                vertices = []
                faces = []
                for a, b, c in tris:
                    tri = []
                    for vid in (a, b, c):
                        if vid not in vertex_map:
                            projected = to_model(positions[vid], primitive=prim)
                            if projected is None:
                                tri = []
                                break
                            x, y, z = projected
                            vertex_map[vid] = len(vertices)
                            vertices.append([x, y, z])
                        tri.append(vertex_map[vid])
                    if len(tri) == 3:
                        faces.append(tri)

                if not vertices or not faces:
                    continue

                if shape_clipper is not None:
                    cx = float(np.mean([v[0] for v in vertices]))
                    cz = float(np.mean([v[2] for v in vertices]))
                    inside = shape_clipper.is_inside(np.array([cx]), np.array([cz]))
                    if not bool(np.any(inside)):
                        continue

                vertices, faces = _repair_mesh(vertices, faces)
                if not vertices or not faces:
                    continue
                if not mesh_has_reasonable_footprint(vertices):
                    continue
                vertices = _inflate_mesh_height(vertices, min_building_height)
                if not mesh_in_model_extent(vertices):
                    continue

                feature_meshes.append(
                    {
                        "type": "building",
                        "id": f"tile_{uri}_{batch_id}_{mesh_counter}",
                        "name": "Cesium Building",
                        "building_type": "yes",
                        "vertices": vertices,
                        "faces": faces,
                        "source": "tiles",
                        "custom_color": None,
                    }
                )
                mesh_counter += 1
        else:
            vertices = []
            faces = []
            for i in range(0, len(indices), 3):
                tri = []
                for vid in (int(indices[i]), int(indices[i + 1]), int(indices[i + 2])):
                    projected = to_model(positions[vid], primitive=prim)
                    if projected is None:
                        tri = []
                        break
                    x, y, z = projected
                    vertices.append([x, y, z])
                    tri.append(len(vertices) - 1)
                if len(tri) == 3:
                    faces.append(tri)

            if not vertices or not faces:
                continue

            if shape_clipper is not None:
                cx = float(np.mean([v[0] for v in vertices]))
                cz = float(np.mean([v[2] for v in vertices]))
                inside = shape_clipper.is_inside(np.array([cx]), np.array([cz]))
                if not bool(np.any(inside)):
                    continue

            vertices, faces = _repair_mesh(vertices, faces)
            if not vertices or not faces:
                continue
            if not mesh_has_reasonable_footprint(vertices):
                continue
            vertices = _inflate_mesh_height(vertices, min_building_height)
            if not mesh_in_model_extent(vertices):
                continue

            feature_meshes.append(
                {
                    "type": "building",
                    "id": f"tile_{uri}_{mesh_counter}",
                    "name": "Cesium Tile Building",
                    "building_type": "yes",
                    "vertices": vertices,
                    "faces": faces,
                    "source": "tiles",
                    "custom_color": None,
                }
            )
            mesh_counter += 1

    return feature_meshes


def fetch_cesium_address_building_mesh(address_location, bounds, scale_factor, vertical_scale, elev_params, options):
    """Fetch a single Cesium building mesh near the address location."""
    token = get_cesium_ion_token()
    if not token:
        raise RuntimeError("CESIUM_ION_TOKEN is not configured")

    lat = float(address_location.get("lat"))
    lon = float(address_location.get("lon"))

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

    tileset_url = _append_access_token(base_url, access_token)
    tileset = requests.get(tileset_url, timeout=timeout).json()

    match = _resolve_content_uri(base_url, access_token, tileset, lon, lat)
    if not match:
        print("[WARN] Cesium address: no matching tile found")
        return None
    uri, transform, region = match
    if uri == "root.b3dm":
        print("[WARN] Cesium address: matched root.b3dm only")
        return None

    tile_url = uri if uri.startswith("http") else f"{base_url.rsplit('/', 1)[0]}/{uri.lstrip('/')}"
    tile_url = _append_access_token(tile_url, access_token)
    tile_resp = requests.get(tile_url, timeout=timeout)
    tile_resp.raise_for_status()
    print(f"[INFO] Cesium address tile uri={uri} bytes={len(tile_resp.content)} magic={tile_resp.content[:4]!r}")

    feature_table, _ft_bin, _batch_table, glb = _parse_b3dm(tile_resp.content)
    print(f"[INFO] Cesium address tile bytes={len(tile_resp.content)} feature_table={feature_table}")
    rtc_center = feature_table.get("RTC_CENTER")
    rtc_center = np.array(rtc_center, dtype=np.float64) if rtc_center is not None else np.zeros(3)
    gltf, bin_chunk = _parse_glb(glb)
    meshes = _extract_mesh_primitives(gltf, bin_chunk)
    if not meshes:
        print("[WARN] Cesium address: no meshes in GLB")
        return None

    # Apply transforms and pick closest batch
    min_elev = elev_params["min_elev"]
    elev_range = elev_params["elev_range"]
    avg_lat = elev_params["avg_lat"]
    size_scale = elev_params.get("size_scale", 1.0)
    region_bbox = _bbox_from_region(region) if isinstance(region, list) and len(region) >= 6 else None

    bounds_pad = {
        "west": bounds["west"] - 0.5,
        "east": bounds["east"] + 0.5,
        "south": bounds["south"] - 0.5,
        "north": bounds["north"] + 0.5,
    }

    def _apply_mat(vertex, mat, use_rtc):
        local = np.array(vertex, dtype=np.float64)
        if use_rtc:
            local = local + rtc_center
        if mat is None:
            world = np.array([local[0], local[1], local[2], 1.0], dtype=np.float64)
        else:
            world = mat @ np.array([local[0], local[1], local[2], 1.0], dtype=np.float64)
        return world

    def _transform_score(sample_vertices, mat, use_rtc):
        inside = 0
        for vertex in sample_vertices:
            world = _apply_mat(vertex, mat, use_rtc)
            lat_v, lon_v, h_v = _ecef_to_lon_lat(world[0], world[1], world[2])
            if not (np.isfinite(lat_v) and np.isfinite(lon_v) and np.isfinite(h_v)):
                continue
            if abs(h_v) > 20000:
                continue
            if (
                bounds_pad["west"] <= lon_v <= bounds_pad["east"]
                and bounds_pad["south"] <= lat_v <= bounds_pad["north"]
            ):
                inside += 1
        return inside

    sample_vertices = []
    for mesh in meshes:
        positions = mesh.get("positions")
        if positions is None or len(positions) == 0:
            continue
        step = max(1, len(positions) // 64)
        sample_vertices.extend(positions[::step][:64].tolist())
        if len(sample_vertices) >= 64:
            break

    fallback_bbox = None
    if region_bbox is not None:
        mins = []
        maxs = []
        for mesh in meshes:
            positions = mesh.get("positions")
            if positions is None or len(positions) == 0:
                continue
            mins.append(np.min(positions, axis=0))
            maxs.append(np.max(positions, axis=0))
        if mins and maxs:
            min_all = np.min(np.array(mins), axis=0)
            max_all = np.max(np.array(maxs), axis=0)
            fallback_bbox = {
                "min_x": float(min_all[0]),
                "max_x": float(max_all[0]),
                "min_y": float(min_all[1]),
                "max_y": float(max_all[1]),
                "min_z": float(min_all[2]),
                "max_z": float(max_all[2]),
            }

    candidate_mats = [("identity", None), ("node", transform), ("node_t", transform.T)]
    if transform is not None:
        try:
            candidate_mats.append(("node_inv", np.linalg.inv(transform)))
        except np.linalg.LinAlgError:
            pass
        try:
            candidate_mats.append(("node_t_inv", np.linalg.inv(transform.T)))
        except np.linalg.LinAlgError:
            pass
    candidate_modes = []
    for name, mat in candidate_mats:
        candidate_modes.append((f"{name}+rtc", mat, True))
        candidate_modes.append((f"{name}-rtc", mat, False))

    best_name, best_mat, best_use_rtc, best_score = "identity+rtc", None, True, -1
    if sample_vertices:
        for name, mat, use_rtc in candidate_modes:
            score = _transform_score(sample_vertices, mat, use_rtc)
            if score > best_score:
                best_name, best_mat, best_use_rtc, best_score = name, mat, use_rtc, score
    print(f"[INFO] Cesium address transform strategy uri={uri} mode={best_name} score={best_score}")

    best_batch = None
    best_dist = None
    batch_centroids = {}
    batch_counts = {}

    def project_lon_lat_to_model(lon_v, lat_v):
        x = (lon_v - bounds["west"]) * scale_factor * np.cos(np.radians(avg_lat))
        z = (bounds["north"] - lat_v) * scale_factor
        return x, z

    def _project_vertex(vertex, primitive=None):
        if best_score > 0:
            world = _apply_mat(vertex, best_mat, best_use_rtc)
            lat_v, lon_v, h_v = _ecef_to_lon_lat(world[0], world[1], world[2])
            x, z = project_lon_lat_to_model(lon_v, lat_v)
            y = ((h_v - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale
            return x, y, z
        if region_bbox is not None and fallback_bbox is not None:
            pb = fallback_bbox
            dx = max(pb["max_x"] - pb["min_x"], 1e-9)
            dy = max(pb["max_y"] - pb["min_y"], 1e-9)
            dz = max(pb["max_z"] - pb["min_z"], 1e-9)
            x_norm = (float(vertex[0]) - pb["min_x"]) / dx
            y_norm = (float(vertex[1]) - pb["min_y"]) / dy
            z_norm = (float(vertex[2]) - pb["min_z"]) / dz
            lon_v = region_bbox["west"] + x_norm * (region_bbox["east"] - region_bbox["west"])
            lat_v = region_bbox["south"] + z_norm * (region_bbox["north"] - region_bbox["south"])
            x, z = project_lon_lat_to_model(lon_v, lat_v)
            y = (0.5 + (y_norm * 2.5)) * size_scale * vertical_scale
            return x, y, z
        return None

    for mesh in meshes:
        positions = mesh["positions"]
        batch_ids = mesh["batch_ids"]
        if batch_ids is None:
            continue
        for idx, pos in enumerate(positions):
            projected = _project_vertex(pos, primitive=mesh)
            if projected is None:
                continue
            x, _y, z = projected
            # Approximate centroid in lat/lon domain for nearest-batch selection.
            lon_v = bounds["west"] + (x / (scale_factor * np.cos(np.radians(avg_lat))))
            lat_v = bounds["north"] - (z / scale_factor)
            bid = int(batch_ids[idx])
            acc = batch_centroids.get(bid)
            if acc is None:
                batch_centroids[bid] = np.array([lat_v, lon_v])
                batch_counts[bid] = 1
            else:
                batch_centroids[bid] += np.array([lat_v, lon_v])
                batch_counts[bid] += 1

    for bid, acc in batch_centroids.items():
        cnt = batch_counts[bid]
        if cnt <= 0:
            continue
        clat, clon = acc / cnt
        dist = (clat - lat) ** 2 + (clon - lon) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_batch = bid

    if best_batch is None:
        print("[WARN] Cesium address: no batch ids found")
        return None
    print(f"[INFO] Cesium address: selected batch {best_batch}")

    best_batch_bbox = None
    if best_score <= 0:
        mins = []
        maxs = []
        for mesh in meshes:
            positions = mesh["positions"]
            batch_ids = mesh["batch_ids"]
            if batch_ids is None:
                continue
            mask = (batch_ids == best_batch)
            if not np.any(mask):
                continue
            selected = positions[mask]
            mins.append(np.min(selected, axis=0))
            maxs.append(np.max(selected, axis=0))
        if mins and maxs:
            min_all = np.min(np.array(mins), axis=0)
            max_all = np.max(np.array(maxs), axis=0)
            best_batch_bbox = {
                "min_x": float(min_all[0]),
                "max_x": float(max_all[0]),
                "min_y": float(min_all[1]),
                "max_y": float(max_all[1]),
                "min_z": float(min_all[2]),
                "max_z": float(max_all[2]),
                "center_x": float((min_all[0] + max_all[0]) * 0.5),
                "center_z": float((min_all[2] + max_all[2]) * 0.5),
            }

    def _project_vertex_for_batch(vertex, primitive=None):
        if best_score > 0:
            return _project_vertex(vertex, primitive)
        if region_bbox is None or fallback_bbox is None or best_batch_bbox is None:
            return _project_vertex(vertex, primitive)
        pb = best_batch_bbox
        tile_pb = fallback_bbox
        dy = max(pb["max_y"] - pb["min_y"], 1e-9)
        tile_dx = max(tile_pb["max_x"] - tile_pb["min_x"], 1e-9)
        tile_dz = max(tile_pb["max_z"] - tile_pb["min_z"], 1e-9)
        lon_per_unit = (region_bbox["east"] - region_bbox["west"]) / tile_dx
        lat_per_unit = (region_bbox["north"] - region_bbox["south"]) / tile_dz
        x_off = float(vertex[0]) - pb["center_x"]
        z_off = float(vertex[2]) - pb["center_z"]
        y_norm = (float(vertex[1]) - pb["min_y"]) / dy
        lon_v = lon + (x_off * lon_per_unit)
        lat_v = lat + (z_off * lat_per_unit)
        x, z = project_lon_lat_to_model(lon_v, lat_v)
        y = (0.5 + (y_norm * 2.5)) * size_scale * vertical_scale
        return x, y, z

    vertices = []
    faces = []
    for mesh in meshes:
        positions = mesh["positions"]
        batch_ids = mesh["batch_ids"]
        indices = mesh["indices"]
        if batch_ids is None or indices is None:
            continue
        for i in range(0, len(indices), 3):
            a, b, c = indices[i : i + 3]
            if batch_ids[a] != best_batch or batch_ids[b] != best_batch or batch_ids[c] != best_batch:
                continue
            tri_idx = [a, b, c]
            face = []
            for vid in tri_idx:
                projected = _project_vertex_for_batch(positions[vid], primitive=mesh)
                if projected is None:
                    face = []
                    break
                x, y, z = projected
                vertices.append([x, y, z])
                face.append(len(vertices) - 1)
            if len(face) == 3:
                faces.append(face)

    if not vertices or not faces:
        print("[WARN] Cesium address: empty vertices/faces after extraction")
        return None

    vertices, faces = _repair_mesh(vertices, faces)
    if not vertices or not faces:
        print("[WARN] Cesium address: mesh invalid after repair")
        return None
    max_building_span = max(
        (bounds["east"] - bounds["west"]) * scale_factor * math.cos(math.radians(avg_lat)),
        (bounds["north"] - bounds["south"]) * scale_factor,
    ) * 0.08
    spans = _mesh_spans(vertices)
    if max(spans["x"], spans["z"]) > max_building_span:
        vertices = _clamp_mesh_footprint(vertices, max_building_span)
    # In address-only mode, keep original footprint shape from Cesium mesh.
    if not show_only_address:
        vertices = _normalize_mesh_aspect(vertices, max_aspect=4.0)
    vertices = _inflate_mesh_height(vertices, 0.8 * size_scale)
    # Address building must remain visible in the preview/final renderer.
    show_only_address = bool(options.get("show_only_address_building"))
    min_span = (12.0 if show_only_address else 6.0) * size_scale
    min_h = (8.0 if show_only_address else 4.0) * size_scale
    vertices = _enforce_min_mesh_size(
        vertices,
        min_span_xz=min_span,
        min_height=min_h,
    )
    # Keep it above local terrain/road overlays so it doesn't disappear visually.
    address_surface_y = options.get("_address_surface_y")
    min_y = min(v[1] for v in vertices)
    min_visible_y = 0.6 * size_scale
    if isinstance(address_surface_y, (int, float)) and np.isfinite(address_surface_y):
        min_visible_y = max(min_visible_y, float(address_surface_y) + (0.4 * size_scale))
    if min_y < min_visible_y:
        lift = min_visible_y - min_y
        vertices = [[x, y + lift, z] for x, y, z in vertices]

    # In address mode, prefer visual correctness at the searched address.
    # Cesium tile transform fallback can drift horizontally; recenter the mesh
    # to the exact address point when drift is noticeable.
    target_x, target_z = project_lon_lat_to_model(lon, lat)
    center_x = float(np.mean([v[0] for v in vertices]))
    center_z = float(np.mean([v[2] for v in vertices]))
    dx = target_x - center_x
    dz = target_z - center_z
    drift = math.hypot(dx, dz)
    if drift > (1.5 * size_scale):
        vertices = [[x + dx, y, z + dz] for x, y, z in vertices]
        print(f"[INFO] Cesium address: recentered mesh by {drift:.2f}mm to searched address")
    print(f"[INFO] Cesium address: mesh verts={len(vertices)} faces={len(faces)}")

    return {
        "type": "building",
        "id": f"cesium_address_{best_batch}",
        "name": "Cesium Address Building",
        "building_type": "yes",
        "vertices": vertices,
        "faces": faces,
        "is_address_building": True,
        "source": "cesium_tiles",
        "custom_color": None,
    }


def fetch_cesium_building_meshes(bounds, scale_factor, vertical_scale, elev_params, options, shape_clipper=None):
    """Fetch georeferenced Cesium building meshes from intersecting b3dm tiles."""
    token = get_cesium_ion_token()
    if not token:
        raise RuntimeError("CESIUM_ION_TOKEN is not configured")

    timeout = get_cesium_timeout_seconds()
    asset_id = get_cesium_osm_asset_id()
    endpoint_url = CESIUM_ION_ENDPOINT_TEMPLATE.format(asset_id=asset_id)
    headers = {"Authorization": f"Bearer {token}"}

    endpoint_resp = requests.get(endpoint_url, headers=headers, timeout=timeout)
    if not endpoint_resp.ok:
        print(
            "[ERROR] Cesium endpoint fetch failed:"
            f" status={endpoint_resp.status_code} url={endpoint_url} body={endpoint_resp.text[:500]!r}"
        )
    endpoint_resp.raise_for_status()
    endpoint = endpoint_resp.json()
    base_url = (endpoint.get("url") or "").strip()
    if not base_url:
        raise RuntimeError("Cesium endpoint response missing tileset URL")
    access_token = endpoint.get("accessToken", token)

    tileset_url = base_url
    tileset_url = _append_access_token(tileset_url, access_token)
    tileset_resp = requests.get(tileset_url, timeout=timeout)
    if not tileset_resp.ok:
        print(
            "[ERROR] Cesium tileset fetch failed:"
            f" status={tileset_resp.status_code} url={tileset_url} body={tileset_resp.text[:500]!r}"
        )
    tileset_resp.raise_for_status()
    tileset = tileset_resp.json()
    root = tileset.get("root") or {}
    max_tiles = 60 if options.get("preview_mode", False) else 180
    max_meshes = 120 if options.get("preview_mode", False) else 320

    debug_bounds = {k: bounds.get(k) for k in ("west", "south", "east", "north")}
    print(f"[INFO] Cesium tiles bounds: {debug_bounds} max_tiles={max_tiles}")

    content_items = _collect_intersecting_content(base_url, access_token, root, bounds, max_tiles)
    if not content_items:
        print("[WARN] Cesium tiles: no intersecting content found")
        return []
    # Prefer deeper LOD tiles and drop very coarse parent levels.
    z_levels = []
    for uri, _transform, _region in content_items:
        parts = str(uri).split("/")
        if parts:
            try:
                z_levels.append(int(parts[0]))
            except ValueError:
                pass
    if z_levels:
        min_keep_z = max(z_levels) - 1
        filtered_items = []
        for item in content_items:
            parts = str(item[0]).split("/")
            keep = True
            if parts:
                try:
                    keep = int(parts[0]) >= min_keep_z
                except ValueError:
                    keep = True
            if keep:
                filtered_items.append(item)
        content_items = filtered_items

    simplify_enabled = bool(options.get("building_mesh_simplify", True))
    ratio = options.get(
        "building_mesh_target_ratio_preview" if options.get("preview_mode", False) else "building_mesh_target_ratio_final",
        0.2 if options.get("preview_mode", False) else 0.4,
    )
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = 0.2 if options.get("preview_mode", False) else 0.4

    meshes = []
    for uri, transform, region_bbox in content_items:
        if len(meshes) >= max_meshes:
            break
        tile_url = _uri_to_url(base_url, access_token, uri)
        tile_resp = requests.get(tile_url, timeout=timeout)
        if tile_resp.status_code != 200:
            continue

        try:
            tile_meshes = _build_feature_meshes_from_tile(
                uri=uri,
                tile_bytes=tile_resp.content,
                transform=transform,
                region_bbox=region_bbox,
                bounds=bounds,
                scale_factor=scale_factor,
                vertical_scale=vertical_scale,
                elev_params=elev_params,
                shape_clipper=shape_clipper if options.get("building_tiles_clip", False) else None,
            )
        except Exception:
            continue

        for mesh in tile_meshes:
            if simplify_enabled:
                mesh["faces"] = _simplify_faces(mesh["faces"], ratio, min_keep=12)
            mesh["vertices"], mesh["faces"] = _repair_mesh(mesh["vertices"], mesh["faces"])
            if not mesh["vertices"] or not mesh["faces"]:
                continue
            meshes.append(mesh)
            if len(meshes) >= max_meshes:
                break

    print(f"[INFO] Cesium tile meshes extracted: {len(meshes)}")
    return meshes
