#!/usr/bin/env python3
"""
Topo3D - Topographical 3D Model Generator
Web application for creating 3D printable topographical models from GPX files and map data.
"""

import os
import time
import uuid
import requests
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils.gpx_parser import parse_gpx_file
from utils.elevation_fetcher import fetch_elevation_data
from utils.osm_fetcher import fetch_osm_features
from utils.mesh_generator import generate_mesh, export_to_stl, export_to_3mf
from utils.mesh_validator import MeshValidator
from utils.geocoder import geocode_address
from utils.app_config import (
    get_cors_origins,
    get_default_building_source_mode,
    get_default_terrain_source_mode,
    get_cesium_ion_token,
    get_cesium_osm_asset_id,
    get_cesium_timeout_seconds,
    parse_env_bool,
)
from utils.cesium_tiles_provider import CESIUM_ION_ENDPOINT_TEMPLATE

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": get_cors_origins()}})

# Configuration
UPLOAD_FOLDER = '/app/uploads'
EXPORT_FOLDER = '/app/exports'
ALLOWED_EXTENSIONS = {'gpx'}
CLEANUP_MAX_AGE_SECONDS = int(os.getenv('TOPO3D_FILE_TTL_SECONDS', str(24 * 3600)))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER
# Export requests include full mesh JSON and can exceed 16MB at final quality.
# Keep configurable for environments with tighter limits.
app.config['MAX_CONTENT_LENGTH'] = int(
    os.getenv('TOPO3D_MAX_CONTENT_LENGTH_MB', '256')
) * 1024 * 1024

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)


def cleanup_old_files(directory, max_age_seconds):
    """Remove files older than `max_age_seconds` from a directory."""
    now = time.time()
    try:
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            age = now - entry.stat().st_mtime
            if age > max_age_seconds:
                os.remove(entry.path)
    except Exception as e:
        print(f"[WARN] Cleanup failed for {directory}: {e}")


def build_unique_path(directory, original_filename, required_ext):
    """Build unique storage path while preserving user-facing download name."""
    sanitized = secure_filename(original_filename) or f"topo_model.{required_ext}"
    if not sanitized.endswith(f".{required_ext}"):
        sanitized = f"{sanitized}.{required_ext}"
    stem = sanitized[:-(len(required_ext) + 1)]
    unique_name = f"{stem}_{uuid.uuid4().hex[:10]}.{required_ext}"
    return sanitized, os.path.join(directory, unique_name)


def clamp_features_for_preview(features):
    """Reduce feature counts for fast preview generation."""
    if not isinstance(features, dict):
        return {}
    return {
        'roads': list(features.get('roads', []))[:80],
        'water': list(features.get('water', []))[:20],
        'buildings': list(features.get('buildings', []))[:80],
        'railways': list(features.get('railways', []))[:40],
        'landuse': list(features.get('landuse', []))[:20],
    }


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Cleanup stale local artifacts on startup.
cleanup_old_files(UPLOAD_FOLDER, CLEANUP_MAX_AGE_SECONDS)
cleanup_old_files(EXPORT_FOLDER, CLEANUP_MAX_AGE_SECONDS)


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_gpx():
    """Handle GPX file upload."""
    try:
        cleanup_old_files(app.config['UPLOAD_FOLDER'], CLEANUP_MAX_AGE_SECONDS)
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only GPX files allowed'}), 400

        download_name, filepath = build_unique_path(app.config['UPLOAD_FOLDER'], file.filename, 'gpx')
        file.save(filepath)
        try:
            gpx_data = parse_gpx_file(filepath)
        finally:
            # Keep upload directory clean; GPX is only needed for immediate parsing.
            if os.path.exists(filepath):
                os.remove(filepath)

        return jsonify({
            'success': True,
            'filename': download_name,
            'data': gpx_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/geocode', methods=['POST'])
def geocode():
    """Geocode an address to coordinates."""
    try:
        data = request.get_json()
        address = data.get('address', '')

        if not address:
            return jsonify({'error': 'No address provided'}), 400

        location = geocode_address(address)

        return jsonify({
            'success': True,
            'location': location
        })

    except Exception as e:
        print(f"[ERROR] Geocode failed for '{address}': {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cesium-diagnose', methods=['GET'])
def cesium_diagnose():
    """Diagnose Cesium asset endpoint + tileset access."""
    try:
        token = get_cesium_ion_token()
        if not token:
            return jsonify({'error': 'CESIUM_ION_TOKEN is not configured'}), 500

        asset_id = get_cesium_osm_asset_id()
        asset_url = f"https://api.cesium.com/v1/assets/{asset_id}"
        endpoint_url = CESIUM_ION_ENDPOINT_TEMPLATE.format(asset_id=asset_id)
        timeout = get_cesium_timeout_seconds()
        headers = {"Authorization": f"Bearer {token}"}

        asset_resp = requests.get(asset_url, headers=headers, timeout=timeout)
        asset_payload = {
            "status": asset_resp.status_code,
            "ok": asset_resp.ok,
            "body": asset_resp.text[:500],
        }
        if not asset_resp.ok:
            print(
                "[ERROR] Cesium diagnose asset fetch failed:"
                f" status={asset_resp.status_code} url={asset_url} body={asset_resp.text[:500]!r}"
            )

        endpoint_resp = requests.get(endpoint_url, headers=headers, timeout=timeout)
        endpoint_payload = {
            "status": endpoint_resp.status_code,
            "ok": endpoint_resp.ok,
            "body": endpoint_resp.text[:500],
        }
        if not endpoint_resp.ok:
            print(
                "[ERROR] Cesium diagnose endpoint fetch failed:"
                f" status={endpoint_resp.status_code} url={endpoint_url} body={endpoint_resp.text[:500]!r}"
            )
            return jsonify({"asset": asset_payload, "endpoint": endpoint_payload}), 502

        endpoint = endpoint_resp.json()
        tileset_url = (endpoint.get("url") or "").strip()
        access_token = endpoint.get("accessToken", token)
        if tileset_url and access_token:
            parsed = urlparse(tileset_url)
            query = dict(parse_qsl(parsed.query))
            query["access_token"] = access_token
            tileset_url = urlunparse(parsed._replace(query=urlencode(query)))

        tileset_resp = requests.get(tileset_url, timeout=timeout)
        tileset_payload = {
            "status": tileset_resp.status_code,
            "ok": tileset_resp.ok,
            "body": tileset_resp.text[:500],
        }
        if not tileset_resp.ok:
            print(
                "[ERROR] Cesium diagnose tileset fetch failed:"
                f" status={tileset_resp.status_code} url={tileset_url} body={tileset_resp.text[:500]!r}"
            )
            return jsonify({"asset": asset_payload, "endpoint": endpoint_payload, "tileset": tileset_payload}), 502

        return jsonify({
            "asset": asset_payload,
            "endpoint": endpoint_payload,
            "tileset": tileset_payload,
        })
    except Exception as e:
        print(f"[ERROR] Cesium diagnose failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/elevation', methods=['POST'])
def get_elevation():
    """Fetch elevation data for a bounding box."""
    try:
        data = request.get_json()
        bounds = data.get('bounds', {})
        resolution = data.get('resolution', 120)
        preview_mode = parse_env_bool(data.get('preview_mode'), default=False)
        # Preview uses hybrid (tries Cesium, falls back automatically); final forces high-detail DEM terrain.
        # Cesium building meshes still come from tiles in mesh generation.
        source_mode = 'hybrid' if preview_mode else 'default'

        if not all(k in bounds for k in ['north', 'south', 'east', 'west']):
            return jsonify({'error': 'Invalid bounds provided'}), 400

        elevation_data = fetch_elevation_data(
            bounds['north'],
            bounds['south'],
            bounds['east'],
            bounds['west'],
            resolution,
            source_mode=source_mode,
            preview_mode=preview_mode
        )

        return jsonify({
            'success': True,
            'elevation': elevation_data,
            'source': elevation_data.get('source', 'default'),
            'fallback_reason': elevation_data.get('fallback_reason')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/osm-features', methods=['POST'])
def get_osm_features():
    """Fetch OpenStreetMap features for a bounding box."""
    try:
        data = request.get_json()
        bounds = data.get('bounds', {})
        features = data.get('features', ['roads', 'water', 'buildings'])

        if not all(k in bounds for k in ['north', 'south', 'east', 'west']):
            return jsonify({'error': 'Invalid bounds provided'}), 400

        osm_data = fetch_osm_features(
            bounds['north'],
            bounds['south'],
            bounds['east'],
            bounds['west'],
            features
        )

        return jsonify({
            'success': True,
            'features': osm_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate_model():
    """Generate 3D model from provided data."""
    try:
        t_start = time.time()
        data = request.get_json()
        t_parse = time.time() - t_start
        print(f"[PERF] Request parsed in {t_parse:.3f}s")

        elevation = data.get('elevation', {})
        features = data.get('features', {})
        options = data.get('options', {})
        generation_mode = options.get('generation_mode')
        preview_mode = options.get('preview_mode')
        if preview_mode is None:
            # Default to preview mode unless explicitly marked final.
            preview_mode = generation_mode != 'final'
        else:
            preview_mode = bool(preview_mode)
        options.setdefault('building_mode', get_default_building_source_mode())
        options.setdefault('terrain_source_mode', get_default_terrain_source_mode())
        options.setdefault('building_mesh_simplify', True)
        options.setdefault('building_mesh_target_ratio_preview', 0.2)
        options.setdefault('building_mesh_target_ratio_final', 0.4)
        options['preview_mode'] = bool(preview_mode)

        if not elevation:
            return jsonify({'error': 'No elevation data provided'}), 400

        if preview_mode:
            features = clamp_features_for_preview(features)

        # Generate 3D mesh
        t_mesh_start = time.time()
        mesh_data = generate_mesh(elevation, features, options)
        t_mesh = time.time() - t_mesh_start
        print(f"[PERF] generate_mesh() took {t_mesh:.3f}s")

        # Validate and auto-fix mesh for 3D printability
        # Optimized with KD-tree and vectorized operations for fast performance
        t_validate_start = time.time()
        validator = MeshValidator()
        try:
            validation_result = validator.validate_and_fix(
                mesh_data,
                validate_features=not preview_mode,
                min_feature_size=250 if preview_mode else 0,
                check_manifold_edges=not preview_mode
            )
            t_validate = time.time() - t_validate_start
            print(f"[PERF] validate_and_fix() took {t_validate:.3f}s")
        except Exception as e:
            print(f"[ERROR] Validation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # If validation fails, continue without it
            validation_result = {
                'is_printable': True,
                'warnings': [f'Validation error: {str(e)}'],
                'fixes_applied': []
            }
            t_validate = time.time() - t_validate_start
            print(f"[PERF] validate_and_fix() failed in {t_validate:.3f}s")

        t_total = time.time() - t_start
        print(f"[PERF] Total /api/generate time: {t_total:.3f}s")
        timings = {
            'parse_seconds': round(t_parse, 4),
            'mesh_seconds': round(t_mesh, 4),
            'validation_seconds': round(t_validate, 4),
            'total_seconds': round(t_total, 4)
        }

        return jsonify({
            'success': True,
            'mesh': mesh_data,
            'validation': validation_result,
            'timings': timings,
            'metadata': mesh_data.get('metadata', {}),
            'mode': 'preview' if preview_mode else 'final'
        })

    except Exception as e:
        import traceback
        print(f"[ERROR] /api/generate failed: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/stl', methods=['POST'])
def export_stl():
    """Export model to STL format for 3D printing."""
    try:
        cleanup_old_files(app.config['EXPORT_FOLDER'], CLEANUP_MAX_AGE_SECONDS)
        data = request.get_json()
        mesh_data = data.get('mesh', {})
        filename = data.get('filename', 'topo_model.stl')

        if not mesh_data:
            return jsonify({'error': 'No mesh data provided'}), 400

        filename, filepath = build_unique_path(app.config['EXPORT_FOLDER'], filename, 'stl')

        # Export to STL
        export_to_stl(mesh_data, filepath)

        return send_file(
            filepath,
            mimetype='application/sla',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/3mf', methods=['POST'])
def export_3mf_route():
    """Export model to 3MF format with separate objects for multi-color printing."""
    try:
        cleanup_old_files(app.config['EXPORT_FOLDER'], CLEANUP_MAX_AGE_SECONDS)
        data = request.get_json()
        mesh_data = data.get('mesh', {})
        filename = data.get('filename', 'topo_model.3mf')

        if not mesh_data:
            return jsonify({'error': 'No mesh data provided'}), 400

        filename, filepath = build_unique_path(app.config['EXPORT_FOLDER'], filename, '3mf')

        # Export to 3MF
        export_to_3mf(mesh_data, filepath)

        return send_file(
            filepath,
            mimetype='application/vnd.ms-package.3dmanufacturing-3dmodel+xml',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Topo3D'
    })


if __name__ == '__main__':
    debug_enabled = parse_env_bool(os.getenv('TOPO3D_DEBUG'), default=False)
    app.run(host='0.0.0.0', port=5001, debug=debug_enabled)
