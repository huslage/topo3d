#!/usr/bin/env python3
"""
Topo3D - Topographical 3D Model Generator
Web application for creating 3D printable topographical models from GPX files and map data.
"""

import os
import json
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils.gpx_parser import parse_gpx_file
from utils.elevation_fetcher import fetch_elevation_data
from utils.osm_fetcher import fetch_osm_features
from utils.mesh_generator import generate_mesh, export_to_stl, export_to_3mf
from utils.mesh_validator import MeshValidator
from utils.geocoder import geocode_address

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/app/uploads'
EXPORT_FOLDER = '/app/exports'
ALLOWED_EXTENSIONS = {'gpx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_gpx():
    """Handle GPX file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only GPX files allowed'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Parse GPX file
        gpx_data = parse_gpx_file(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
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
        return jsonify({'error': str(e)}), 500


@app.route('/api/elevation', methods=['POST'])
def get_elevation():
    """Fetch elevation data for a bounding box."""
    try:
        data = request.get_json()
        bounds = data.get('bounds', {})
        resolution = data.get('resolution', 100)

        if not all(k in bounds for k in ['north', 'south', 'east', 'west']):
            return jsonify({'error': 'Invalid bounds provided'}), 400

        elevation_data = fetch_elevation_data(
            bounds['north'],
            bounds['south'],
            bounds['east'],
            bounds['west'],
            resolution
        )

        return jsonify({
            'success': True,
            'elevation': elevation_data
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
        data = request.get_json()

        elevation = data.get('elevation', {})
        features = data.get('features', {})
        options = data.get('options', {})

        if not elevation:
            return jsonify({'error': 'No elevation data provided'}), 400

        # Generate 3D mesh
        mesh_data = generate_mesh(elevation, features, options)

        # Validate and auto-fix mesh for 3D printability
        validator = MeshValidator()
        validation_result = validator.validate_and_fix(mesh_data)

        return jsonify({
            'success': True,
            'mesh': mesh_data,
            'validation': validation_result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/stl', methods=['POST'])
def export_stl():
    """Export model to STL format for 3D printing."""
    try:
        data = request.get_json()
        mesh_data = data.get('mesh', {})
        filename = data.get('filename', 'topo_model.stl')

        if not mesh_data:
            return jsonify({'error': 'No mesh data provided'}), 400

        # Sanitize filename
        filename = secure_filename(filename)
        if not filename.endswith('.stl'):
            filename += '.stl'

        filepath = os.path.join(app.config['EXPORT_FOLDER'], filename)

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
        data = request.get_json()
        mesh_data = data.get('mesh', {})
        filename = data.get('filename', 'topo_model.3mf')

        if not mesh_data:
            return jsonify({'error': 'No mesh data provided'}), 400

        # Sanitize filename
        filename = secure_filename(filename)
        if not filename.endswith('.3mf'):
            filename += '.3mf'

        filepath = os.path.join(app.config['EXPORT_FOLDER'], filename)

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
    app.run(host='0.0.0.0', port=5001, debug=True)
