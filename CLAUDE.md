# Topo3D - 3D Topographical Model Generator

## Project Overview

A Flask web application for creating 3D printable topographical models from GPX tracks and map locations. Users can upload GPX files or enter addresses to generate detailed terrain models with roads, buildings, water bodies, and custom tracks.

## Architecture

```
topo3d/
├── app/
│   ├── main.py              # Flask application and API routes
│   ├── templates/
│   │   └── index.html       # Single-page web UI (vanilla JS + Three.js)
│   └── utils/
│       ├── mesh_generator.py    # 3D mesh generation (terrain, features, base)
│       ├── elevation_fetcher.py # AWS Terrain RGB tile fetching
│       ├── osm_fetcher.py       # OpenStreetMap feature fetching
│       ├── geocoder.py          # Address geocoding via Google Maps URLs
│       └── gpx_parser.py        # GPX file parsing
├── uploads/                 # Uploaded GPX files
├── exports/                 # Generated STL/3MF files
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Key Technologies

- **Backend**: Flask, Python 3.11
- **Frontend**: Vanilla JavaScript, Three.js for 3D preview
- **Data Sources**:
  - AWS Terrain RGB tiles (elevation data)
  - OpenStreetMap/Overpass API (roads, buildings, water)
  - Google Maps URL parsing (geocoding)
- **3D Export**: numpy-stl for STL, custom XML for 3MF

## Running the Application

```bash
# Docker (recommended)
docker-compose up -d
# Open http://localhost:5001

# Local development
cd topo3d
pip install -r requirements.txt
python app/main.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/upload` | POST | Upload GPX file |
| `/api/geocode` | POST | Geocode address from Google Maps URL |
| `/api/elevation` | POST | Fetch elevation grid for bounds |
| `/api/osm-features` | POST | Fetch OSM features (roads, water, buildings) |
| `/api/generate` | POST | Generate 3D mesh from elevation + features |
| `/api/export/stl` | POST | Export mesh to STL format |
| `/api/export/3mf` | POST | Export mesh to 3MF format (multi-color) |

## Core Code Concepts

### Coordinate Systems

- **Geographic**: lat/lon in degrees (WGS84)
- **Three.js Model**: X = east-west, Y = up (elevation), Z = north-south
- **Elevation normalization**: `(raw_elev - min_elev) / elev_range * 20.0 * vertical_scale`

### Mesh Generation (`mesh_generator.py`)

Key functions:
- `generate_mesh()` - Main entry point, combines terrain + features
- `generate_terrain_mesh()` - Grid-based terrain from elevation data
- `generate_circular_base()` - Smooth circular wall for circular models
- `generate_building_meshes()` - Building boxes from OSM data
- `generate_road_meshes()` - Road strips following terrain
- `generate_water_meshes()` - Water body polygons
- `generate_gpx_track_mesh()` - GPX track as 3D ribbon

### Circular Model Clipping

For circular models:
1. Terrain vertices outside circle radius are excluded via `inside_circle` mask
2. Boundary vertices are moved outward to exact circle radius
3. Smooth 360-segment wall follows terrain contour (interpolated elevation)
4. Buildings/features are clipped if any corner falls outside circle

### Face Winding

All faces use CCW (counter-clockwise) winding when viewed from outside:
- Terrain: CCW from above (+Y)
- Walls: CCW from outside (outward normal)
- Bottom: CCW from below (-Y)

## Common Tasks

### Adding a New Feature Type

1. Add fetch function in `osm_fetcher.py`
2. Add mesh generator in `mesh_generator.py` (follow `generate_road_meshes` pattern)
3. Add to `generate_mesh()` feature processing
4. Update frontend to toggle feature in UI

### Debugging Mesh Issues

1. Check face winding (should be CCW from outside)
2. Verify vertex indices in face arrays
3. Use Three.js `DoubleSide` material temporarily to see both face sides
4. Export to STL and check in slicer for non-manifold edges

### Elevation Data Issues

- AWS tiles may have gaps at high latitudes - fallback to SRTM
- Tile boundaries can cause seams - gaussian_filter smooths them
- Resolution affects detail vs performance (default: 200x200 grid)

## Testing

Run the Flask app locally and use the web UI at http://localhost:5001. Test with:
1. GPX file upload (sample in `example.gpx`)
2. Google Maps URL paste for address geocoding
3. Various model sizes and circular/square modes
4. STL and 3MF export

## Export Formats

### STL
- Single mesh, no color
- Universal 3D printing compatibility

### 3MF
- Separate objects per feature type (terrain, roads, buildings, water, GPX track)
- Color-coded for multi-material printing
- Proper units (millimeters)

## Known Limitations

- Elevation tiles limited to ~30m resolution globally
- OSM data quality varies by region
- Large areas (>10km) may be slow to process
- Circular model edge has slight variation from grid-based terrain mesh
