# Topo3D - Topographical 3D Model Generator

Create stunning 3D printable topographical models from GPX files and open data sources.

![Topo3D](https://via.placeholder.com/800x400?text=Topo3D+Screenshot)

## Features

- 📁 **GPX File Upload**: Import your hiking tracks, bike routes, or any GPX data
- 🌍 **Elevation Data**: Automatic SRTM elevation data fetching
- 🗺️ **OpenStreetMap Integration**: Add roads, buildings, water bodies, and railways
- 📍 **Address Highlighting**: Geocode and highlight specific addresses
- 🎨 **Interactive 3D Viewer**: Real-time Three.js visualization with pan, rotate, zoom
- 🎯 **Object Selection**: Click to select and delete individual features
- 🏷️ **Custom Labels**: Add text labels to your model
- ⚙️ **Customizable Options**:
  - Vertical scale (exaggeration)
  - Model size (width in mm)
  - Base height for 3D printing
  - Feature selection (toggle roads, water, buildings, etc.)
- 📦 **STL Export**: One-click export for 3D printing
- 🐳 **Docker Ready**: Easy deployment with Docker Compose

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   cd topo3d
   ```

2. **Start the container**:
   ```bash
   docker-compose up -d
   ```

3. **Open your browser**:
   ```
   http://localhost:5001
   ```

### Manual Installation

1. **Install Python 3.11+**:
   ```bash
   python3 --version
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   cd app
   python main.py
   ```

4. **Open your browser**:
   ```
   http://localhost:5001
   ```

## How to Use

### 1. Upload a GPX File

- Click "Choose GPX File" and select your .gpx file
- The app will automatically extract:
  - Track points and routes
  - Waypoints
  - Geographic bounds

### 2. Configure Map Features

Select which features to include:
- ✅ **Roads**: Highways, streets, paths
- ✅ **Water Bodies**: Lakes, rivers, streams
- ✅ **Buildings**: All structures with height data
- ⬜ **Railways**: Train tracks and stations

### 3. Adjust Model Options

Fine-tune your model:
- **Vertical Scale** (0.5x - 5x): Exaggerate elevation changes
- **Model Width** (50mm - 500mm): Final print size
- **Base Height** (0mm - 50mm): Platform height for printing
- **Include Base**: Add solid base for 3D printing stability

### 4. Generate the Model

Click **"Generate 3D Model"** and watch your terrain come to life in the 3D viewer!

### 5. Interact with the Model

**3D Viewer Controls:**
- **Rotate**: Left click + drag
- **Pan**: Right click + drag
- **Zoom**: Scroll wheel
- **Select**: Click on any object

**Object Management:**
- View all objects in the sidebar list
- Click to select/highlight objects
- Click "Delete" to remove unwanted features

### 6. Add Labels (Optional)

1. Enter text in the "Add Label" field
2. Click "Add"
3. Position labels by selecting them in the 3D viewer

### 7. Highlight an Address (Optional)

1. Enter an address in the "Highlight Address" field
2. Click "Geocode Address"
3. A red marker will appear at the location

### 8. Export to STL

Click **"Export to STL"** to download your model for 3D printing!

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Web Browser                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Three.js 3D Viewer                              │   │
│  │  - Interactive controls                          │   │
│  │  - Object selection                              │   │
│  │  - Real-time rendering                           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      │ REST API
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Flask Backend (Python)                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  API Endpoints                                   │   │
│  │  - /api/upload (GPX file)                        │   │
│  │  - /api/geocode (Address lookup)                 │   │
│  │  - /api/elevation (SRTM data)                    │   │
│  │  - /api/osm-features (OpenStreetMap)             │   │
│  │  - /api/generate (3D mesh)                       │   │
│  │  - /api/export/stl (STL file)                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Utility Modules                                 │   │
│  │  - gpx_parser.py                                 │   │
│  │  - elevation_fetcher.py (SRTM)                   │   │
│  │  - osm_fetcher.py (Overpass API)                 │   │
│  │  - geocoder.py (Nominatim)                       │   │
│  │  - mesh_generator.py (3D geometry)               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  External Data Sources                                  │
│  - SRTM Elevation Data                                  │
│  - OpenStreetMap (Overpass API)                         │
│  - Nominatim Geocoding                                  │
└─────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend
- **Flask**: Web framework
- **NumPy**: Numerical computations
- **numpy-stl**: STL file generation
- **gpxpy**: GPX file parsing
- **srtm.py**: Elevation data access
- **overpy**: OpenStreetMap Overpass API client
- **geopy**: Geocoding with Nominatim

### Frontend
- **Three.js**: 3D visualization
- **Vanilla JavaScript**: No framework overhead
- **Responsive CSS**: Mobile-friendly design

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Easy orchestration

## Data Sources

### Elevation Data: SRTM
The app uses Shuttle Radar Topography Mission (SRTM) data:
- **Coverage**: Global (60°N to 56°S)
- **Resolution**: ~30m (1 arc-second)
- **Source**: NASA/USGS
- **License**: Public domain

### Map Features: OpenStreetMap
OpenStreetMap provides feature data via the Overpass API:
- **Roads**: All highway types
- **Buildings**: With height data when available
- **Water**: Natural water bodies and waterways
- **Railways**: Train tracks
- **License**: ODbL (Open Database License)

### Geocoding: Nominatim
Address geocoding powered by OSM Nominatim:
- **Coverage**: Worldwide
- **Rate Limit**: 1 request/second (please be respectful)
- **License**: ODbL

## 3D Printing Tips

### Model Preparation
1. **Choose appropriate vertical scale**: 1.5x-2x works well for most terrains
2. **Set base height**: 10mm provides good stability
3. **Model size**: 150-200mm width is ideal for most printers
4. **Include base**: Always enable for successful prints

### Printing Settings
- **Layer Height**: 0.2mm (standard)
- **Infill**: 15-20% (models are mostly solid terrain)
- **Supports**: Usually not needed with base
- **Material**: PLA recommended for beginners
- **Build Plate Adhesion**: Brim or raft recommended

### Post-Processing
- Sand smooth if desired
- Paint with acrylics:
  - Green/brown for terrain
  - Blue for water
  - Gray for buildings/roads
- Apply clear coat for protection

## API Reference

### POST /api/upload
Upload a GPX file.

**Request:**
- Multipart form data with `file` field

**Response:**
```json
{
  "success": true,
  "filename": "route.gpx",
  "data": {
    "tracks": [...],
    "waypoints": [...],
    "bounds": {
      "north": 45.5,
      "south": 45.0,
      "east": -122.5,
      "west": -123.0
    }
  }
}
```

### POST /api/geocode
Geocode an address.

**Request:**
```json
{
  "address": "1600 Amphitheatre Parkway, Mountain View, CA"
}
```

**Response:**
```json
{
  "success": true,
  "location": {
    "address": "1600 Amphitheatre Pkwy, Mountain View, CA 94043",
    "lat": 37.4224764,
    "lon": -122.0842499
  }
}
```

### POST /api/elevation
Fetch elevation data for a bounding box.

**Request:**
```json
{
  "bounds": {
    "north": 45.5,
    "south": 45.0,
    "east": -122.5,
    "west": -123.0
  },
  "resolution": 100
}
```

**Response:**
```json
{
  "success": true,
  "elevation": {
    "grid": [[...], [...]],
    "lats": [...],
    "lons": [...],
    "bounds": {...},
    "resolution": 100,
    "min_elevation": 100.0,
    "max_elevation": 500.0
  }
}
```

### POST /api/osm-features
Fetch OpenStreetMap features.

**Request:**
```json
{
  "bounds": {
    "north": 45.5,
    "south": 45.0,
    "east": -122.5,
    "west": -123.0
  },
  "features": ["roads", "water", "buildings"]
}
```

**Response:**
```json
{
  "success": true,
  "features": {
    "roads": [...],
    "water": [...],
    "buildings": [...]
  }
}
```

### POST /api/generate
Generate 3D mesh from data.

**Request:**
```json
{
  "elevation": {...},
  "features": {...},
  "options": {
    "vertical_scale": 1.5,
    "model_width": 200,
    "base_height": 10,
    "include_base": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "mesh": {
    "terrain": {...},
    "features": [...],
    "metadata": {...}
  }
}
```

### POST /api/export/stl
Export model to STL format.

**Request:**
```json
{
  "mesh": {...},
  "filename": "my_model.stl"
}
```

**Response:**
Binary STL file download

## Development

### Project Structure
```
topo3d/
├── app/
│   ├── main.py              # Flask application
│   ├── templates/
│   │   └── index.html       # Web UI
│   └── utils/
│       ├── __init__.py
│       ├── gpx_parser.py    # GPX file parsing
│       ├── geocoder.py      # Address geocoding
│       ├── elevation_fetcher.py  # SRTM data
│       ├── osm_fetcher.py   # OpenStreetMap data
│       └── mesh_generator.py     # 3D mesh generation
├── uploads/                 # Uploaded GPX files
├── exports/                 # Generated STL files
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker image
├── docker-compose.yml      # Docker Compose config
└── README.md              # This file
```

### Running Tests
```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest --cov=app tests/
```

### Adding New Features
Want to contribute? Here are some ideas:
- Historical elevation profile charts
- Multi-route comparison
- Custom color schemes
- Advanced labeling (elevation markers, contour lines)
- Integration with other mapping services
- Support for KML files
- Texture mapping from satellite imagery

## Troubleshooting

### "No elevation data available"
- SRTM data has limited coverage (60°N to 56°S)
- Try a different location
- Check your internet connection

### "OSM query timeout"
- Large areas can take time to fetch
- Try reducing the bounding box size
- Disable some feature types

### "STL export failed"
- Ensure model was generated successfully first
- Check browser console for errors
- Try with fewer features selected

### Docker container won't start
```bash
# Check logs
docker-compose logs -f

# Rebuild container
docker-compose down
docker-compose up --build
```

## Performance Notes

- **Small areas** (< 10km²): Fast, real-time generation
- **Medium areas** (10-50km²): 10-30 seconds
- **Large areas** (> 50km²): May take several minutes

**Tips for better performance:**
- Reduce resolution for large areas
- Disable features you don't need
- Use smaller model width for testing

## License

This project is released under the MIT License.

## Credits

Built with love using:
- [Flask](https://flask.palletsprojects.com/)
- [Three.js](https://threejs.org/)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [SRTM Data](https://www2.jpl.nasa.gov/srtm/)
- [Nominatim](https://nominatim.org/)

## Support

Found a bug? Have a feature request?

Open an issue on GitHub or contribute a pull request!

## Acknowledgments

Special thanks to:
- NASA/USGS for SRTM elevation data
- OpenStreetMap contributors
- The open-source community

---

**Happy 3D Printing!** 🏔️🖨️
