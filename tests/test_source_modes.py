import unittest
from unittest.mock import patch
import sys
import types
import numpy as np

# Lightweight stubs for environments without optional dependencies.
for module_name in ("requests", "srtm"):
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

if "PIL" not in sys.modules:
    pil_module = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")
    pil_module.Image = pil_image_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image_module

if "scipy" not in sys.modules:
    fake_scipy = types.ModuleType("scipy")
    fake_interpolate = types.ModuleType("scipy.interpolate")
    fake_ndimage = types.ModuleType("scipy.ndimage")
    fake_spatial = types.ModuleType("scipy.spatial")

    class _FakeRectBivariateSpline:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, lats, lons):
            return np.zeros((len(lats), len(lons)))

    class _FakeDelaunay:
        def __init__(self, *_args, **_kwargs):
            self.simplices = np.array([])

    fake_interpolate.RectBivariateSpline = _FakeRectBivariateSpline
    fake_ndimage.gaussian_filter = lambda arr, sigma=0: arr
    fake_spatial.Delaunay = _FakeDelaunay
    fake_scipy.interpolate = fake_interpolate
    fake_scipy.ndimage = fake_ndimage
    fake_scipy.spatial = fake_spatial
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.interpolate"] = fake_interpolate
    sys.modules["scipy.ndimage"] = fake_ndimage
    sys.modules["scipy.spatial"] = fake_spatial

if "stl" not in sys.modules:
    fake_stl = types.ModuleType("stl")
    fake_stl.mesh = types.SimpleNamespace(Mesh=object)
    sys.modules["stl"] = fake_stl

from app.utils.elevation_fetcher import fetch_elevation_data
from app.utils.mesh_generator import generate_mesh


def _sample_elevation(bounds):
    return {
        "grid": [[100.0, 102.0], [101.0, 103.0]],
        "lats": [bounds["south"], bounds["north"]],
        "lons": [bounds["west"], bounds["east"]],
        "bounds": bounds,
        "resolution": 2,
        "min_elevation": 100.0,
        "max_elevation": 103.0,
        "source": "default",
    }


def _sample_options():
    return {
        "vertical_scale": 1.2,
        "base_height": 5.0,
        "model_width": 100.0,
        "include_base": True,
        "model_shape": "square",
        "building_height_scale": 1.0,
    }


class TerrainSourceTests(unittest.TestCase):
    def test_hybrid_terrain_falls_back_to_default(self):
        bounds = {"north": 37.1, "south": 37.0, "east": -121.9, "west": -122.0}
        with patch("app.utils.elevation_fetcher.fetch_cesium_terrain_data", side_effect=RuntimeError("boom")):
            with patch("app.utils.elevation_fetcher.fetch_default_elevation_data", return_value=_sample_elevation(bounds)):
                payload = fetch_elevation_data(
                    bounds["north"],
                    bounds["south"],
                    bounds["east"],
                    bounds["west"],
                    resolution=2,
                    source_mode="hybrid",
                    preview_mode=False,
                )
        self.assertEqual(payload["source"], "default")
        self.assertIn("fallback_reason", payload)


class BuildingSourceTests(unittest.TestCase):
    def test_building_tiles_source_used_when_available(self):
        bounds = {"north": 37.1, "south": 37.0, "east": -121.9, "west": -122.0}
        elevation = _sample_elevation(bounds)
        tile_buildings = [
            {
                "type": "building",
                "id": "tile_1",
                "name": "Tile Building",
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "faces": [[0, 1, 2]],
            }
        ]
        with patch("app.utils.mesh_generator.fetch_cesium_building_meshes", return_value=tile_buildings):
            mesh = generate_mesh(elevation, {"buildings": []}, {**_sample_options(), "building_mode": "tiles"})
        self.assertEqual(mesh["metadata"]["building_source_used"], "tiles")

    def test_hybrid_falls_back_to_osm_extrude(self):
        bounds = {"north": 37.1, "south": 37.0, "east": -121.9, "west": -122.0}
        elevation = _sample_elevation(bounds)
        features = {
            "buildings": [
                {
                    "id": 100,
                    "type": "building",
                    "coordinates": [
                        {"lat": 37.01, "lon": -121.99},
                        {"lat": 37.02, "lon": -121.99},
                        {"lat": 37.02, "lon": -121.98},
                        {"lat": 37.01, "lon": -121.98},
                    ],
                    "tags": {"building": "yes"},
                    "height": 10.0,
                }
            ]
        }
        with patch("app.utils.mesh_generator.fetch_cesium_building_meshes", side_effect=RuntimeError("nope")):
            mesh = generate_mesh(elevation, features, {**_sample_options(), "building_mode": "hybrid"})
        self.assertEqual(mesh["metadata"]["building_source_used"], "osm_extrude")
        self.assertTrue(mesh["metadata"]["fallback_reasons"]["building"])


if __name__ == "__main__":
    unittest.main()
