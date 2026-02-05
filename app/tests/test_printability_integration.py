import unittest
from collections import Counter
import sys
import types
import io
import os
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

# `generate_mesh` does not use `stl`, but mesh_generator imports it at module load.
# Provide a minimal stub for local environments without numpy-stl.
try:
    from stl import mesh as _stl_mesh  # noqa: F401
except ModuleNotFoundError:
    fake_stl = types.ModuleType("stl")
    fake_stl.mesh = types.SimpleNamespace(Mesh=object)
    sys.modules["stl"] = fake_stl

# `generate_mesh` does not require Delaunay in these test paths, but mesh_generator imports it.
try:
    from scipy.spatial import Delaunay as _delaunay  # noqa: F401
except ModuleNotFoundError:
    fake_scipy = types.ModuleType("scipy")
    fake_spatial = types.ModuleType("scipy.spatial")

    class _FakeDelaunay:
        def __init__(self, *_args, **_kwargs):
            self.simplices = np.array([])

    fake_spatial.Delaunay = _FakeDelaunay
    fake_scipy.spatial = fake_spatial
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.spatial"] = fake_spatial

from utils.mesh_generator import generate_mesh, export_to_stl, export_to_3mf
try:
    from utils.mesh_validator import MeshValidator
except Exception:  # pragma: no cover - local fallback when scipy is unavailable
    MeshValidator = None
from main import app


def make_elevation_data(resolution=18):
    south, north = 37.0, 37.02
    west, east = -122.02, -122.0
    lats = np.linspace(south, north, resolution)
    lons = np.linspace(west, east, resolution)

    # Smooth hill-like surface with mild variation.
    grid = np.zeros((resolution, resolution), dtype=np.float64)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            grid[i, j] = (
                120.0
                + 8.0 * np.sin((lat - south) * 220.0)
                + 6.0 * np.cos((lon - west) * 220.0)
            )

    return {
        "grid": grid.tolist(),
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "bounds": {"north": north, "south": south, "east": east, "west": west},
        "resolution": resolution,
        "min_elevation": float(np.min(grid)),
        "max_elevation": float(np.max(grid)),
    }


def make_elevation_data_for_bounds(bounds, resolution=24):
    south = bounds["south"]
    north = bounds["north"]
    west = bounds["west"]
    east = bounds["east"]

    lats = np.linspace(south, north, resolution)
    lons = np.linspace(west, east, resolution)
    grid = np.zeros((resolution, resolution), dtype=np.float64)

    lat_span = max(north - south, 1e-9)
    lon_span = max(east - west, 1e-9)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            lat_phase = (lat - south) / lat_span
            lon_phase = (lon - west) / lon_span
            grid[i, j] = 120.0 + 10.0 * np.sin(lat_phase * np.pi) + 8.0 * np.cos(lon_phase * np.pi)

    return {
        "grid": grid.tolist(),
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "bounds": bounds,
        "resolution": resolution,
        "min_elevation": float(np.min(grid)),
        "max_elevation": float(np.max(grid)),
    }


def make_features():
    return {
        "buildings": [
            {
                "id": 1001,
                "type": "building",
                "coordinates": [
                    {"lat": 37.0080, "lon": -122.0140},
                    {"lat": 37.0090, "lon": -122.0140},
                    {"lat": 37.0090, "lon": -122.0130},
                    {"lat": 37.0080, "lon": -122.0130},
                ],
                "tags": {"building": "yes"},
                "height": 12.0,
                "name": "test_building",
            }
        ],
        "roads": [
            {
                "id": 2001,
                "type": "road",
                "coordinates": [
                    {"lat": 37.0040, "lon": -122.0190},
                    {"lat": 37.0100, "lon": -122.0120},
                    {"lat": 37.0160, "lon": -122.0030},
                ],
                "road_type": "residential",
                "name": "test_road",
            }
        ],
        "water": [
            {
                "id": 3001,
                "type": "water",
                "coordinates": [
                    {"lat": 37.0110, "lon": -122.0180},
                    {"lat": 37.0135, "lon": -122.0165},
                    {"lat": 37.0130, "lon": -122.0140},
                    {"lat": 37.0105, "lon": -122.0138},
                ],
                "name": "test_pond",
            }
        ],
    }


def edge_multiplicity(faces):
    edge_counter = Counter()
    for face in faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        edge_counter[tuple(sorted((v0, v1)))] += 1
        edge_counter[tuple(sorted((v1, v2)))] += 1
        edge_counter[tuple(sorted((v2, v0)))] += 1
    return edge_counter


def assert_manifold_and_non_degenerate(testcase, mesh):
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int32)

    testcase.assertGreater(len(vertices), 0, "mesh has no vertices")
    testcase.assertGreater(len(faces), 0, "mesh has no faces")
    testcase.assertTrue(np.all(faces >= 0), "negative face index detected")
    testcase.assertTrue(np.all(faces < len(vertices)), "face index out of bounds")

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    testcase.assertTrue(np.all(areas > 1e-12), "degenerate face(s) detected")

    # Printable watertight solids should have every undirected edge used exactly twice.
    counts = edge_multiplicity(faces)
    bad_edges = [edge for edge, count in counts.items() if count != 2]
    testcase.assertEqual([], bad_edges, "non-manifold edge(s) detected")


class PrintabilityIntegrationTests(unittest.TestCase):
    GPX_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Topo3D Test"
     xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Integration Track</name>
    <trkseg>
      <trkpt lat="37.9130" lon="-122.5965"><ele>100</ele></trkpt>
      <trkpt lat="37.9170" lon="-122.5985"><ele>200</ele></trkpt>
      <trkpt lat="37.9210" lon="-122.5920"><ele>500</ele></trkpt>
      <trkpt lat="37.9130" lon="-122.5965"><ele>100</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""

    def test_terrain_is_watertight_for_all_supported_shapes(self):
        elevation = make_elevation_data()
        for shape in ("square", "circle", "rectangle", "hexagon"):
            with self.subTest(model_shape=shape):
                mesh = generate_mesh(
                    elevation,
                    {"buildings": [], "roads": [], "water": []},
                    {
                        "vertical_scale": 1.5,
                        "base_height": 10.0,
                        "model_width": 160.0,
                        "include_base": True,
                        "model_shape": shape,
                    },
                )
                assert_manifold_and_non_degenerate(self, mesh["terrain"])

    def test_generated_scene_is_printable_with_features(self):
        elevation = make_elevation_data()
        features = make_features()
        mesh = generate_mesh(
            elevation,
            features,
            {
                "vertical_scale": 1.3,
                "base_height": 8.0,
                "model_width": 180.0,
                "include_base": True,
                "model_shape": "square",
                "building_height_scale": 1.0,
                "road_height": 0.2,
            },
        )

        assert_manifold_and_non_degenerate(self, mesh["terrain"])
        for feature in mesh["features"]:
            assert_manifold_and_non_degenerate(self, feature)

    @unittest.skipIf(MeshValidator is None, "MeshValidator dependencies unavailable")
    def test_validator_reports_printable_for_generated_scene(self):
        elevation = make_elevation_data()
        features = make_features()
        mesh = generate_mesh(
            elevation,
            features,
            {
                "vertical_scale": 1.3,
                "base_height": 8.0,
                "model_width": 180.0,
                "include_base": True,
                "model_shape": "square",
                "building_height_scale": 1.0,
                "road_height": 0.2,
            },
        )
        result = MeshValidator().validate_and_fix(
            mesh, validate_features=True, min_feature_size=0
        )
        self.assertTrue(result["is_printable"])
        self.assertEqual([], result["warnings"])

    def test_api_upload_and_generate_mesh_from_gpx(self):
        client = app.test_client()

        upload_response = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(self.GPX_SAMPLE.encode("utf-8")), "integration.gpx")},
            content_type="multipart/form-data",
        )
        self.assertEqual(200, upload_response.status_code)
        upload_payload = upload_response.get_json()
        self.assertTrue(upload_payload["success"])
        self.assertIn("data", upload_payload)
        self.assertIn("tracks", upload_payload["data"])
        self.assertGreater(len(upload_payload["data"]["tracks"]), 0)

        bounds = upload_payload["data"]["bounds"]
        elevation = make_elevation_data_for_bounds(bounds, resolution=28)

        generate_response = client.post(
            "/api/generate",
            json={
                "elevation": elevation,
                "features": {"buildings": [], "roads": [], "water": []},
                "options": {
                    "include_base": True,
                    "model_shape": "square",
                    "model_width": 160.0,
                    "vertical_scale": 1.5,
                    "base_height": 8.0,
                    "gpx_tracks": upload_payload["data"]["tracks"],
                },
            },
        )
        self.assertEqual(200, generate_response.status_code)
        payload = generate_response.get_json()
        self.assertTrue(payload["success"])
        self.assertIn("mesh", payload)
        self.assertIn("validation", payload)
        self.assertIn("timings", payload)
        self.assertIn("total_seconds", payload["timings"])
        self.assertTrue(payload["validation"]["is_printable"])

        terrain = payload["mesh"]["terrain"]
        assert_manifold_and_non_degenerate(self, terrain)

        gpx_track = payload["mesh"]["gpx_track"]
        self.assertIsNotNone(gpx_track)
        assert_manifold_and_non_degenerate(self, gpx_track)

    def test_api_upload_and_generate_from_fixture_gpx(self):
        client = app.test_client()
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "mount_tam_loop.gpx")
        with open(fixture_path, "rb") as f:
            upload_response = client.post(
                "/api/upload",
                data={"file": (io.BytesIO(f.read()), "mount_tam_loop.gpx")},
                content_type="multipart/form-data",
            )
        self.assertEqual(200, upload_response.status_code)
        upload_payload = upload_response.get_json()
        self.assertTrue(upload_payload["success"])
        self.assertGreater(len(upload_payload["data"]["tracks"][0]["points"]), 10)

        bounds = upload_payload["data"]["bounds"]
        elevation = make_elevation_data_for_bounds(bounds, resolution=30)
        generate_response = client.post(
            "/api/generate",
            json={
                "elevation": elevation,
                "features": {"buildings": [], "roads": [], "water": []},
                "options": {
                    "include_base": True,
                    "model_shape": "circle",
                    "model_width": 170.0,
                    "vertical_scale": 1.6,
                    "base_height": 10.0,
                    "gpx_tracks": upload_payload["data"]["tracks"],
                },
            },
        )
        self.assertEqual(200, generate_response.status_code)
        payload = generate_response.get_json()
        self.assertTrue(payload["success"])
        assert_manifold_and_non_degenerate(self, payload["mesh"]["terrain"])

    def test_export_roundtrip_stl_and_3mf(self):
        elevation = make_elevation_data()
        mesh = generate_mesh(
            elevation,
            make_features(),
            {
                "vertical_scale": 1.3,
                "base_height": 8.0,
                "model_width": 180.0,
                "include_base": True,
                "model_shape": "hexagon",
                "building_height_scale": 1.0,
                "road_height": 0.2,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = os.path.join(tmpdir, "scene.stl")
            out_stl = export_to_stl(mesh, stl_path)
            self.assertTrue(out_stl["success"])
            self.assertTrue(os.path.exists(stl_path))
            self.assertGreater(os.path.getsize(stl_path), 0)

            three_mf_path = os.path.join(tmpdir, "scene.3mf")
            out_3mf = export_to_3mf(mesh, three_mf_path)
            self.assertTrue(out_3mf["success"])
            self.assertTrue(os.path.exists(three_mf_path))
            self.assertGreater(os.path.getsize(three_mf_path), 0)

            with zipfile.ZipFile(three_mf_path, "r") as zf:
                model_xml = zf.read("3D/3dmodel.model")
            root = ET.fromstring(model_xml)
            ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
            objects = root.findall(".//m:object", ns)
            self.assertGreater(len(objects), 0)
            for obj in objects:
                vertices = []
                for v in obj.findall(".//m:vertex", ns):
                    vertices.append([
                        float(v.attrib["x"]),
                        float(v.attrib["y"]),
                        float(v.attrib["z"]),
                    ])
                faces = []
                for tri in obj.findall(".//m:triangle", ns):
                    faces.append([
                        int(tri.attrib["v1"]),
                        int(tri.attrib["v2"]),
                        int(tri.attrib["v3"]),
                    ])
                if vertices and faces:
                    assert_manifold_and_non_degenerate(
                        self, {"vertices": vertices, "faces": faces}
                    )

    def test_shape_clipping_drops_partial_polygons(self):
        elevation = make_elevation_data()
        partially_outside_water = {
            "water": [
                {
                    "id": 4001,
                    "type": "water",
                    "coordinates": [
                        {"lat": 37.0005, "lon": -122.0195},
                        {"lat": 37.0195, "lon": -122.0195},
                        {"lat": 37.0195, "lon": -122.0005},
                        {"lat": 37.0300, "lon": -121.9900},
                    ],
                    "name": "outside_water",
                }
            ],
            "roads": [],
            "buildings": [],
        }
        mesh = generate_mesh(
            elevation,
            partially_outside_water,
            {
                "vertical_scale": 1.2,
                "base_height": 8.0,
                "model_width": 180.0,
                "include_base": True,
                "model_shape": "circle",
            },
        )
        self.assertEqual([], mesh["features"], "Partially out-of-bounds polygon should be rejected")


if __name__ == "__main__":
    unittest.main()
