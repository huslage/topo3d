import unittest
import os
import sys

import numpy as np

# Ensure imports work when tests run from repository root.
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from main import app


def make_elevation_data(resolution=18):
    south, north = 37.0, 37.02
    west, east = -122.02, -122.0
    lats = np.linspace(south, north, resolution)
    lons = np.linspace(west, east, resolution)

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


class GenerateApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _generate(self, excluded_keys=None):
        response = self.client.post(
            "/api/generate",
            json={
                "elevation": make_elevation_data(),
                "features": make_features(),
                "options": {
                    "include_base": True,
                    "model_shape": "square",
                    "model_width": 180.0,
                    "vertical_scale": 1.4,
                    "base_height": 10.0,
                    "preview_mode": False,
                    "excluded_feature_keys": excluded_keys or [],
                },
            },
        )
        self.assertEqual(200, response.status_code)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        return payload

    def test_feature_key_present_on_every_feature(self):
        payload = self._generate()
        for feature in payload["mesh"]["features"]:
            self.assertIn("feature_key", feature)
            self.assertIsInstance(feature["feature_key"], str)
            self.assertIn(":", feature["feature_key"])

    def test_excluded_feature_keys_remove_targeted_features(self):
        payload = self._generate(excluded_keys=["road:2001"])
        returned_keys = {feature["feature_key"] for feature in payload["mesh"]["features"]}
        self.assertNotIn("road:2001", returned_keys)

    def test_feature_counts_by_type_matches_feature_payload(self):
        payload = self._generate()
        counts = {}
        for feature in payload["mesh"]["features"]:
            counts[feature["type"]] = counts.get(feature["type"], 0) + 1

        metadata = payload["metadata"]
        self.assertIn("feature_type_counts", metadata)
        self.assertIn("feature_counts_by_type", metadata)
        self.assertEqual(metadata["feature_type_counts"], metadata["feature_counts_by_type"])
        self.assertEqual(metadata["feature_counts_by_type"], counts)

    def test_backward_compatible_feature_fields_preserved(self):
        payload = self._generate()
        for feature in payload["mesh"]["features"]:
            self.assertIn("id", feature)
            self.assertIn("type", feature)


if __name__ == "__main__":
    unittest.main()
