import os
import unittest
from unittest.mock import patch

from app.utils.app_config import (
    get_cors_origins,
    get_default_building_source_mode,
    get_default_terrain_source_mode,
    get_terrain_max_samples,
    parse_env_bool,
)


class ParseEnvBoolTests(unittest.TestCase):
    def test_true_values(self):
        for value in ["1", "true", "TRUE", " yes ", "On", "y"]:
            self.assertTrue(parse_env_bool(value, default=False))

    def test_false_values(self):
        for value in ["0", "false", "FALSE", " no ", "Off", "n"]:
            self.assertFalse(parse_env_bool(value, default=True))

    def test_unknown_value_uses_default(self):
        self.assertTrue(parse_env_bool("maybe", default=True))
        self.assertFalse(parse_env_bool("maybe", default=False))

    def test_none_uses_default(self):
        self.assertTrue(parse_env_bool(None, default=True))
        self.assertFalse(parse_env_bool(None, default=False))


class CorsOriginsTests(unittest.TestCase):
    def test_default_origins_are_localhost_only_patterns(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TOPO3D_CORS_ORIGINS", None)
            origins = get_cors_origins()
        self.assertEqual(
            origins,
            [r"^http://localhost(:\d+)?$", r"^http://127\.0\.0\.1(:\d+)?$"],
        )

    def test_env_override_parses_csv_and_trims(self):
        with patch.dict(
            os.environ,
            {"TOPO3D_CORS_ORIGINS": " http://localhost:3000,https://example.com , "},
            clear=False,
        ):
            origins = get_cors_origins()
        self.assertEqual(origins, ["http://localhost:3000", "https://example.com"])


class AppConfigTests(unittest.TestCase):
    def test_default_modes(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TOPO3D_BUILDING_SOURCE_DEFAULT", None)
            os.environ.pop("TOPO3D_TERRAIN_SOURCE_DEFAULT", None)
            self.assertEqual(get_default_building_source_mode(), "hybrid")
            self.assertEqual(get_default_terrain_source_mode(), "hybrid")

    def test_invalid_modes_fall_back_to_hybrid(self):
        with patch.dict(
            os.environ,
            {
                "TOPO3D_BUILDING_SOURCE_DEFAULT": "bad",
                "TOPO3D_TERRAIN_SOURCE_DEFAULT": "bad",
            },
            clear=False,
        ):
            self.assertEqual(get_default_building_source_mode(), "hybrid")
            self.assertEqual(get_default_terrain_source_mode(), "hybrid")

    def test_terrain_max_samples_switches_by_preview(self):
        with patch.dict(
            os.environ,
            {
                "TOPO3D_CESIUM_TERRAIN_MAX_SAMPLES_PREVIEW": "111",
                "TOPO3D_CESIUM_TERRAIN_MAX_SAMPLES_FINAL": "222",
            },
            clear=False,
        ):
            self.assertEqual(get_terrain_max_samples(True), 111)
            self.assertEqual(get_terrain_max_samples(False), 222)


if __name__ == "__main__":
    unittest.main()
