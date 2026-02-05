import os
import unittest
from unittest.mock import patch

from app.utils.app_config import (
    get_default_building_source_mode,
    get_default_terrain_source_mode,
    get_terrain_max_samples,
)


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
