"""Application configuration helpers."""

import os


DEFAULT_CESIUM_OSM_ASSET_ID = 96188
DEFAULT_CESIUM_TERRAIN_ASSET_ID = 1

def parse_env_bool(value, default=False):
    """Parse a boolean-like environment value with a fallback default."""
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def get_cors_origins():
    """
    Return CORS origins from env, or localhost-only defaults.

    `TOPO3D_CORS_ORIGINS` supports a comma-separated list.
    """
    raw = os.getenv('TOPO3D_CORS_ORIGINS', '')
    if raw.strip():
        return [origin.strip() for origin in raw.split(',') if origin.strip()]
    return [r"^http://localhost(:\d+)?$", r"^http://127\.0\.0\.1(:\d+)?$"]


def parse_env_int(name, default):
    """Parse an integer environment value with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def get_cesium_ion_token():
    return os.getenv("CESIUM_ION_TOKEN", "").strip()


def get_cesium_timeout_seconds():
    return max(1, parse_env_int("TOPO3D_CESIUM_TIMEOUT_SECONDS", 15))


def get_cesium_osm_asset_id():
    return parse_env_int("CESIUM_OSM_ASSET_ID", DEFAULT_CESIUM_OSM_ASSET_ID)


def get_cesium_terrain_asset_id():
    return parse_env_int("CESIUM_TERRAIN_ASSET_ID", DEFAULT_CESIUM_TERRAIN_ASSET_ID)


def get_default_terrain_source_mode():
    mode = os.getenv("TOPO3D_TERRAIN_SOURCE_DEFAULT", "hybrid").strip().lower()
    if mode in {"default", "cesium", "hybrid"}:
        return mode
    return "hybrid"


def get_default_building_source_mode():
    mode = os.getenv("TOPO3D_BUILDING_SOURCE_DEFAULT", "hybrid").strip().lower()
    if mode in {"osm_extrude", "tiles", "hybrid"}:
        return mode
    return "hybrid"


def get_terrain_max_samples(preview_mode):
    """Maximum terrain sample points allowed for Cesium source."""
    if preview_mode:
        return max(1, parse_env_int("TOPO3D_CESIUM_TERRAIN_MAX_SAMPLES_PREVIEW", 20000))
    return max(1, parse_env_int("TOPO3D_CESIUM_TERRAIN_MAX_SAMPLES_FINAL", 60000))
