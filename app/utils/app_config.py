"""Application configuration helpers."""

import os


def parse_env_bool(value, default=False):
    """Parse a boolean-like environment value with a fallback default."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
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
