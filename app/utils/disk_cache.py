"""Simple JSON disk cache helpers for local development."""

import hashlib
import json
import os
import time


DEFAULT_CACHE_DIR = os.getenv("TOPO3D_CACHE_DIR", "/app/exports/cache")


def _cache_file_path(namespace, key_payload, cache_dir=DEFAULT_CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    serialized = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{namespace}_{digest}.json")


def load_json_cache(namespace, key_payload, max_age_seconds=None, cache_dir=DEFAULT_CACHE_DIR):
    """Load a cached JSON object if present and not expired."""
    path = _cache_file_path(namespace, key_payload, cache_dir=cache_dir)
    if not os.path.exists(path):
        return None

    if max_age_seconds is not None:
        age_seconds = time.time() - os.path.getmtime(path)
        if age_seconds > max_age_seconds:
            return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json_cache(namespace, key_payload, value, cache_dir=DEFAULT_CACHE_DIR):
    """Persist a JSON-serializable value in the disk cache."""
    path = _cache_file_path(namespace, key_payload, cache_dir=cache_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f)
    return path
