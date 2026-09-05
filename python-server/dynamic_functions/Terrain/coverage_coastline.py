"""Read-only GTK50 coastline geometry for the coverage atlas."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from shapely import wkb as shapely_wkb
from shapely.ops import transform as shapely_transform

from dynamic_functions.Terrain.coastline import (
    BLOCK_DIR,
    SOURCE,
    VERSION,
    _TO_STEREO,
    _gpkg_wkb,
)


SIMPLIFY_METERS = 50.0
_cache_lock = threading.Lock()
_cache_fingerprint: tuple | None = None
_cache_payload: dict | None = None


def _fingerprint(paths: list[Path]) -> tuple:
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    )


def _line_coordinates(geometry) -> list[list[list[int]]]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        coordinates = [
            [round(float(x)), round(float(y))]
            for x, y, *_ in geometry.coords
        ]
        return [coordinates] if len(coordinates) >= 2 else []
    lines = []
    for part in getattr(geometry, "geoms", ()):
        lines.extend(_line_coordinates(part))
    return lines


def _read_boundary_lines(path: Path) -> list[list[list[int]]]:
    lines = []
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = source.execute(
            'SELECT geom FROM "landwaterboundary_c"'
        ).fetchall()
        for (blob,) in rows:
            if blob is None:
                continue
            geometry = shapely_wkb.loads(_gpkg_wkb(blob))
            projected = shapely_transform(_TO_STEREO.transform, geometry)
            simplified = projected.simplify(
                SIMPLIFY_METERS,
                preserve_topology=False,
            )
            lines.extend(_line_coordinates(simplified))
    finally:
        source.close()
    return lines


def query_available_coastline() -> dict:
    """Return detailed boundaries from every locally acquired GTK50 block."""

    global _cache_fingerprint, _cache_payload
    paths = sorted(BLOCK_DIR.glob("*.gpkg"))
    fingerprint = _fingerprint(paths)
    with _cache_lock:
        if fingerprint == _cache_fingerprint and _cache_payload is not None:
            return _cache_payload
        lines = []
        failures = []
        for path in paths:
            try:
                lines.extend(_read_boundary_lines(path))
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                failures.append({"block": path.name, "error": str(exc)})
        payload = {
            "source": SOURCE,
            "version": VERSION,
            "crs": "EPSG:3413",
            "simplifyMeters": SIMPLIFY_METERS,
            "blocks": [path.name for path in paths],
            "lines": lines,
            "failures": failures,
        }
        _cache_fingerprint = fingerprint
        _cache_payload = payload
        return payload
