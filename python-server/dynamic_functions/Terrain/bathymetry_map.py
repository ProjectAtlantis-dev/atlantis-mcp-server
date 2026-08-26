"""Read-only map payload for target-owned bathymetry coverage."""

from __future__ import annotations

import math
import sqlite3


def _intersects_circle(
    bbox: tuple[float, float, float, float], qx: float, qy: float, radius: float
) -> bool:
    x_min, y_min, x_max, y_max = bbox
    x = min(max(qx, x_min), x_max)
    y = min(max(qy, y_min), y_max)
    return math.hypot(x - qx, y - qy) <= radius


def query_bathymetry_map(
    connection: sqlite3.Connection,
    qx: float,
    qy: float,
    max_range: float,
    *,
    ox: float,
    oy: float,
) -> dict:
    radius = max(float(max_range), 0.0)
    bounds = (qx - radius, qy - radius, qx + radius, qy + radius)
    rows = connection.execute(
        "SELECT b.tile_id,t.x_min,t.y_min,t.x_max,t.y_max,"
        "b.source,b.version,b.updated_at FROM bathymetry b "
        "JOIN tiles t ON t.tile_id=b.tile_id "
        "WHERE t.x_max>=? AND t.x_min<=? AND t.y_max>=? AND t.y_min<=? "
        "ORDER BY b.tile_id",
        (bounds[0], bounds[2], bounds[1], bounds[3]),
    ).fetchall()
    coverage = []
    for row in rows:
        bbox = tuple(float(value) for value in row[1:5])
        if not _intersects_circle(bbox, qx, qy, radius):
            continue
        coverage.append({
            "tileId": row[0],
            "bbox": [bbox[0] - ox, bbox[1] - oy, bbox[2] - ox, bbox[3] - oy],
            "source": row[5], "version": int(row[6]), "updatedAt": row[7],
        })
    return {
        "coverage": coverage,
        "soundings": [],
        "coverageCount": len(coverage),
        "soundingCount": 0,
        "soundingStatus": "not_imported",
        "qx": qx,
        "qy": qy,
    }
