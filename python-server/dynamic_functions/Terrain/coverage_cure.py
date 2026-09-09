"""Read-only cure inventory for terrain coverage and viewer diagnostics."""

from __future__ import annotations

import sqlite3

from dynamic_functions.Terrain.coastline import (
    SOURCE as COASTLINE_SOURCE,
    VERSION as COASTLINE_VERSION,
)
from dynamic_functions.Terrain.Database.database import connection_lock, db
from dynamic_functions.Terrain.terrain_config import MAX_TILE_DEPTH


def coverage_cure_inventory(
    connection: sqlite3.Connection,
    cure_depth: int = 11,
) -> dict:
    """Require exact DEM, current coastline, and texture at the cure depth."""

    if isinstance(cure_depth, bool) or not isinstance(cure_depth, int):
        raise TypeError("cure_depth must be an integer")
    if not 0 <= cure_depth <= MAX_TILE_DEPTH:
        raise ValueError(f"cure_depth must be between 0 and {MAX_TILE_DEPTH}")

    inventory = []
    rows = connection.execute(
        "SELECT t.tile_id, t.depth, t.x_min, t.y_min, t.x_max, t.y_max, "
        "       t.heightmap IS NOT NULL, c.source, c.version, "
        "       CASE WHEN length(x.texture) > 0 THEN x.source END "
        "FROM tiles t "
        "LEFT JOIN coastline_masks c ON c.tile_id = t.tile_id "
        "LEFT JOIN textures x ON x.tile_id = t.tile_id "
        "WHERE t.depth BETWEEN 0 AND ? "
        "  AND (t.heightmap IS NOT NULL OR c.tile_id IS NOT NULL "
        "       OR x.tile_id IS NOT NULL) "
        "ORDER BY t.depth, t.tile_id",
        (cure_depth,),
    )
    for row in rows:
        (
            tile_id,
            depth,
            x_min,
            y_min,
            x_max,
            y_max,
            has_dem,
            coastline_source,
            coastline_version,
            texture_source,
        ) = row
        coastline_cured = bool(
            coastline_source == COASTLINE_SOURCE
            and coastline_version is not None
            and int(coastline_version) >= COASTLINE_VERSION
        )
        exact_depth = int(depth) == cure_depth
        cured = (
            exact_depth and bool(has_dem) and coastline_cured
            and bool(texture_source)
        )
        inventory.append(
            {
                "tile": tile_id,
                "depth": int(depth),
                "bbox": [x_min, y_min, x_max, y_max],
                "status": "cured" if cured else "partial",
                "dem": bool(has_dem),
                "coastline": coastline_cured,
                "texture": texture_source,
            }
        )

    exact = [tile for tile in inventory if tile["depth"] == cure_depth]
    return {
        "cureDepth": cure_depth,
        "cureVersion": 2,
        "definition": (
            f"depth-{cure_depth} DEM, current authoritative coastline mask, "
            f"and exact depth-{cure_depth} texture (no ancestor imagery)"
        ),
        "summary": {
            "cured": sum(tile["status"] == "cured" for tile in exact),
            "partial": sum(tile["status"] == "partial" for tile in exact),
            "coarse": sum(tile["depth"] < cure_depth for tile in inventory),
        },
        "tiles": inventory,
    }


def query_coverage_cure() -> dict:
    """Read the cure inventory through the one shared Terrain connection."""

    with connection_lock():
        return coverage_cure_inventory(db())
