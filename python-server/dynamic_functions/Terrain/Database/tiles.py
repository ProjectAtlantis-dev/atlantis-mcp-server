"""Core quadtree tile storage primitives for the Terrain service port.

Connection lifecycle belongs exclusively to ``Database/database.py``. Every
function here accepts that shared connection explicitly and never opens or
closes SQLite itself.
"""

from __future__ import annotations

import datetime
import sqlite3

import numpy as np

from dynamic_functions.Terrain.terrain_config import (
    GREENLAND_BBOX,
    MAX_TILE_DEPTH,
)
from dynamic_functions.Terrain.tile_address import format_tile_id


# 64 cells represented by 65 shared-edge vertices per axis.
GRID_N = 65


def _parent_id(depth: int, column: int, row: int) -> str | None:
    if depth == 0:
        return None
    return format_tile_id(depth - 1, column // 2, row // 2)


def tile_bbox(
    depth: int,
    column: int,
    row: int,
    root_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Compute one tile's bounds in the canonical quadtree square."""

    if root_bbox is None:
        root_bbox = GREENLAND_BBOX
    root_x0, root_y0, root_x1, root_y1 = root_bbox
    tiles_per_axis = 1 << depth
    tile_width = (root_x1 - root_x0) / tiles_per_axis
    tile_height = (root_y1 - root_y0) / tiles_per_axis
    x0 = root_x0 + column * tile_width
    y0 = root_y0 + row * tile_height
    return x0, y0, x0 + tile_width, y0 + tile_height


def compute_geometric_error(heightmap: np.ndarray | None) -> float:
    """Measure detail lost by halving and bilinearly restoring a heightmap."""

    if heightmap is None:
        return 0.0

    height, width = heightmap.shape
    downsampled = heightmap[::2, ::2]
    down_height, down_width = downsampled.shape
    row_index = np.linspace(0, down_height - 1, height)
    column_index = np.linspace(0, down_width - 1, width)
    row0 = np.floor(row_index).astype(int)
    column0 = np.floor(column_index).astype(int)
    row1 = np.minimum(row0 + 1, down_height - 1)
    column1 = np.minimum(column0 + 1, down_width - 1)
    row_fraction = (row_index - row0).astype(np.float32)
    column_fraction = (column_index - column0).astype(np.float32)
    restored = (
        downsampled[np.ix_(row0, column0)]
        * (1 - row_fraction[:, None])
        * (1 - column_fraction[None, :])
        + downsampled[np.ix_(row0, column1)]
        * (1 - row_fraction[:, None])
        * column_fraction[None, :]
        + downsampled[np.ix_(row1, column0)]
        * row_fraction[:, None]
        * (1 - column_fraction[None, :])
        + downsampled[np.ix_(row1, column1)]
        * row_fraction[:, None]
        * column_fraction[None, :]
    )
    return float(np.max(np.abs(heightmap - restored)))


def seed_tiles(
    db: sqlite3.Connection,
    max_depth: int = MAX_TILE_DEPTH,
    root_bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Populate empty quadtree skeletons through ``max_depth``."""

    if root_bbox is None:
        root_bbox = GREENLAND_BBOX
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    db.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("grid_resolution", str(GRID_N)),
    )
    db.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("max_depth", str(max_depth)),
    )
    db.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("bbox", ",".join(str(value) for value in root_bbox)),
    )

    for depth in range(max_depth + 1):
        rows = []
        for column in range(1 << depth):
            for row in range(1 << depth):
                tile_id = format_tile_id(depth, column, row)
                bbox = tile_bbox(depth, column, row, root_bbox)
                rows.append(
                    (
                        tile_id,
                        depth,
                        column,
                        row,
                        *bbox,
                        _parent_id(depth, column, row),
                        0.0,
                        "empty",
                        now,
                        None,
                        None,
                    )
                )
        db.executemany(
            "INSERT OR IGNORE INTO tiles "
            "(tile_id, depth, col, row, x_min, y_min, x_max, y_max, "
            "parent_id, geometric_error, source, updated_at, heightmap, "
            "confidence_map) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    db.commit()


def read_tile_metadata(
    db: sqlite3.Connection,
    tile_id: str,
) -> dict | None:
    """Read traversal metadata without decompressing terrain payloads."""

    row = db.execute(
        "SELECT tile_id, depth, col, row, x_min, y_min, x_max, y_max, "
        "geometric_error, source FROM tiles WHERE tile_id = ?",
        (tile_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "tile_id": row[0],
        "depth": row[1],
        "col": row[2],
        "row": row[3],
        "bbox": (row[4], row[5], row[6], row[7]),
        "geometric_error": row[8],
        "source": row[9],
    }


def get_metadata(db: sqlite3.Connection, key: str) -> str | None:
    """Read one database-level terrain setting."""

    row = db.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (key,),
    ).fetchone()
    return row[0] if row else None
