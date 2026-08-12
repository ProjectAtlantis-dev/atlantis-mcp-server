"""Core quadtree tile storage primitives for the Terrain service port.

Connection lifecycle belongs exclusively to ``Database/database.py``. Every
function here accepts that shared connection explicitly and never opens or
closes SQLite itself.
"""

from __future__ import annotations

import datetime
import sqlite3
import zlib

import numpy as np

from dynamic_functions.Terrain.terrain_config import (
    GREENLAND_BBOX,
    MAX_TILE_DEPTH,
)
from dynamic_functions.Terrain.tile_address import (
    ancestor_tile_ids,
    format_tile_id,
    require_tile_id,
)


# 64 cells represented by 65 shared-edge vertices per axis.
GRID_N = 65

CONFIDENCE = {
    "empty": 0,
    "arcticdem": 5,
    "arcticdem_10m": 6,
}


class TileClobberError(RuntimeError):
    """Raised when a DEM write would replace a different stored payload."""

    def __init__(
        self,
        tile_id: str,
        existing_source: str,
        incoming_source: str,
        existing_updated_at: str,
    ) -> None:
        self.tile_id = tile_id
        self.existing_source = existing_source
        self.incoming_source = incoming_source
        self.existing_updated_at = existing_updated_at
        super().__init__(
            f"Refusing to clobber tile {tile_id}: "
            f"existing source={existing_source} "
            f"updated_at={existing_updated_at}, "
            f"incoming source={incoming_source}"
        )


def _parent_id(depth: int, column: int, row: int) -> str | None:
    if depth == 0:
        return None
    return format_tile_id(depth - 1, column // 2, row // 2)


def _compress_array(array: np.ndarray) -> bytes:
    return zlib.compress(array.tobytes(), level=6)


def _decompress_float32(blob: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=np.float32).reshape(
        (GRID_N, GRID_N)
    ).copy()


def _decompress_uint8(blob: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(
        (GRID_N, GRID_N)
    ).copy()


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


def ensure_tile_row(db: sqlite3.Connection, tile_id: str) -> bool:
    """Create the empty skeleton row for one explicitly requested tile."""

    depth, column, row = require_tile_id(tile_id)
    tiles_per_axis = 1 << depth
    if column >= tiles_per_axis or row >= tiles_per_axis:
        raise ValueError(f"terrain tile address is outside depth {depth}: {tile_id!r}")
    bbox = tile_bbox(depth, column, row)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor = db.execute(
        "INSERT OR IGNORE INTO tiles "
        "(tile_id, depth, col, row, x_min, y_min, x_max, y_max, "
        "parent_id, geometric_error, source, updated_at, heightmap, "
        "confidence_map) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        ),
    )
    return cursor.rowcount == 1


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


def write_dem(
    db: sqlite3.Connection,
    tile_id: str,
    heightmap: np.ndarray,
    source: str,
    vertical_datum: str,
    *,
    commit: bool = True,
) -> bool:
    """Store one decoded DEM without silently replacing existing data.

    Returns ``True`` for a new payload and ``False`` when the exact payload is
    already present. NaNs are stored as zero-height samples with zero
    confidence, matching the Flask write path.
    """

    if not isinstance(heightmap, np.ndarray):
        raise TypeError("heightmap must be a numpy array")
    if heightmap.shape != (GRID_N, GRID_N):
        raise ValueError(
            f"heightmap shape {heightmap.shape} != ({GRID_N}, {GRID_N})"
        )
    if heightmap.dtype != np.float32:
        raise TypeError(f"heightmap dtype {heightmap.dtype} != float32")
    if source not in CONFIDENCE or CONFIDENCE[source] == 0:
        raise ValueError(f"Unknown measured DEM source: {source}")
    if not isinstance(vertical_datum, str) or not vertical_datum.strip():
        raise ValueError("vertical_datum must be a non-empty string")

    ensure_tile_row(db, tile_id)
    confidence = np.where(
        np.isfinite(heightmap),
        np.uint8(CONFIDENCE[source]),
        np.uint8(0),
    )
    stored_heightmap = np.where(np.isfinite(heightmap), heightmap, 0.0).astype(
        np.float32
    )
    heightmap_blob = _compress_array(stored_heightmap)
    confidence_blob = _compress_array(confidence)
    geometric_error = compute_geometric_error(stored_heightmap)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    cursor = db.execute(
        "UPDATE tiles SET heightmap = ?, confidence_map = ?, "
        "geometric_error = ?, source = ?, vertical_datum = ?, updated_at = ? "
        "WHERE tile_id = ? AND heightmap IS NULL AND confidence_map IS NULL",
        (
            heightmap_blob,
            confidence_blob,
            geometric_error,
            source,
            vertical_datum,
            now,
            tile_id,
        ),
    )
    if cursor.rowcount == 1:
        if commit:
            db.commit()
        return True

    row = db.execute(
        "SELECT source, vertical_datum, updated_at, heightmap, confidence_map "
        "FROM tiles WHERE tile_id = ?",
        (tile_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown tile_id: {tile_id}")

    (
        existing_source,
        existing_vertical_datum,
        existing_updated_at,
        existing_heightmap,
        existing_confidence,
    ) = row
    if (
        existing_source == source
        and existing_vertical_datum == vertical_datum
        and existing_heightmap == heightmap_blob
        and existing_confidence == confidence_blob
    ):
        return False
    raise TileClobberError(
        tile_id,
        existing_source,
        source,
        existing_updated_at,
    )


def read_dem_payload(db: sqlite3.Connection, tile_id: str) -> dict | None:
    """Read and decode one stored DEM, restoring no-confidence samples to NaN."""

    row = db.execute(
        "SELECT source, vertical_datum, updated_at, geometric_error, "
        "heightmap, confidence_map "
        "FROM tiles WHERE tile_id = ?",
        (tile_id,),
    ).fetchone()
    if row is None or row[4] is None or row[5] is None:
        return None

    stored_heightmap = _decompress_float32(row[4])
    confidence_map = _decompress_uint8(row[5])
    heightmap = stored_heightmap.copy()
    heightmap[confidence_map == 0] = np.nan
    return {
        "tile_id": tile_id,
        "source": row[0],
        "vertical_datum": row[1],
        "updated_at": row[2],
        "geometric_error": row[3],
        "heightmap": heightmap,
        "confidence_map": confidence_map,
    }


def read_dem_with_ancestor(
    db: sqlite3.Connection,
    tile_id: str,
) -> dict | None:
    """Read exact DEM data or the nearest stored ancestor without writing."""

    requested_depth, _, _ = require_tile_id(tile_id)
    for candidate_id in ancestor_tile_ids(tile_id, include_self=True):
        payload = read_dem_payload(db, candidate_id)
        if payload is None:
            continue
        resolved_depth, _, _ = require_tile_id(candidate_id)
        return {
            **payload,
            "requested_tile_id": tile_id,
            "resolved_tile_id": candidate_id,
            "depth_delta": requested_depth - resolved_depth,
            "exact": candidate_id == tile_id,
        }
    return None


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
