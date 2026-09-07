"""Exact input coverage for one fixed depth-8 regional bathymetry job."""

import sqlite3
import zlib

import numpy as np

from dynamic_functions.Terrain.tile_address import require_tile_id

JOB_DEPTH = 8
SOLVE_DEPTH = 9
SLOPE_DEPTH = 11
CARVE_DEPTH = 12
GRID = 65


def job_regions(tile_id: str) -> dict[int, tuple[int, int, int, int]]:
    depth, col, row = require_tile_id(tile_id)
    if depth != JOB_DEPTH:
        raise ValueError(f"bathymetry jobs must be depth {JOB_DEPTH}")
    if col >= (1 << depth) or row >= (1 << depth):
        raise ValueError(f"bathymetry tile address outside depth-{depth} grid")
    if tile_id != f"{depth}-{col}-{row}":
        raise ValueError(f"noncanonical bathymetry tile id: {tile_id!r}")
    regions = {}
    for level in range(JOB_DEPTH, CARVE_DEPTH + 1):
        scale = 1 << (level - depth)
        regions[level] = (col * scale, (col + 1) * scale - 1,
                          row * scale, (row + 1) * scale - 1)
    # The solve has one depth-9 tile of halo. Detailed slope evidence covers
    # that same rectangle; the final carve has one depth-12 tile of apron.
    c0, c1, r0, r1 = regions[SOLVE_DEPTH]
    limit = (1 << SOLVE_DEPTH) - 1
    solve = (max(0, c0 - 1), min(limit, c1 + 1),
             max(0, r0 - 1), min(limit, r1 + 1))
    regions[SOLVE_DEPTH] = solve
    factor = 1 << (SLOPE_DEPTH - SOLVE_DEPTH)
    regions[SLOPE_DEPTH] = (solve[0] * factor, (solve[1] + 1) * factor - 1,
                            solve[2] * factor, (solve[3] + 1) * factor - 1)
    c0, c1, r0, r1 = regions[CARVE_DEPTH]
    limit = (1 << CARVE_DEPTH) - 1
    regions[CARVE_DEPTH] = (max(0, c0 - 1), min(limit, c1 + 1),
                            max(0, r0 - 1), min(limit, r1 + 1))
    return regions


def region_rows(connection: sqlite3.Connection, depth: int, rect: tuple) -> dict:
    rows = connection.execute(
        "SELECT t.tile_id,t.heightmap,c.width,c.height,c.mask "
        "FROM tiles t LEFT JOIN coastline_masks c ON c.tile_id=t.tile_id "
        "WHERE t.depth=? AND t.col BETWEEN ? AND ? AND t.row BETWEEN ? AND ?",
        (depth, *rect),
    ).fetchall()
    return {row[0]: row[1:] for row in rows}


def decode_water(width, height, blob) -> np.ndarray:
    if (width, height) != (GRID, GRID):
        raise ValueError(f"bathymetry requires {GRID}x{GRID} coastline masks")
    values = np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(GRID, GRID)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("coastline mask contains non-boolean samples")
    return values.astype(bool)


def missing_inputs(connection: sqlite3.Connection, tile_id: str) -> dict[str, list[str]]:
    missing = {"dem": [], "coastline": []}
    for depth, rect in job_regions(tile_id).items():
        rows = region_rows(connection, depth, rect)
        for col in range(rect[0], rect[1] + 1):
            for row in range(rect[2], rect[3] + 1):
                tid = f"{depth}-{col}-{row}"
                heightmap, width, height, mask = rows.get(tid, (None, None, None, None))
                all_water = False
                if mask is None:
                    missing["coastline"].append(tid)
                else:
                    all_water = bool(decode_water(width, height, mask).all())
                if heightmap is None and not all_water:
                    missing["dem"].append(tid)
    return missing
