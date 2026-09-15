"""Build and atomically publish one Terrain-owned regional bathymetry job.

Run as a module with the server's Python interpreter. All computation reads a
single SQLite snapshot; the source DEM and coastline tables are never written.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np
from scipy import ndimage

from dynamic_functions.Terrain.Database.bathymetry import write_bathymetry
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import require_tile_id
from .detail import add_detail
from .inputs import (CARVE_DEPTH, GRID, JOB_DEPTH, SOLVE_DEPTH, SLOPE_DEPTH,
                     decode_water, job_regions, missing_inputs, region_rows)
from .lod import decimate, parents_of
from .raster import (limit_shoreline_drop, pin_waterline,
                     repair_resampled_waterline, sample_global)
from .solver import (combine_multiscale_slopes, deposit, flow_accumulation,
                     shore_slope_field, solve_bed)

DATA = Path(__file__).with_name("data")
SPAN = GREENLAND_BBOX[2] - GREENLAND_BBOX[0]
SOURCE = "carve_v1"
LOD_SOURCE = "carve_v1_lod"
VERSION = 1


def _mosaic(connection, depth, rect):
    c0, c1, r0, r1 = rect
    step = GRID - 1
    shape = ((r1 - r0 + 1) * step + 1, (c1 - c0 + 1) * step + 1)
    z = np.full(shape, np.nan, dtype=np.float32)
    water = np.zeros(shape, dtype=bool)
    rows = region_rows(connection, depth, rect)
    for col in range(c0, c1 + 1):
        for row in range(r0, r1 + 1):
            blob, width, height, mask = rows[f"{depth}-{col}-{row}"]
            wet = decode_water(width, height, mask)
            if blob is None:
                if not wet.all():
                    raise ValueError("missing land DEM in bathymetry input snapshot")
                values = np.zeros((GRID, GRID), dtype=np.float32)
            else:
                values = np.frombuffer(zlib.decompress(blob), dtype=np.float32).reshape(GRID, GRID)
            x, y = (col - c0) * step, (r1 - row) * step
            z[y:y + GRID, x:x + GRID] = values[::-1]
            water[y:y + GRID, x:x + GRID] = wet[::-1]
    # Preserve the source model's exact retired synthetic-floor correction.
    z[np.isin(z, (-3.0, -5.0, -10.0))] = 0.0
    size = SPAN / (1 << depth)
    meta = {"x0": GREENLAND_BBOX[0] + c0 * size,
            "y1": GREENLAND_BBOX[1] + (r1 + 1) * size,
            "res": size / step}
    return z, water & np.isfinite(z), meta


def _restore_land(carved, original, width, height, mask, depth, coverage=None):
    wet = decode_water(width, height, mask)
    if original is None:
        if not wet.all():
            raise ValueError("cannot restore land without a DEM")
        orig = np.zeros((GRID, GRID), dtype=np.float32)
    else:
        orig = np.frombuffer(zlib.decompress(original), dtype=np.float32).reshape(GRID, GRID)
    result = np.where(wet, carved, orig).astype(np.float32)
    res = SPAN / (1 << depth) / (GRID - 1)
    source_res = max(SPAN / (1 << SOLVE_DEPTH) / (GRID - 1), 2 * res)
    repair_resampled_waterline(result, wet, res, source_res)
    limit_shoreline_drop(result, wet, res)
    pin_waterline(result, wet)
    if coverage is not None:
        result[wet & ~coverage] = np.nan
    return result, wet


def build(connection, tile_id):
    """Return all finest and derived rows without modifying the database."""
    missing = missing_inputs(connection, tile_id)
    if any(missing.values()):
        raise ValueError("bathymetry coverage incomplete: " + ", ".join(
            f"{name}={len(ids)}" for name, ids in missing.items()))
    regions = job_regions(tile_id)
    z, water, meta = _mosaic(connection, SOLVE_DEPTH, regions[SOLVE_DEPTH])
    if not water.any() or water.all():
        raise ValueError("bathymetry solve requires both coastline land and water")
    slope = shore_slope_field(z, water, ~water & (z > 0), meta["res"], min_land_elev=1.0)
    fine, fine_water, fine_meta = _mosaic(connection, SLOPE_DEPTH, regions[SLOPE_DEPTH])
    fine_slope = shore_slope_field(
        fine, fine_water, ~fine_water & (fine > 0), fine_meta["res"],
        curvature_projection_m=250.0, min_land_elev=1.0,
    )
    slope = combine_multiscale_slopes(slope, fine_slope, 1 << (SLOPE_DEPTH - SOLVE_DEPTH))
    flux, _ = flow_accumulation(z, water, meta["res"])
    bed, _, _ = solve_bed(z, water, ~water & (z > 0), meta["res"], slope,
                          14000.0, 0.02, 1.5, flux, 0.35, flux_hi=2.5)
    bed = deposit(bed, water, meta["res"], 0.7)
    if not np.isfinite(bed[water]).all():
        raise ValueError("regional bathymetry solve contains non-finite water depths")
    # The established pipeline hands a float32 regional raster to the carve.
    # Preserve that precision boundary even though no intermediate file is needed.
    bed = bed.astype(np.float32)

    z, water, local_meta = _mosaic(connection, CARVE_DEPTH, regions[CARVE_DEPTH])
    res = local_meta["res"]
    distance = ndimage.distance_transform_edt(water) * res
    neighborhood = max(3, int(round(8000.0 / res)) | 1)
    nearby_land = ndimage.uniform_filter((~water).astype(np.float64), size=neighborhood)
    coverage = water & ((nearby_land >= 0.12) | (distance <= 2000.0))
    sampled = sample_global(np.nan_to_num(bed), meta, z.shape,
                            local_meta["x0"], local_meta["y1"], res)
    repair_resampled_waterline(sampled, water, res, meta["res"])
    shore = pin_waterline(sampled, water)
    carved = add_detail(
        np.where(water, sampled, z), water, np.maximum(-sampled, 0.0), res,
        local_meta["x0"], local_meta["y1"],
        np.load(DATA / "detail_atlas.npy", allow_pickle=False),
        float(np.load(DATA / "detail_atlas_res.npy", allow_pickle=False)[0]),
        drape=0.006, lineation=0.006 * 0.55, gully=0.006 * 0.8,
    )
    limit_shoreline_drop(carved, water, res)
    carved[shore] = 0.0
    _, root_col, root_row = require_tile_id(tile_id)
    scale = 1 << (CARVE_DEPTH - JOB_DEPTH)
    c0, _, _, r1 = regions[CARVE_DEPTH]
    originals = region_rows(connection, CARVE_DEPTH, regions[CARVE_DEPTH])
    level, output = {}, []
    for col in range(root_col * scale, (root_col + 1) * scale):
        for row in range(root_row * scale, (root_row + 1) * scale):
            tid = f"{CARVE_DEPTH}-{col}-{row}"
            x, y = (col - c0) * (GRID - 1), (r1 - row) * (GRID - 1)
            grid = carved[y:y + GRID, x:x + GRID][::-1].astype(np.float32)
            covered = coverage[y:y + GRID, x:x + GRID][::-1]
            if not covered.any():
                continue
            grid, wet = _restore_land(grid, *originals[tid], CARVE_DEPTH, covered)
            if not np.isfinite(grid).any():
                continue
            level[col, row] = grid
            output.append((tid, grid, int(covered.sum()), SOURCE))
    for depth in range(CARVE_DEPTH - 1, JOB_DEPTH - 1, -1):
        originals = region_rows(connection, depth, regions[depth])
        # Shared vertices use every child's own coastline, including children
        # without carved water. This matches the source-grid masking order.
        child_rows = region_rows(connection, depth + 1, regions[depth + 1])
        wet_groups = parents_of({
            require_tile_id(tid)[1:]: decode_water(width, height, mask)
            for tid, (_, width, height, mask) in child_rows.items()
        })
        next_level = {}
        for (col, row), children in parents_of(level).items():
            tid = f"{depth}-{col}-{row}"
            grid = decimate(children, wet_groups[col, row])
            grid, wet = _restore_land(grid, *originals[tid], depth)
            water_px = int((wet & np.isfinite(grid)).sum())
            if not water_px:
                continue
            next_level[col, row] = grid
            output.append((tid, grid, water_px, LOD_SOURCE))
        level = next_level
    if not any(tid == tile_id for tid, *_ in output):
        raise ValueError(f"bathymetry produced no contract-depth water for {tile_id}")
    return output


def run(database_path: Path, tile_id: str) -> dict:
    job_regions(tile_id)
    with closing(sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)) as reader:
        reader.execute("BEGIN")
        if reader.execute("SELECT 1 FROM bathymetry WHERE tile_id=?", (tile_id,)).fetchone():
            return {"tileId": tile_id, "written": False, "rows": 0}
        rows = build(reader, tile_id)
    with closing(sqlite3.connect(database_path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)) as writer:
        writer.execute("PRAGMA foreign_keys=ON")
        with writer:
            writer.execute("BEGIN IMMEDIATE")
            if writer.execute("SELECT 1 FROM bathymetry WHERE tile_id=?", (tile_id,)).fetchone():
                return {"tileId": tile_id, "written": False, "rows": 0}
            for tid, grid, water_px, source in rows:
                write_bathymetry(writer, tid, grid, source=source, version=VERSION,
                                 water_px=water_px, commit=False)
    return {"tileId": tile_id, "written": True, "rows": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.db, args.tile)))
