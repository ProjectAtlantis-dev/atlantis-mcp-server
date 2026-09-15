"""Independent underwater terrain storage and LOD sampling.

Bathymetry uses the same south-first, shared-edge raster convention as DEM
tiles but never replaces the canonical DEM. Stored depth-8 rasters are cropped
and bilinearly sampled into detailed descendants; coarse requests can assemble
their nearest available descendant depth while leaving uncovered samples NaN.
"""

from __future__ import annotations

import datetime
import sqlite3
import zlib

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from dynamic_functions.Terrain.Database.tiles import ensure_tile_row
from dynamic_functions.Terrain.tile_address import format_tile_id, require_tile_id


BATHYMETRY_CONTRACT_DEPTH = 8
MAX_SHORE_SLOPE = 1.5


def write_bathymetry(
    connection: sqlite3.Connection,
    tile_id: str,
    heightmap,
    *,
    source: str,
    version: int,
    water_px: int | None = None,
    commit: bool = True,
) -> bool:
    """Publish one validated south-first float32 bathymetry raster."""

    require_tile_id(tile_id)
    values = np.asarray(heightmap, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("bathymetry heightmap must be a 2D array at least 2x2")
    if values.shape[0] != values.shape[1]:
        raise ValueError("bathymetry heightmap must be square")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("bathymetry source must be non-empty")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("bathymetry heightmap must contain a finite sample")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("bathymetry version must be a non-negative integer")
    if water_px is None:
        water_px = int(np.sum(np.isfinite(values) & (values < 0.0)))

    ensure_tile_row(connection, tile_id)
    encoded = zlib.compress(values.tobytes(), level=6)
    existing = connection.execute(
        "SELECT heightmap,water_px,min_z,max_z,source,version "
        "FROM bathymetry WHERE tile_id=?",
        (tile_id,),
    ).fetchone()
    identity = (
        encoded,
        int(water_px),
        float(np.min(finite)),
        float(np.max(finite)),
        source.strip(),
        version,
    )
    if existing is not None and tuple(existing) == identity:
        return False

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    connection.execute(
        "INSERT INTO bathymetry "
        "(tile_id,heightmap,water_px,min_z,max_z,source,version,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(tile_id) DO UPDATE SET "
        "heightmap=excluded.heightmap,water_px=excluded.water_px,"
        "min_z=excluded.min_z,max_z=excluded.max_z,source=excluded.source,"
        "version=excluded.version,updated_at=excluded.updated_at",
        (tile_id, *identity, now),
    )
    if commit:
        connection.commit()
    return True


def complete_bathymetry_for_water(
    values,
    water,
    *,
    cell_size_m: float,
    max_shore_slope: float = MAX_SHORE_SLOPE,
) -> np.ndarray:
    """Complete coarse submarine evidence over a finer connected water mask."""

    result = np.asarray(values, dtype=np.float32).copy()
    water = np.asarray(water, dtype=bool)
    if result.shape != water.shape:
        raise ValueError(
            f"bathymetry shape {result.shape} does not match water mask "
            f"{water.shape}"
        )
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive and finite")
    if not np.isfinite(max_shore_slope) or max_shore_slope <= 0.0:
        raise ValueError("max_shore_slope must be positive and finite")

    submarine = water & np.isfinite(result) & (result < 0.0)
    if not np.any(submarine):
        return result

    labels, component_count = label(
        water, structure=np.ones((3, 3), dtype=np.uint8)
    )
    supported_water = np.zeros_like(water)
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        evidence = component & submarine
        if not np.any(evidence):
            continue
        supported_water |= component
        incomplete = component & np.isfinite(result) & ~submarine
        if np.any(incomplete):
            _, nearest = distance_transform_edt(
                ~evidence, return_distances=True, return_indices=True
            )
            result[incomplete] = result[
                nearest[0][incomplete], nearest[1][incomplete]
            ]

    shore = supported_water & binary_dilation(
        ~water, structure=np.ones((3, 3), dtype=bool)
    )
    if np.any(shore):
        result[shore] = 0.0
        distance_m = distance_transform_edt(~shore) * float(cell_size_m)
        shallow_limit = -float(max_shore_slope) * distance_m
        result[supported_water] = np.maximum(
            result[supported_water], shallow_limit[supported_water]
        )
        result[shore] = 0.0
    return result


def _decode_row(tile_id: str, blob) -> np.ndarray:
    values = np.frombuffer(zlib.decompress(blob), dtype=np.float32)
    resolution = int(np.sqrt(values.size))
    if resolution < 2 or resolution * resolution != values.size:
        raise ValueError(
            f"bathymetry heightmap for {tile_id} has {values.size} values; "
            "expected a square float32 raster"
        )
    return values.reshape((resolution, resolution)).copy()


def _sample_point(values: np.ndarray, row: float, column: float) -> np.float32:
    """Strict bilinear sample: any contributing no-data corner is no data."""

    row = float(np.clip(row, 0.0, values.shape[0] - 1))
    column = float(np.clip(column, 0.0, values.shape[1] - 1))
    r0, c0 = int(np.floor(row)), int(np.floor(column))
    r1 = min(r0 + 1, values.shape[0] - 1)
    c1 = min(c0 + 1, values.shape[1] - 1)
    rf, cf = row - r0, column - c0
    samples = np.asarray(
        [values[r0, c0], values[r0, c1], values[r1, c0], values[r1, c1]],
        dtype=np.float64,
    )
    weights = np.asarray(
        [
            (1.0 - rf) * (1.0 - cf),
            (1.0 - rf) * cf,
            rf * (1.0 - cf),
            rf * cf,
        ],
        dtype=np.float64,
    )
    contributing = weights > 1e-12
    if np.any(contributing & ~np.isfinite(samples)):
        return np.float32(np.nan)
    return np.float32(np.sum(np.where(contributing, samples * weights, 0.0)))


def _resample_ancestor(
    values: np.ndarray,
    source_address: tuple[int, int, int],
    target_address: tuple[int, int, int],
    output_shape: tuple[int, int],
) -> np.ndarray:
    source_depth, source_col, source_row = source_address
    target_depth, target_col, target_row = target_address
    levels = target_depth - source_depth
    if levels < 0:
        raise ValueError("target must be a descendant of the bathymetry source")
    scale = 1 << levels
    offset_col = target_col - source_col * scale
    offset_row = target_row - source_row * scale
    if not (0 <= offset_col < scale and 0 <= offset_row < scale):
        raise ValueError("target is outside the bathymetry source tile")

    rows = np.linspace(
        offset_row * (values.shape[0] - 1) / scale,
        (offset_row + 1) * (values.shape[0] - 1) / scale,
        output_shape[0],
    )
    columns = np.linspace(
        offset_col * (values.shape[1] - 1) / scale,
        (offset_col + 1) * (values.shape[1] - 1) / scale,
        output_shape[1],
    )
    r0 = np.floor(rows).astype(np.intp)
    c0 = np.floor(columns).astype(np.intp)
    r1 = np.minimum(r0 + 1, values.shape[0] - 1)
    c1 = np.minimum(c0 + 1, values.shape[1] - 1)
    rf = (rows - r0)[:, None]
    cf = (columns - c0)[None, :]
    samples = (
        values[np.ix_(r0, c0)],
        values[np.ix_(r0, c1)],
        values[np.ix_(r1, c0)],
        values[np.ix_(r1, c1)],
    )
    weights = (
        (1.0 - rf) * (1.0 - cf),
        (1.0 - rf) * cf,
        rf * (1.0 - cf),
        rf * cf,
    )
    output = np.zeros(output_shape, dtype=np.float64)
    invalid = np.zeros(output_shape, dtype=bool)
    for sample, weight in zip(samples, weights):
        finite = np.isfinite(sample)
        contributing = weight > 1e-12
        invalid |= contributing & ~finite
        output += np.where(finite, sample, 0.0) * weight
    output[invalid] = np.nan
    return output.astype(np.float32)


def _axis_candidates(value: float, lower: int, upper: int):
    nearest = int(round(value))
    if abs(value - nearest) <= 1e-10:
        candidates = []
        if lower <= nearest < upper:
            candidates.append((nearest, 0.0))
        if lower <= nearest - 1 < upper:
            candidates.append((nearest - 1, 1.0))
        return candidates
    index = int(np.floor(value))
    return [(index, value - index)] if lower <= index < upper else []


def _resample_descendants(
    rasters: dict[tuple[int, int], np.ndarray],
    target_address: tuple[int, int, int],
    source_depth: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    target_depth, target_col, target_row = target_address
    scale = 1 << (source_depth - target_depth)
    lower_col, lower_row = target_col * scale, target_row * scale
    upper_col, upper_row = (target_col + 1) * scale, (target_row + 1) * scale
    output = np.full(output_shape, np.nan, dtype=np.float32)
    source_rows = np.linspace(lower_row, upper_row, output_shape[0])
    source_columns = np.linspace(lower_col, upper_col, output_shape[1])

    for output_row, global_y in enumerate(source_rows):
        row_candidates = _axis_candidates(global_y, lower_row, upper_row)
        for output_column, global_x in enumerate(source_columns):
            column_candidates = _axis_candidates(global_x, lower_col, upper_col)
            for source_row, local_y in row_candidates:
                for source_col, local_x in column_candidates:
                    raster = rasters.get((source_col, source_row))
                    if raster is None:
                        continue
                    value = _sample_point(
                        raster,
                        local_y * (raster.shape[0] - 1),
                        local_x * (raster.shape[1] - 1),
                    )
                    if np.isfinite(value):
                        output[output_row, output_column] = value
                        break
                if np.isfinite(output[output_row, output_column]):
                    break
    return output


def read_bathymetry(
    connection: sqlite3.Connection,
    tile_id: str,
    output_shape: tuple[int, int],
) -> np.ndarray | None:
    """Return bathymetry sampled into ``tile_id`` or None if uncovered."""

    target = require_tile_id(tile_id)
    target_depth, target_col, target_row = target
    if (
        len(output_shape) != 2
        or min(output_shape) < 2
        or not all(isinstance(value, int) for value in output_shape)
    ):
        raise ValueError("bathymetry output_shape must be two integers >= 2")

    ancestor_ids = []
    for source_depth in range(target_depth, -1, -1):
        shift = target_depth - source_depth
        ancestor_ids.append(
            format_tile_id(
                source_depth,
                target_col >> shift,
                target_row >> shift,
            )
        )
    placeholders = ",".join("?" for _ in ancestor_ids)
    ancestor_rows = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT tile_id,heightmap FROM bathymetry "
            f"WHERE tile_id IN ({placeholders})",
            ancestor_ids,
        )
    }
    for source_id in ancestor_ids:
        blob = ancestor_rows.get(source_id)
        if blob is None:
            continue
        return _resample_ancestor(
            _decode_row(source_id, blob),
            require_tile_id(source_id),
            target,
            output_shape,
        )

    if target_depth >= BATHYMETRY_CONTRACT_DEPTH:
        return None

    maximum = connection.execute(
        "SELECT MAX(t.depth) FROM bathymetry b "
        "JOIN tiles t ON t.tile_id=b.tile_id"
    ).fetchone()
    maximum_depth = (
        int(maximum[0]) if maximum and maximum[0] is not None else target_depth
    )
    for source_depth in range(target_depth + 1, maximum_depth + 1):
        scale = 1 << (source_depth - target_depth)
        rows = connection.execute(
            "SELECT t.col,t.row,b.tile_id,b.heightmap "
            "FROM bathymetry b JOIN tiles t ON t.tile_id=b.tile_id "
            "WHERE t.depth=? AND t.col BETWEEN ? AND ? "
            "AND t.row BETWEEN ? AND ?",
            (
                source_depth,
                target_col * scale,
                (target_col + 1) * scale - 1,
                target_row * scale,
                (target_row + 1) * scale - 1,
            ),
        ).fetchall()
        if not rows:
            continue
        rasters = {
            (int(column), int(row)): _decode_row(source_id, blob)
            for column, row, source_id, blob in rows
        }
        return _resample_descendants(
            rasters, target, source_depth, output_shape
        )
    return None
