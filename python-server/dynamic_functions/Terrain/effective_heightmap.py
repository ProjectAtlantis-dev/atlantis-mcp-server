"""Read-only coastline-aware render heightmap derivation."""

from __future__ import annotations

import hashlib
import sqlite3

import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import read_dem_payload
from dynamic_functions.Terrain.coastline import read_coastline_mask
from dynamic_functions.Terrain.tidal_connectivity import (
    connected_hydrography_for_tile,
)
from dynamic_functions.Terrain.tile_address import require_tile_id


SOURCE = "derived_effective_heightmap"
VERSION = 1
WATER_FLOOR_DROP_M = 5.0
SHORELINE_SEAFLOOR_DROP_M = 1.0


def effective_water_mask(
    connection: sqlite3.Connection,
    tile_id: str,
) -> tuple[np.ndarray | None, bool]:
    """Combine exact GTK50 sea with connected WMS water for one tile.

    The boolean result records whether an exact authoritative coastline row
    exists. It controls the stale-water land clip independently of whether
    the final mask also contains connected hydrography.
    """

    require_tile_id(tile_id)
    coastline = read_coastline_mask(connection, tile_id)
    authoritative = coastline["mask"] if coastline is not None else None
    connected = connected_hydrography_for_tile(connection, tile_id)
    if connected is None or not np.any(connected):
        return (
            authoritative.copy() if authoritative is not None else None,
            authoritative is not None,
        )
    if authoritative is None:
        return connected, False
    if authoritative.shape != connected.shape:
        raise ValueError(
            f"coastline/hydrography mask shape mismatch for {tile_id}: "
            f"{authoritative.shape} vs {connected.shape}"
        )
    return authoritative | connected, True


def apply_water_mask(heightmap: np.ndarray, water: np.ndarray) -> np.ndarray:
    """Return a fallback seabed without modifying the input heightmap."""

    result = np.asarray(heightmap, dtype=np.float32).copy()
    mask = np.asarray(water, dtype=bool)
    if result.ndim != 2:
        raise ValueError(f"heightmap must be 2-D, got {result.ndim}-D")
    if mask.shape != result.shape:
        raise ValueError(
            f"water mask shape {mask.shape} does not match DEM {result.shape}"
        )
    result[mask] = np.float32(-WATER_FLOOR_DROP_M)
    return result


def effective_heightmap_for_tile(
    connection: sqlite3.Connection,
    tile_id: str,
) -> dict | None:
    """Derive render geometry from current source rows without writing them."""

    require_tile_id(tile_id)
    water, has_exact_coastline = effective_water_mask(connection, tile_id)
    payload = read_dem_payload(connection, tile_id)
    raw = payload["heightmap"] if payload is not None else None
    mask_source = "coastline_connectivity"

    if raw is None:
        if water is None or not np.all(water):
            return None
        result = np.full(
            water.shape,
            -WATER_FLOOR_DROP_M,
            dtype=np.float32,
        )
    elif water is None:
        # Preserve Flask's stored-DEM fallback where no vector coverage is
        # available: nonpositive canonical samples follow the water plate.
        water = np.asarray(raw <= 0.0, dtype=bool)
        mask_source = "dem_nonpositive_fallback"
        if not np.any(water):
            result = np.asarray(raw, dtype=np.float32).copy()
        else:
            result = apply_water_mask(raw, water)
    else:
        result = apply_water_mask(raw, water)
        if has_exact_coastline:
            stale_water_on_land = (
                ~water & np.isfinite(raw) & (raw <= 0.0)
            )
            result[stale_water_on_land] = np.float32(0.0)

    if water is None:
        raise AssertionError("effective water mask was not resolved")
    if result.shape != water.shape:
        raise ValueError(
            f"effective water mask shape {water.shape} does not match "
            f"heightmap {result.shape} for {tile_id}"
        )
    submerged = water & np.isfinite(result) & (result <= 0.0)
    result[submerged] -= np.float32(SHORELINE_SEAFLOOR_DROP_M)
    return {
        "tile_id": tile_id,
        "heightmap": result,
        "water_mask": water,
        "mask_source": mask_source,
        "has_exact_coastline": has_exact_coastline,
        "canonical_dem_found": payload is not None,
        "vertical_datum": payload["vertical_datum"] if payload else None,
    }


def _response(connection: sqlite3.Connection, tile_id: str) -> dict:
    require_tile_id(tile_id)
    payload = effective_heightmap_for_tile(connection, tile_id)
    if payload is None:
        return {"tileId": tile_id, "found": False}
    heightmap = payload["heightmap"]
    water = payload["water_mask"]
    valid = heightmap[np.isfinite(heightmap)]
    values = water.astype(np.uint8)
    return {
        "tileId": tile_id,
        "found": True,
        "source": SOURCE,
        "version": VERSION,
        "maskSource": payload["mask_source"],
        "hasExactCoastline": payload["has_exact_coastline"],
        "canonicalDemFound": payload["canonical_dem_found"],
        "verticalDatum": payload["vertical_datum"],
        "waterFloorDropMeters": WATER_FLOOR_DROP_M,
        "shorelineSeafloorDropMeters": SHORELINE_SEAFLOOR_DROP_M,
        "shape": [int(heightmap.shape[0]), int(heightmap.shape[1])],
        "dtype": str(heightmap.dtype),
        "waterCount": int(water.sum()),
        "landCount": int(water.size - water.sum()),
        "minimum": float(np.min(valid)) if valid.size else None,
        "maximum": float(np.max(valid)) if valid.size else None,
        "nanCount": int(np.isnan(heightmap).sum()),
        "waterDigest": hashlib.sha256(values.tobytes()).hexdigest(),
        "digest": hashlib.sha256(heightmap.tobytes()).hexdigest(),
    }


@visible
def derive_effective_heightmap(tile_id: str) -> dict:
    """Derive coastline-aware render heights without changing stored data."""

    return _response(db(), tile_id)
