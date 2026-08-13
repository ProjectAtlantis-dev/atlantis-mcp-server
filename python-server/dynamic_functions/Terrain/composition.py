"""Bounded, read-only composition for explicit terrain tile IDs."""

from __future__ import annotations

import base64
import hashlib
import sqlite3

import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.textures import read_texture_with_ancestor
from dynamic_functions.Terrain.Database.tiles import read_dem_with_ancestor
from dynamic_functions.Terrain.coastline import read_coastline_mask
from dynamic_functions.Terrain.effective_heightmap import (
    SHORELINE_SEAFLOOR_DROP_M,
    SOURCE as EFFECTIVE_SOURCE,
    VERSION as EFFECTIVE_VERSION,
    WATER_FLOOR_DROP_M,
    apply_water_mask,
)
from dynamic_functions.Terrain.hydrography import read_hydrography_mask
from dynamic_functions.Terrain.tidal_connectivity import (
    read_connectivity_snapshot,
)
from dynamic_functions.Terrain.tile_address import require_tile_id


MAX_COMPOSE_TILES = 256


def _validated_tile_ids(tile_ids: list[str]) -> tuple[str, ...]:
    if not isinstance(tile_ids, list):
        raise TypeError("tile_ids must be a list")
    if not tile_ids:
        raise ValueError("tile_ids must contain at least one tile ID")
    if len(tile_ids) > MAX_COMPOSE_TILES:
        raise ValueError(
            f"tile_ids contains {len(tile_ids)} entries; maximum is "
            f"{MAX_COMPOSE_TILES}"
        )
    result = []
    seen = set()
    for tile_id in tile_ids:
        depth, column, row = require_tile_id(tile_id)
        if tile_id != f"{depth}-{column}-{row}":
            raise ValueError(f"terrain tile id is not canonical: {tile_id!r}")
        if column >= 1 << depth or row >= 1 << depth:
            raise ValueError(
                f"terrain tile address is outside depth {depth}: {tile_id!r}"
            )
        if tile_id in seen:
            raise ValueError(f"duplicate terrain tile id: {tile_id}")
        seen.add(tile_id)
        result.append(tile_id)
    return tuple(result)


def _water_state(connection: sqlite3.Connection, tile_id: str) -> dict:
    """Return only source rows and an already-published connectivity mask."""

    coastline = read_coastline_mask(connection, tile_id)
    hydrography = read_hydrography_mask(connection, tile_id)
    connectivity = read_connectivity_snapshot(connection, tile_id)
    masks = []
    if coastline is not None:
        masks.append(coastline["mask"])
    if connectivity is not None:
        masks.append(connectivity["mask"])
    if masks and any(mask.shape != masks[0].shape for mask in masks[1:]):
        raise ValueError(f"ready water mask shape mismatch for {tile_id}")
    effective = None
    if masks:
        effective = masks[0].copy()
        for mask in masks[1:]:
            effective |= mask
    if connectivity is not None:
        connectivity_state = "ready"
    elif hydrography is not None:
        connectivity_state = "pending"
    else:
        connectivity_state = "missing"
    return {
        "mask": effective,
        "has_exact_coastline": coastline is not None,
        "status": {
            "coastline": "ready" if coastline is not None else "missing",
            "hydrography": "ready" if hydrography is not None else "missing",
            "tidalConnectivity": connectivity_state,
        },
    }


def _effective_from_ready(
    connection: sqlite3.Connection,
    tile_id: str,
    dem_payload: dict | None,
) -> tuple[dict | None, dict]:
    """Compose render heights without invoking cross-tile derivation."""

    water = _water_state(connection, tile_id)
    mask = water["mask"]
    raw = dem_payload["heightmap"] if dem_payload is not None else None
    mask_source = "ready_water_snapshot"
    if raw is None:
        if mask is None or not np.all(mask):
            return None, water["status"]
        heightmap = np.full(mask.shape, -WATER_FLOOR_DROP_M, dtype=np.float32)
    elif mask is None:
        mask = np.asarray(raw <= 0.0, dtype=bool)
        mask_source = "dem_nonpositive_fallback"
        heightmap = apply_water_mask(raw, mask) if np.any(mask) else raw.copy()
    else:
        heightmap = apply_water_mask(raw, mask)
        if water["has_exact_coastline"]:
            stale_water_on_land = ~mask & np.isfinite(raw) & (raw <= 0.0)
            heightmap[stale_water_on_land] = np.float32(0.0)
    if mask.shape != heightmap.shape:
        raise ValueError(
            f"ready water mask shape {mask.shape} does not match DEM "
            f"{heightmap.shape} for {tile_id}"
        )
    submerged = mask & np.isfinite(heightmap) & (heightmap <= 0.0)
    heightmap[submerged] -= np.float32(SHORELINE_SEAFLOOR_DROP_M)
    valid = heightmap[np.isfinite(heightmap)]
    payload = heightmap.astype(np.float32, copy=False).tobytes()
    return {
        "state": "ready",
        "tileId": tile_id,
        "source": EFFECTIVE_SOURCE,
        "version": EFFECTIVE_VERSION,
        "maskSource": mask_source,
        "verticalDatum": dem_payload["vertical_datum"] if dem_payload else None,
        "shape": [int(value) for value in heightmap.shape],
        "dtype": "float32",
        "minimum": float(np.min(valid)) if valid.size else None,
        "maximum": float(np.max(valid)) if valid.size else None,
        "nanCount": int(np.isnan(heightmap).sum()),
        "waterCount": int(mask.sum()),
        "digest": hashlib.sha256(payload).hexdigest(),
        "contentBase64": base64.b64encode(payload).decode("ascii"),
    }, water["status"]


def _compose_dem(connection: sqlite3.Connection, tile_id: str) -> dict:
    dem = read_dem_with_ancestor(connection, tile_id)
    resolved_id = dem["resolved_tile_id"] if dem is not None else tile_id
    effective, water = _effective_from_ready(connection, resolved_id, dem)
    if dem is None and effective is None:
        return {"state": "missing", "water": water, "heightmap": {"state": "missing"}}
    return {
        "state": "ready" if dem is not None else "missing",
        "exact": dem["exact"] if dem is not None else False,
        "resolvedTileId": resolved_id if dem is not None else None,
        "depthDelta": dem["depth_delta"] if dem is not None else None,
        "source": dem["source"] if dem is not None else None,
        "verticalDatum": dem["vertical_datum"] if dem is not None else None,
        "geometricError": dem["geometric_error"] if dem is not None else None,
        "water": water,
        "heightmap": effective or {"state": "missing"},
    }


def _compose_texture(connection: sqlite3.Connection, tile_id: str) -> dict:
    texture = read_texture_with_ancestor(connection, tile_id)
    if texture is None:
        return {"state": "missing"}
    payload = texture["texture"]
    return {
        "state": "ready",
        "exact": texture["exact"],
        "resolvedTileId": texture["resolved_tile_id"],
        "depthDelta": texture["depth_delta"],
        "source": texture["source"],
        "updatedAt": texture["updated_at"],
        "mediaType": "image/jpeg",
        "contentLength": len(payload),
        "digest": hashlib.sha256(payload).hexdigest(),
        "contentBase64": base64.b64encode(payload).decode("ascii"),
    }


def _domain_result(function, connection, tile_id: str) -> dict:
    try:
        return function(connection, tile_id)
    except Exception as exc:
        return {
            "state": "error",
            "errorType": type(exc).__name__,
            "error": str(exc),
        }


def compose_tiles_from_ready_data(
    connection: sqlite3.Connection,
    tile_ids: list[str],
) -> dict:
    """Compose independent local domains for a bounded explicit tile batch."""

    requested = _validated_tile_ids(tile_ids)
    tiles = []
    for tile_id in requested:
        tiles.append(
            {
                "tileId": tile_id,
                "dem": _domain_result(_compose_dem, connection, tile_id),
                "texture": _domain_result(_compose_texture, connection, tile_id),
            }
        )
    return {
        "tiles": tiles,
        "tileCount": len(tiles),
        "readOnly": True,
        "networkAccess": False,
        "scheduledWork": False,
    }


@visible
def compose_tiles(tile_ids: list[str]) -> dict:
    """Return every ready local domain for explicit tile IDs without demand."""

    return compose_tiles_from_ready_data(db(), tile_ids)
