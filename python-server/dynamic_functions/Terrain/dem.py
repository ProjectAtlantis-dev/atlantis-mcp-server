"""Directly callable access to persisted terrain DEMs."""

import atlantis
import hashlib

import numpy as np

from dynamic_functions.Terrain.arctic_dem import (
    _fetch_heightmap,
    _heightmap_summary,
)
from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import (
    read_dem_payload,
    read_dem_with_ancestor,
    write_dem,
)


@visible
async def list_tiles() -> list[tuple]:
    """Return address, source, datum, parent, and update time for every tile."""

    return await atlantis.client_command(
        "@Database/query 'SELECT tile_id, depth, source, vertical_datum,col, row, parent_id, updated_at FROM tiles'"
    )


def _read_dem(connection, tile_id: str) -> dict:
    """Build the JSON-safe response for one stored DEM."""

    payload = read_dem_payload(connection, tile_id)
    if payload is None:
        return {"tileId": tile_id, "found": False}

    confidence_map = payload["confidence_map"]
    return {
        "tileId": tile_id,
        "found": True,
        "source": payload["source"],
        "verticalDatum": payload["vertical_datum"],
        "updatedAt": payload["updated_at"],
        "geometricError": payload["geometric_error"],
        "confidenceLevels": [
            int(value) for value in np.unique(confidence_map)
        ],
        "confidenceDigest": hashlib.sha256(
            confidence_map.tobytes()
        ).hexdigest(),
        **_heightmap_summary(payload["heightmap"]),
    }


@visible
def read_dem(tile_id: str) -> dict:
    """Return decoded metadata for one stored DEM tile."""

    return _read_dem(db(), tile_id)


@visible
def read_dem_fallback(tile_id: str) -> dict:
    """Return exact DEM metadata or the nearest stored ancestor explicitly."""

    payload = read_dem_with_ancestor(db(), tile_id)
    if payload is None:
        return {"tileId": tile_id, "found": False}
    return {
        "tileId": tile_id,
        "found": True,
        "exact": payload["exact"],
        "resolvedTileId": payload["resolved_tile_id"],
        "depthDelta": payload["depth_delta"],
        "source": payload["source"],
        "verticalDatum": payload["vertical_datum"],
        "updatedAt": payload["updated_at"],
        "geometricError": payload["geometric_error"],
        "confidenceLevels": [
            int(value) for value in np.unique(payload["confidence_map"])
        ],
        "confidenceDigest": hashlib.sha256(
            payload["confidence_map"].tobytes()
        ).hexdigest(),
        **_heightmap_summary(payload["heightmap"]),
    }


@visible
def fetch_dem(tile_id: str) -> dict:
    """Fetch one live ArcticDEM tile and persist it through the shared DB."""

    # Acquisition finishes before the database is touched. Provider failures
    # therefore cannot create or alter a terrain row.
    heightmap, sources, geoid_undulation = _fetch_heightmap(tile_id)
    connection = db()
    written = write_dem(
        connection,
        tile_id,
        heightmap,
        "arcticdem_10m",
        "EGM2008",
    )
    return {
        "provider": "arcticdem",
        "dataset": "mosaics/v4.1/10m",
        "verticalDatum": "EGM2008",
        "geoidUndulation": geoid_undulation,
        "written": written,
        "sources": sources,
        **_read_dem(connection, tile_id),
    }
