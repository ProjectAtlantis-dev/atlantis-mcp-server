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
    write_dem,
)


@visible
async def list_tiles() -> list[tuple]:
    """Return the tile ID, depth, and source for every terrain tile row."""

    return await atlantis.client_command("@Database/query 'SELECT tile_id, depth, source FROM tiles'");


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
def fetch_dem(tile_id: str) -> dict:
    """Fetch one live ArcticDEM tile and persist it through the shared DB."""

    # Acquisition finishes before the database is touched. Provider failures
    # therefore cannot create or alter a terrain row.
    heightmap, sources = _fetch_heightmap(tile_id)
    connection = db()
    written = write_dem(
        connection,
        tile_id,
        heightmap,
        "arcticdem_10m",
    )
    return {
        "provider": "arcticdem",
        "dataset": "mosaics/v4.1/10m",
        "written": written,
        "sources": sources,
        **_read_dem(connection, tile_id),
    }
