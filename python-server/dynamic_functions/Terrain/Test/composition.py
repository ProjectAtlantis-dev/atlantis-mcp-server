"""Rollback-only gate for read-only explicit multi-tile composition."""

from __future__ import annotations

import datetime
from unittest.mock import patch

import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import ensure_tile_row, write_dem
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.composition import compose_tiles_from_ready_data
from dynamic_functions.Terrain.hydrography import write_hydrography_mask
from dynamic_functions.Terrain.tidal_connectivity import (
    write_connectivity_snapshot,
)
from dynamic_functions.Terrain.tile_address import ancestor_tile_ids


_FALLBACK = "10-803-403"
_DEM_PARENT = "8-200-100"
_TEXTURE_PARENT = "9-401-201"
_TEXTURE_ONLY = "10-804-404"
_DEM_ONLY = "10-805-405"
_PENDING_WATER = "10-806-406"
_READY_WATER = "10-807-407"
_CORRUPT_DEM = "10-808-408"
_MISSING = "10-900-900"
_TILE_IDS = (
    _FALLBACK,
    _DEM_PARENT,
    _TEXTURE_PARENT,
    _TEXTURE_ONLY,
    _DEM_ONLY,
    _PENDING_WATER,
    _READY_WATER,
    _CORRUPT_DEM,
    _MISSING,
)
_GRID = (65, 65)


def _cleanup(connection) -> None:
    tile_ids: set[str] = set(_TILE_IDS)
    for tile_id in _TILE_IDS:
        tile_ids.update(ancestor_tile_ids(tile_id, include_self=True))
    marks = ",".join("?" for _ in tile_ids)
    args = tuple(tile_ids)
    connection.execute(
        f"DELETE FROM tidal_connectivity_masks WHERE tile_id IN ({marks})",
        args,
    )
    connection.execute(
        f"DELETE FROM coastline_masks WHERE tile_id IN ({marks})",
        args,
    )
    connection.execute(
        f"DELETE FROM hydrography_masks WHERE tile_id IN ({marks})",
        args,
    )
    connection.execute(
        f"DELETE FROM textures WHERE tile_id IN ({marks})",
        args,
    )
    connection.execute(f"DELETE FROM tiles WHERE tile_id IN ({marks})", args)


def _texture(connection, tile_id: str, payload: bytes) -> None:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    connection.execute(
        "INSERT INTO textures (tile_id,source,texture,updated_at) "
        "VALUES (?, ?, ?, ?)",
        (tile_id, "fixture_texture", payload, now),
    )


@visible
def composition_offline() -> dict:
    """Prove fallback, water readiness, isolation, and zero side effects."""

    connection = db()
    connection.execute("SAVEPOINT composition_test")
    try:
        _cleanup(connection)
        base = np.full(_GRID, 100.0, dtype=np.float32)
        write_dem(
            connection,
            _DEM_PARENT,
            base,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        for tile_id in (_DEM_ONLY, _PENDING_WATER, _READY_WATER, _CORRUPT_DEM):
            write_dem(
                connection,
                tile_id,
                base + np.float32(int(tile_id.split("-")[1]) % 10),
                "arcticdem_10m",
                "EGM2008",
                commit=False,
            )
        for tile_id in (_TEXTURE_PARENT, _TEXTURE_ONLY, _CORRUPT_DEM):
            ensure_tile_row(connection, tile_id)
            _texture(connection, tile_id, f"texture:{tile_id}".encode())

        coast = np.zeros(_GRID, dtype=bool)
        coast[4, 4] = True
        hydro = np.zeros(_GRID, dtype=bool)
        hydro[4, 4:7] = True
        connected = np.zeros(_GRID, dtype=bool)
        connected[4, 4:6] = True
        for tile_id in (_PENDING_WATER, _READY_WATER):
            write_coastline_mask(
                connection, tile_id, coast, "fixture_coast", 1, commit=False
            )
            write_hydrography_mask(
                connection, tile_id, hydro, "fixture_hydro", 1, commit=False
            )
        write_connectivity_snapshot(
            connection, _READY_WATER, connected, commit=False
        )

        # A broken DEM row is a domain-local failure and must not suppress its
        # independently ready texture.
        connection.execute(
            "UPDATE tiles SET heightmap=? WHERE tile_id=?",
            (b"not-zlib", _CORRUPT_DEM),
        )
        before = connection.total_changes
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("composition attempted network access"),
        ):
            response = compose_tiles_from_ready_data(
                connection,
                [
                    _FALLBACK,
                    _TEXTURE_ONLY,
                    _DEM_ONLY,
                    _PENDING_WATER,
                    _READY_WATER,
                    _CORRUPT_DEM,
                    _MISSING,
                ],
            )
        read_only = connection.total_changes == before
        by_id = {tile["tileId"]: tile for tile in response["tiles"]}
        fallback = by_id[_FALLBACK]
        texture_only = by_id[_TEXTURE_ONLY]
        dem_only = by_id[_DEM_ONLY]
        pending = by_id[_PENDING_WATER]
        ready = by_id[_READY_WATER]
        corrupt = by_id[_CORRUPT_DEM]
        missing = by_id[_MISSING]
        return {
            "inputOrderPreserved": [tile["tileId"] for tile in response["tiles"]]
            == [
                _FALLBACK,
                _TEXTURE_ONLY,
                _DEM_ONLY,
                _PENDING_WATER,
                _READY_WATER,
                _CORRUPT_DEM,
                _MISSING,
            ],
            "demNearestAncestor": bool(
                fallback["dem"]["state"] == "ready"
                and fallback["dem"]["resolvedTileId"] == _DEM_PARENT
                and fallback["dem"]["depthDelta"] == 2
            ),
            "textureNearestAncestor": bool(
                fallback["texture"]["state"] == "ready"
                and fallback["texture"]["resolvedTileId"] == _TEXTURE_PARENT
                and fallback["texture"]["depthDelta"] == 1
            ),
            "textureSurvivesMissingDem": bool(
                texture_only["dem"]["state"] == "missing"
                and texture_only["texture"]["state"] == "ready"
            ),
            "demSurvivesMissingTexture": bool(
                dem_only["dem"]["state"] == "ready"
                and dem_only["dem"]["heightmap"]["state"] == "ready"
                and dem_only["texture"]["state"] == "missing"
            ),
            "missingConnectivityIsPending": bool(
                pending["dem"]["water"]["tidalConnectivity"] == "pending"
                and pending["dem"]["heightmap"]["state"] == "ready"
                and pending["dem"]["heightmap"]["waterCount"] == 1
            ),
            "readyConnectivityComposed": bool(
                ready["dem"]["water"]["tidalConnectivity"] == "ready"
                and ready["dem"]["heightmap"]["waterCount"] == 2
            ),
            "domainErrorIsolated": bool(
                corrupt["dem"]["state"] == "error"
                and corrupt["texture"]["state"] == "ready"
            ),
            "explicitMiss": bool(
                missing["dem"]["state"] == "missing"
                and missing["texture"]["state"] == "missing"
            ),
            "readOnlyComposition": read_only,
            "noNetworkOrScheduling": bool(
                response["networkAccess"] is False
                and response["scheduledWork"] is False
            ),
            "tileCount": response["tileCount"],
        }
    finally:
        connection.execute("ROLLBACK TO composition_test")
        connection.execute("RELEASE composition_test")
