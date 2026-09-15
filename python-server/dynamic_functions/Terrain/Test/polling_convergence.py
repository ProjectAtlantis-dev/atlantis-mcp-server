"""Rollback-only gate for polling, LOD convergence, and cache persistence."""

from __future__ import annotations

import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import write_dem
from dynamic_functions.Terrain.camera_lod import resolve_lod_coverage
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.demand import polling_state
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import ancestor_tile_ids, tile_bounds


_PARENT = "9-400-200"
_CHILDREN = (
    "10-800-400",
    "10-801-400",
    "10-800-401",
    "10-801-401",
)


def _selection() -> dict:
    tiles = []
    for tile_id in _CHILDREN:
        tiles.append(
            {
                "tileId": tile_id,
                "depth": 10,
                "bbox": [
                    float(value)
                    for value in tile_bounds(tile_id, GREENLAND_BBOX)
                ],
                "distance": 0.0,
            }
        )
    return {"tiles": tiles}


def _coarse_selection() -> dict:
    return {
        "requestedDepth": 10,
        "tiles": [{
            "tileId": _PARENT,
            "depth": 9,
            "bbox": [
                float(value)
                for value in tile_bounds(_PARENT, GREENLAND_BBOX)
            ],
            "distance": 0.0,
        }],
    }


def _cleanup(connection) -> None:
    fixture_ids: set[str] = set()
    for tile_id in (_PARENT, *_CHILDREN):
        fixture_ids.update(ancestor_tile_ids(tile_id, include_self=True))
    marks = ",".join("?" for _ in fixture_ids)
    args = tuple(fixture_ids)
    for table in (
        "tidal_connectivity_masks",
        "coastline_masks",
        "hydrography_masks",
        "textures",
        "tiles",
    ):
        connection.execute(
            f"DELETE FROM {table} WHERE tile_id IN ({marks})", args
        )


@visible
def polling_convergence_offline() -> dict:
    """Prove polling and monotonic convergence per current camera request."""

    active = polling_state(
        {
            "dem": {
                "claimedActive": ["a"],
                "pending": ["b"],
                "failures": {},
            }
        },
        now=100.0,
    )
    retry = polling_state(
        {
            "dem": {
                "claimedActive": [],
                "pending": [],
                "failures": {
                    "a": {
                        "claimed": True,
                        "retryable": True,
                        "retryAt": 110.0,
                    }
                },
            }
        },
        now=100.0,
    )
    terminal = polling_state(
        {
            "dem": {
                "claimedActive": [],
                "pending": [],
                "failures": {
                    "bad": {
                        "claimed": True,
                        "retryable": False,
                        "retryAt": None,
                    },
                    "stale": {
                        "claimed": False,
                        "retryable": True,
                        "retryAt": 101.0,
                    },
                },
            }
        },
        now=100.0,
    )

    connection = db()
    connection.execute("SAVEPOINT polling_convergence_test")
    try:
        _cleanup(connection)
        values = np.full((65, 65), 30.0, dtype=np.float32)
        write_dem(
            connection,
            _PARENT,
            values,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        land = np.zeros((65, 65), dtype=bool)
        write_coastline_mask(
            connection,
            _PARENT,
            land,
            "fixture_coastline",
            1,
            commit=False,
        )
        snapshots = [resolve_lod_coverage(connection, _selection())]
        coarse_snapshots = [
            resolve_lod_coverage(connection, _coarse_selection())
        ]
        for index, tile_id in enumerate(_CHILDREN):
            write_dem(
                connection,
                tile_id,
                values + np.float32(index + 1),
                "arcticdem_10m",
                "EGM2008",
                commit=False,
            )
            write_coastline_mask(
                connection,
                tile_id,
                land,
                "fixture_coastline",
                1,
                commit=False,
            )
            snapshots.append(resolve_lod_coverage(connection, _selection()))
            coarse_snapshots.append(
                resolve_lod_coverage(connection, _coarse_selection())
            )

        exact_counts = [item["exactTargetCount"] for item in snapshots]
        missing_counts = [item["missingTileCount"] for item in snapshots]
        stored_detail_count = connection.execute(
            "SELECT COUNT(*) FROM tiles WHERE tile_id IN (?,?,?,?) "
            "AND heightmap IS NOT NULL AND confidence_map IS NOT NULL",
            _CHILDREN,
        ).fetchone()[0]
        return {
            "activePollsSoon": bool(
                active["nextAction"] == "poll"
                and active["retryAfterMs"] == 1000
                and active["activeWork"] == 1
                and active["pendingWork"] == 1
            ),
            "futureRetryWaits": bool(
                retry["nextAction"] == "retry"
                and retry["retryAfterMs"] == 10000
                and retry["nextRetryAt"] == 110.0
            ),
            "terminalIsIdle": bool(
                terminal["nextAction"] == "idle"
                and terminal["shouldPoll"] is False
                and terminal["terminalFailures"] == 1
            ),
            "staleFailureIgnored": terminal["retryableFailures"] == 0,
            "exactMonotonic": exact_counts == [0, 1, 2, 3, 4],
            "missingMonotonic": missing_counts == [4, 3, 2, 1, 0],
            "fallbackAlwaysVisible": all(
                snapshot["coverageTileCount"] > 0 for snapshot in snapshots
            ),
            "currentRequestQualityNeverRegresses": bool(
                snapshots[0]["coverageTileIds"] == [_PARENT]
                and all(
                    set(snapshot["coverageTileIds"])
                    == {_PARENT, *_CHILDREN[:index]}
                    for index, snapshot in enumerate(snapshots[1:-1], 1)
                )
                and set(snapshots[-1]["coverageTileIds"]) == set(_CHILDREN)
            ),
            "currentLodControlsRendering": all(
                snapshot["coverageTileIds"] == [_PARENT]
                for snapshot in coarse_snapshots
            ),
            "detailedTilesRemainCached": stored_detail_count == len(_CHILDREN),
        }
    finally:
        connection.execute("ROLLBACK TO polling_convergence_test")
        connection.execute("RELEASE polling_convergence_test")
