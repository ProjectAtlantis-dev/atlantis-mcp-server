"""Rollback-only topology checks for same-depth tidal connectivity."""

from __future__ import annotations

import hashlib

import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import write_dem
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.hydrography import write_hydrography_mask
from dynamic_functions.Terrain.tidal_connectivity import (
    SEA_SEED_MAX_ELEV_M,
    _build_connected_hydrography,
    _response,
)


_DEPTH = 5
_COAST_SEED = "5-10-10"
_EDGE_CONNECTED = "5-11-10"
_NEIGHBOR_COAST = "5-12-10"
_COAST_ONLY = "5-13-10"
_WHOLE_TILE_LOW = "5-14-10"
_ONE_LOW_SAMPLE = "5-15-10"
_TILE_IDS = (
    _COAST_SEED,
    _EDGE_CONNECTED,
    _NEIGHBOR_COAST,
    _COAST_ONLY,
    _WHOLE_TILE_LOW,
    _ONE_LOW_SAMPLE,
)
_COMBINED_DIGEST = (
    "c55d7a51b88b997d5b1a3a553b738194a48f2a8d6630fc51f28fd2e431dba63d"
)


def _empty() -> np.ndarray:
    return np.zeros((65, 65), dtype=bool)


@visible
def tidal_connectivity_offline() -> dict:
    """Verify cross-tile flood topology and the whole-tile 0.5 m seed rule."""

    connection = db()
    connection.execute("SAVEPOINT tidal_connectivity_test")
    try:
        marks = ",".join("?" for _ in _TILE_IDS)
        connection.execute(
            f"DELETE FROM coastline_masks WHERE tile_id IN ({marks})",
            _TILE_IDS,
        )
        connection.execute(
            f"DELETE FROM hydrography_masks WHERE tile_id IN ({marks})",
            _TILE_IDS,
        )
        connection.execute(
            f"DELETE FROM tiles WHERE tile_id IN ({marks})",
            _TILE_IDS,
        )

        coast_seed = _empty()
        coast_seed[20, 30:] = True
        coast_seed[40, 10:13] = True  # disconnected inland component
        edge_connected = _empty()
        edge_connected[20, :31] = True
        edge_connected[45, 8:11] = True  # disconnected inland component
        neighbor_coast = _empty()
        neighbor_coast[30, 62:] = True
        whole_tile_low = _empty()
        whole_tile_low[10, 10] = True
        whole_tile_low[50, 50] = True
        one_low_sample = _empty()
        one_low_sample[12, 12] = True

        hydro_masks = {
            _COAST_SEED: coast_seed,
            _EDGE_CONNECTED: edge_connected,
            _NEIGHBOR_COAST: neighbor_coast,
            _WHOLE_TILE_LOW: whole_tile_low,
            _ONE_LOW_SAMPLE: one_low_sample,
        }
        for tile_id, mask in hydro_masks.items():
            write_hydrography_mask(
                connection,
                tile_id,
                mask,
                "fixture_hydrography",
                1,
                commit=False,
            )

        same_tile_coast = _empty()
        same_tile_coast[20, 30] = True
        write_coastline_mask(
            connection,
            _COAST_SEED,
            same_tile_coast,
            "fixture_coastline",
            1,
            commit=False,
        )
        adjacent_coast = _empty()
        adjacent_coast[30, 0] = True
        write_coastline_mask(
            connection,
            _COAST_ONLY,
            adjacent_coast,
            "fixture_coastline",
            1,
            commit=False,
        )

        write_dem(
            connection,
            _WHOLE_TILE_LOW,
            np.full((65, 65), SEA_SEED_MAX_ELEV_M, dtype=np.float32),
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        partly_elevated = np.full((65, 65), 10.0, dtype=np.float32)
        partly_elevated[12, 12] = 0.0
        write_dem(
            connection,
            _ONE_LOW_SAMPLE,
            partly_elevated,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )

        after_setup = connection.total_changes
        connected = _build_connected_hydrography(connection, _DEPTH)
        coast_response = _response(connection, _COAST_SEED)
        low_response = _response(connection, _WHOLE_TILE_LOW)
        reads_changed_no_rows = connection.total_changes == after_setup

        expected_coast_seed = _empty()
        expected_coast_seed[20, 30:] = True
        expected_edge_connected = _empty()
        expected_edge_connected[20, :31] = True
        expected_neighbor_coast = neighbor_coast.copy()
        expected_whole_low = whole_tile_low.copy()
        expected_one_low = _empty()
        expected = {
            _COAST_SEED: expected_coast_seed,
            _EDGE_CONNECTED: expected_edge_connected,
            _NEIGHBOR_COAST: expected_neighbor_coast,
            _WHOLE_TILE_LOW: expected_whole_low,
            _ONE_LOW_SAMPLE: expected_one_low,
        }
        exact = {
            tile_id: np.array_equal(connected.get(tile_id), wanted)
            for tile_id, wanted in expected.items()
        }
        combined = hashlib.sha256()
        for tile_id in sorted(expected):
            combined.update(tile_id.encode("ascii"))
            combined.update(connected[tile_id].astype(np.uint8).tobytes())
        combined_digest = combined.hexdigest()
        if combined_digest != _COMBINED_DIGEST:
            raise AssertionError(
                f"tidal-connectivity fixture digest changed: {combined_digest}"
            )

        return {
            "sameDepth": _DEPTH,
            "seaSeedMaxElevation": SEA_SEED_MAX_ELEV_M,
            "sameTileCoastSeeds": exact[_COAST_SEED],
            "sharedEdgePropagates": exact[_EDGE_CONNECTED],
            "neighborCoastSeeds": exact[_NEIGHBOR_COAST],
            "wholeLowTileSeedsAllComponents": exact[_WHOLE_TILE_LOW],
            "oneLowSampleDoesNotSeedTile": exact[_ONE_LOW_SAMPLE],
            "disconnectedInlandRejected": (
                not bool(connected[_COAST_SEED][40, 10:13].any())
                and not bool(connected[_EDGE_CONNECTED][45, 8:11].any())
            ),
            "coastResponseCounts": {
                "hydrography": coast_response["hydrographyCount"],
                "connected": coast_response["connectedCount"],
                "rejected": coast_response["rejectedCount"],
            },
            "lowTileResponseCounts": {
                "hydrography": low_response["hydrographyCount"],
                "connected": low_response["connectedCount"],
                "rejected": low_response["rejectedCount"],
            },
            "readOnlyDerivation": reads_changed_no_rows,
            "combinedDigest": combined_digest,
        }
    finally:
        connection.execute("ROLLBACK TO tidal_connectivity_test")
        connection.execute("RELEASE tidal_connectivity_test")
