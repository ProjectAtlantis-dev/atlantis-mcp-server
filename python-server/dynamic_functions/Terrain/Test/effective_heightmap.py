"""Rollback-only checks for coastline-aware effective heightmaps."""

from __future__ import annotations

import hashlib
import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import read_dem_payload, write_dem
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.effective_heightmap import (
    SHORELINE_SEAFLOOR_DROP_M,
    WATER_FLOOR_DROP_M,
    _response,
    effective_heightmap_for_tile,
)
from dynamic_functions.Terrain.hydrography import write_hydrography_mask


_DERIVED = "6-20-20"
_NO_MASK = "6-22-20"
_ALL_WATER_NO_DEM = "6-23-20"
_MIXED_NO_DEM = "6-24-20"
_SHAPE_MISMATCH = "6-25-20"
_TILE_IDS = (
    _DERIVED,
    _NO_MASK,
    _ALL_WATER_NO_DEM,
    _MIXED_NO_DEM,
    _SHAPE_MISMATCH,
)
_GRID = (65, 65)


def _empty(shape: tuple[int, int] = _GRID) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


@visible
def effective_heightmap_offline() -> dict:
    """Verify water composition, masking, synthesis, and source immutability."""

    connection = db()
    connection.execute("SAVEPOINT effective_heightmap_test")
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

        raw = np.full(_GRID, 120.0, dtype=np.float32)
        raw[5, 5] = -20.0
        raw[10, 10] = -3.0
        raw[30, 30] = np.nan
        write_dem(
            connection,
            _DERIVED,
            raw,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        coast = _empty()
        coast[10, 10] = True
        write_coastline_mask(
            connection,
            _DERIVED,
            coast,
            "fixture_coastline",
            1,
            commit=False,
        )
        hydro = _empty()
        hydro[10, 10:13] = True
        hydro[40, 40] = True
        write_hydrography_mask(
            connection,
            _DERIVED,
            hydro,
            "fixture_hydrography",
            1,
            commit=False,
        )

        before = connection.total_changes
        derived = effective_heightmap_for_tile(connection, _DERIVED)
        response = _response(connection, _DERIVED)
        read_only = connection.total_changes == before
        if derived is None:
            raise AssertionError("fixture effective heightmap was not derived")
        effective = derived["heightmap"]
        water = derived["water_mask"]
        expected_water = _empty()
        expected_water[10, 10:13] = True
        expected_floor = np.float32(
            -WATER_FLOOR_DROP_M - SHORELINE_SEAFLOOR_DROP_M
        )
        stored = read_dem_payload(connection, _DERIVED)
        if stored is None:
            raise AssertionError("fixture canonical DEM disappeared")

        fallback_raw = np.full(_GRID, 9.0, dtype=np.float32)
        fallback_raw[1, 1] = -2.0
        fallback_raw[2, 2] = 0.0
        write_dem(
            connection,
            _NO_MASK,
            fallback_raw,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        rejected_hydro = _empty()
        rejected_hydro[50, 50] = True
        write_hydrography_mask(
            connection,
            _NO_MASK,
            rejected_hydro,
            "fixture_hydrography",
            1,
            commit=False,
        )
        fallback = effective_heightmap_for_tile(connection, _NO_MASK)
        if fallback is None:
            raise AssertionError("DEM fallback heightmap was not derived")

        all_water = np.ones(_GRID, dtype=bool)
        write_coastline_mask(
            connection,
            _ALL_WATER_NO_DEM,
            all_water,
            "fixture_coastline",
            1,
            commit=False,
        )
        synthesized = effective_heightmap_for_tile(
            connection,
            _ALL_WATER_NO_DEM,
        )
        mixed = _empty()
        mixed[0, 0] = True
        write_coastline_mask(
            connection,
            _MIXED_NO_DEM,
            mixed,
            "fixture_coastline",
            1,
            commit=False,
        )
        mixed_missing = effective_heightmap_for_tile(
            connection,
            _MIXED_NO_DEM,
        )

        mismatch_raw = np.ones(_GRID, dtype=np.float32)
        write_dem(
            connection,
            _SHAPE_MISMATCH,
            mismatch_raw,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        write_coastline_mask(
            connection,
            _SHAPE_MISMATCH,
            np.ones((3, 3), dtype=bool),
            "fixture_coastline",
            1,
            commit=False,
        )
        mismatch_rejected = False
        try:
            effective_heightmap_for_tile(connection, _SHAPE_MISMATCH)
        except ValueError:
            mismatch_rejected = True

        return {
            "coastAndConnectedHydroUnion": bool(
                np.array_equal(water, expected_water)
            ),
            "disconnectedInlandRejected": not bool(water[40, 40]),
            "waterFloorApplied": bool(
                np.all(effective[expected_water] == expected_floor)
            ),
            "staleWaterOnLandClipped": bool(effective[5, 5] == 0.0),
            "measuredLandPreserved": bool(effective[0, 0] == 120.0),
            "canonicalDemPreserved": bool(
                np.array_equal(raw, stored["heightmap"], equal_nan=True)
            ),
            "readOnlyDerivation": read_only,
            "responseMatchesArray": response["digest"]
            == hashlib.sha256(effective.tobytes()).hexdigest(),
            "noMaskUsesDemFallback": bool(
                fallback["mask_source"] == "dem_nonpositive_fallback"
                and fallback["water_mask"].sum() == 2
                and fallback["heightmap"][1, 1] == expected_floor
                and fallback["heightmap"][2, 2] == expected_floor
                and fallback["heightmap"][0, 0] == 9.0
            ),
            "allWaterWithoutDemSynthesized": bool(
                synthesized is not None
                and not synthesized["canonical_dem_found"]
                and np.all(synthesized["heightmap"] == expected_floor)
            ),
            "mixedWaterWithoutDemRejected": mixed_missing is None,
            "shapeMismatchRejected": mismatch_rejected,
            "waterCount": response["waterCount"],
            "digest": response["digest"],
        }
    finally:
        connection.execute("ROLLBACK TO effective_heightmap_test")
        connection.execute("RELEASE effective_heightmap_test")
