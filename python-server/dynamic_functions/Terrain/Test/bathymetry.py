"""Rollback-only checks for stored and sampled underwater terrain."""

from __future__ import annotations

import numpy as np

from dynamic_functions.Terrain.Database.bathymetry import (
    complete_bathymetry_for_water,
    read_bathymetry,
    write_bathymetry,
)
from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import read_dem_payload, write_dem
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.effective_heightmap import (
    SHORELINE_SEAFLOOR_DROP_M,
    WATER_FLOOR_DROP_M,
    effective_heightmap_for_tile,
)


_SOURCE = "8-10-20"
_TARGET = "10-41-82"
_COARSE = "7-5-10"
_GRID = (65, 65)


@visible
def bathymetry_offline() -> dict:
    """Prove persistence, LOD sampling, water-only overlay, and fallback."""

    connection = db()
    connection.execute("SAVEPOINT bathymetry_test")
    try:
        tile_ids = (_SOURCE, _TARGET, _COARSE)
        marks = ",".join("?" for _ in tile_ids)
        connection.execute(
            f"DELETE FROM bathymetry WHERE tile_id IN ({marks})", tile_ids
        )
        connection.execute(
            f"DELETE FROM coastline_masks WHERE tile_id IN ({marks})", tile_ids
        )
        connection.execute(
            f"DELETE FROM tiles WHERE tile_id IN ({marks})", tile_ids
        )

        raw = np.full(_GRID, 100.0, dtype=np.float32)
        for tile_id in tile_ids:
            write_dem(
                connection,
                tile_id,
                raw,
                "arcticdem_10m",
                "EGM2008",
                commit=False,
            )

        water = np.ones(_GRID, dtype=bool)
        water[0, -1] = False
        write_coastline_mask(
            connection,
            _TARGET,
            water,
            "fixture_coastline",
            1,
            commit=False,
        )
        write_coastline_mask(
            connection,
            _COARSE,
            np.ones(_GRID, dtype=bool),
            "fixture_coastline",
            1,
            commit=False,
        )

        source_rows = np.arange(_GRID[0], dtype=np.float32)[:, None]
        source_columns = np.arange(_GRID[1], dtype=np.float32)[None, :]
        depths = -(source_rows * 10.0 + source_columns + 1.0)
        first_write = write_bathymetry(
            connection,
            _SOURCE,
            depths,
            source="fixture_bathymetry",
            version=1,
            commit=False,
        )
        duplicate_write = write_bathymetry(
            connection,
            _SOURCE,
            depths,
            source="fixture_bathymetry",
            version=1,
            commit=False,
        )

        sampled = read_bathymetry(connection, _TARGET, _GRID)
        if sampled is None:
            raise AssertionError("ancestor bathymetry was not sampled")
        expected_rows = np.linspace(32.0, 48.0, _GRID[0])[:, None]
        expected_columns = np.linspace(16.0, 32.0, _GRID[1])[None, :]
        expected = -(expected_rows * 10.0 + expected_columns + 1.0)

        before = connection.total_changes
        effective = effective_heightmap_for_tile(connection, _TARGET)
        read_only = connection.total_changes == before
        if effective is None:
            raise AssertionError("bathymetry-backed effective tile was missing")
        canonical = read_dem_payload(connection, _TARGET)
        if canonical is None:
            raise AssertionError("canonical DEM disappeared")

        # A coarse tile can assemble one covered child and must retain NaN in
        # every unsupported quadrant so composition leaves the -5 m floor.
        coarse_sample = read_bathymetry(connection, _COARSE, _GRID)
        if coarse_sample is None:
            raise AssertionError("descendant bathymetry was not assembled")
        coarse_effective = effective_heightmap_for_tile(connection, _COARSE)
        if coarse_effective is None:
            raise AssertionError("coarse effective tile was missing")

        completion_values = np.full((9, 9), 20.0, dtype=np.float32)
        completion_water = np.zeros((9, 9), dtype=bool)
        completion_water[1:4, 1:4] = True
        completion_water[5:8, 5:8] = True
        completion_values[2, 2] = -30.0
        completed = complete_bathymetry_for_water(
            completion_values,
            completion_water,
            cell_size_m=10.0,
        )

        heightmap = effective["heightmap"]
        coarse_heightmap = coarse_effective["heightmap"]
        return {
            "schemaPresent": {
                row[1]
                for row in connection.execute("PRAGMA table_info(bathymetry)")
            }
            == {
                "tile_id", "heightmap", "water_px", "min_z", "max_z",
                "source", "version", "updated_at",
            },
            "firstWrite": first_write,
            "duplicateWrite": duplicate_write,
            "ancestorResampled": bool(
                np.allclose(sampled, expected, atol=1e-5)
            ),
            "bathymetryApplied": bool(
                effective["bathymetry_found"]
                and effective["bathymetry_vertices"] > 0
                and np.allclose(
                    heightmap[16:, :32],
                    expected[16:, :32] - SHORELINE_SEAFLOOR_DROP_M,
                    atol=1e-5,
                )
            ),
            "landPreserved": bool(heightmap[0, -1] == raw[0, -1]),
            "canonicalDemPreserved": bool(
                np.array_equal(canonical["heightmap"], raw)
            ),
            "readOnlyComposition": read_only,
            "coarseCoverageUsesFallback": bool(
                np.any(coarse_heightmap < -WATER_FLOOR_DROP_M)
                and np.any(
                    coarse_heightmap
                    == -WATER_FLOOR_DROP_M - SHORELINE_SEAFLOOR_DROP_M
                )
            ),
            "completionDoesNotCrossLand": bool(
                np.all(completed[5:8, 5:8] == 20.0)
            ),
        }
    finally:
        connection.execute("ROLLBACK TO bathymetry_test")
        connection.execute("RELEASE bathymetry_test")
