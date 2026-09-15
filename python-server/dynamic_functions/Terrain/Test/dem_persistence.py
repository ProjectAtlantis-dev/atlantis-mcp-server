"""Directly callable, rollback-only DEM persistence check."""

from pathlib import Path

import numpy as np

from dynamic_functions.Terrain.arctic_dem import (
    _correct_vertical_datum,
    _decode_source,
    _heightmap_summary,
)
from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import (
    TileClobberError,
    read_dem_payload,
    write_dem,
)
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import tile_bounds


_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "arctic_dem.tif"


@visible
def dem_persistence(tile_id: str) -> dict:
    """Verify round-trip, idempotence, and no-clobber using one savepoint."""

    connection = db()
    bbox = tile_bounds(tile_id, GREENLAND_BBOX)
    heightmap, geoid_undulation = _correct_vertical_datum(
        _decode_source(_FIXTURE_PATH, bbox),
        bbox,
    )
    connection.execute("SAVEPOINT dem_persistence_test")
    try:
        # The live MVP gate may already have persisted this tile. Remove it
        # only inside the savepoint so the offline check is deterministic and
        # the caller's row is restored by the final rollback.
        connection.execute("DELETE FROM tiles WHERE tile_id = ?", (tile_id,))
        first_write = write_dem(
            connection,
            tile_id,
            heightmap,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        duplicate_write = write_dem(
            connection,
            tile_id,
            heightmap,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        datum_clobber_blocked = False
        try:
            write_dem(
                connection,
                tile_id,
                heightmap,
                "arcticdem_10m",
                "WGS84_ELLIPSOIDAL",
                commit=False,
            )
        except TileClobberError:
            datum_clobber_blocked = True
        stored = read_dem_payload(connection, tile_id)
        if stored is None:
            raise AssertionError("stored DEM could not be read back")
        exact_round_trip = bool(
            np.array_equal(heightmap, stored["heightmap"], equal_nan=True)
        )

        changed = heightmap.copy()
        changed[0, 0] = np.float32(12345.0)
        clobber_blocked = False
        try:
            write_dem(
                connection,
                tile_id,
                changed,
                "arcticdem_10m",
                "EGM2008",
                commit=False,
            )
        except TileClobberError:
            clobber_blocked = True

        unchanged = read_dem_payload(connection, tile_id)
        if unchanged is None:
            raise AssertionError("clobber attempt removed the stored DEM")
        existing_preserved = bool(
            np.array_equal(heightmap, unchanged["heightmap"], equal_nan=True)
        )

        failed_acquisition_rejected = False
        try:
            write_dem(
                connection,
                tile_id,
                None,
                "arcticdem_10m",
                "EGM2008",
                commit=False,
            )
        except TypeError:
            failed_acquisition_rejected = True
        after_failure = read_dem_payload(connection, tile_id)
        if after_failure is None:
            raise AssertionError("failed acquisition removed the stored DEM")
        failed_acquisition_preserved = bool(
            np.array_equal(
                heightmap,
                after_failure["heightmap"],
                equal_nan=True,
            )
        )
        return {
            "tileId": tile_id,
            "firstWrite": first_write,
            "duplicateWrite": duplicate_write,
            "datumClobberBlocked": datum_clobber_blocked,
            "exactRoundTrip": exact_round_trip,
            "clobberBlocked": clobber_blocked,
            "existingPreserved": existing_preserved,
            "failedAcquisitionRejected": failed_acquisition_rejected,
            "failedAcquisitionPreserved": failed_acquisition_preserved,
            "source": stored["source"],
            "verticalDatum": stored["vertical_datum"],
            "geoidUndulation": geoid_undulation,
            "geometricError": stored["geometric_error"],
            **_heightmap_summary(stored["heightmap"]),
        }
    finally:
        connection.execute("ROLLBACK TO dem_persistence_test")
        connection.execute("RELEASE dem_persistence_test")
