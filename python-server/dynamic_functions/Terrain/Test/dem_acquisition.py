"""Deterministic checks for dual-provider DEM acquisition."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import numpy as np
from rasterio.errors import RasterioIOError

from dynamic_functions.Terrain.copernicus_dem import copernicus_request
from dynamic_functions.Terrain.Database import schema
from dynamic_functions.Terrain.Database.tiles import read_dem_payload, write_dem
from dynamic_functions.Terrain.dem_acquisition import (
    DemAcquisitionError,
    fetch_best_dem,
)
from dynamic_functions.Terrain.demand import retryable_failure


_TILE_ID = "10-334-192"


def _heightmap(valid_samples: int, value: float) -> np.ndarray:
    result = np.full((65, 65), np.nan, dtype=np.float32)
    result.flat[:valid_samples] = np.float32(value)
    return result


@visible
def dem_acquisition_offline() -> dict:
    """Prove URL construction, provider selection, and explicit failures."""

    arctic_full = (_heightmap(65 * 65, 10.0), [{"url": "arctic"}], 28.5)
    arctic_partial = (_heightmap(100, 10.0), [{"url": "arctic"}], 28.5)
    copernicus_full = (_heightmap(65 * 65, 11.0), [{"url": "copernicus"}])
    copernicus_partial = (_heightmap(200, 11.0), [{"url": "copernicus"}])

    with (
        patch(
            "dynamic_functions.Terrain.dem_acquisition._fetch_arcticdem",
            return_value=arctic_full,
        ),
        patch(
            "dynamic_functions.Terrain.dem_acquisition._fetch_copernicus",
            return_value=copernicus_full,
        ),
    ):
        tie = fetch_best_dem(_TILE_ID)

    with (
        patch(
            "dynamic_functions.Terrain.dem_acquisition._fetch_arcticdem",
            return_value=arctic_partial,
        ),
        patch(
            "dynamic_functions.Terrain.dem_acquisition._fetch_copernicus",
            return_value=copernicus_partial,
        ),
    ):
        more_complete = fetch_best_dem(_TILE_ID)

    arctic_404 = RasterioIOError("HTTP response code: 404")
    with (
        patch(
            "dynamic_functions.Terrain.dem_acquisition._fetch_arcticdem",
            side_effect=arctic_404,
        ),
        patch(
            "dynamic_functions.Terrain.dem_acquisition._fetch_copernicus",
            return_value=copernicus_full,
        ),
    ):
        copernicus_after_arctic_failure = fetch_best_dem(_TILE_ID)

    copernicus_timeout = TimeoutError("Copernicus connection timed out")
    with (
        patch(
            "dynamic_functions.Terrain.dem_acquisition._fetch_arcticdem",
            side_effect=arctic_404,
        ),
        patch(
            "dynamic_functions.Terrain.dem_acquisition._fetch_copernicus",
            side_effect=copernicus_timeout,
        ),
    ):
        aggregate = None
        try:
            fetch_best_dem(_TILE_ID)
        except DemAcquisitionError as exc:
            aggregate = exc

    request = copernicus_request(_TILE_ID)
    connection = sqlite3.connect(":memory:")
    schema.create(connection)
    write_dem(
        connection,
        _TILE_ID,
        copernicus_full[0],
        "copernicus",
        "EGM2008",
    )
    persisted = read_dem_payload(connection, _TILE_ID)
    connection.close()
    checks = {
        "copernicusUrl": request["sources"][0]["url"].endswith(
            "/Copernicus_DSM_COG_10_N64_00_W053_00_DEM/"
            "Copernicus_DSM_COG_10_N64_00_W053_00_DEM.tif"
        ),
        "copernicusDatum": request["verticalDatum"] == "EGM2008",
        "arcticWinsTie": tie["provider"] == "arcticdem",
        "moreCompleteWins": more_complete["provider"] == "copernicus",
        "copernicusPersists": (
            persisted is not None
            and persisted["source"] == "copernicus"
            and persisted["vertical_datum"] == "EGM2008"
        ),
        "providerFailover": (
            copernicus_after_arctic_failure["provider"] == "copernicus"
            and copernicus_after_arctic_failure["attempts"][0]["provider"]
            == "arcticdem"
            and copernicus_after_arctic_failure["attempts"][0]["status"]
            == "failed"
        ),
        "aggregateFailureVisible": (
            aggregate is not None
            and "provider=arcticdem" in str(aggregate)
            and "provider=copernicus" in str(aggregate)
        ),
        "aggregateRetryable": (
            aggregate is not None and retryable_failure(aggregate)
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(
            "DEM acquisition checks failed: " + ", ".join(failed)
        )
    return checks
