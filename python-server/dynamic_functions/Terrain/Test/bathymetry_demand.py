"""Rollback-only checks for coastal bathymetry demand eligibility."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from dynamic_functions.Terrain.bathymetry_demand import (
    BathymetryDeferredError,
    eligible_fjord_jobs,
    run_bathymetry_job,
)
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.Database.bathymetry import write_bathymetry
from dynamic_functions.Terrain.Database.database import DATABASE_PATH, db


_MIXED = "12-1599-1600"
_NEAR_WATER = "12-1602-1600"
_FAR_WATER = "12-1616-1600"
_LAND = "12-1600-1601"
_NO_MASK = "12-1601-1601"
_MIXED_JOB = "8-99-100"
_NEAR_JOB = "8-100-100"
_FAR_JOB = "8-101-100"


@visible
def bathymetry_demand_offline() -> dict:
    """Prove mixed/near-water selection, coalescing, and persistence gating."""

    connection = db()
    connection.execute("SAVEPOINT bathymetry_demand_test")
    try:
        tile_ids = (
            _MIXED,
            _NEAR_WATER,
            _FAR_WATER,
            _LAND,
            _NO_MASK,
            _MIXED_JOB,
            _NEAR_JOB,
            _FAR_JOB,
        )
        marks = ",".join("?" for _ in tile_ids)
        connection.execute(
            f"DELETE FROM coastline_masks WHERE tile_id IN ({marks})",
            tile_ids,
        )
        connection.execute(
            f"DELETE FROM bathymetry WHERE tile_id IN ({marks})", tile_ids
        )
        connection.execute(
            f"DELETE FROM tiles WHERE tile_id IN ({marks})", tile_ids
        )

        land = np.zeros((65, 65), dtype=bool)
        mixed = land.copy()
        mixed[:, :32] = True
        water = np.ones((65, 65), dtype=bool)
        for tile_id, mask in (
            (_MIXED, mixed),
            (_NEAR_WATER, water),
            (_FAR_WATER, water),
            (_LAND, land),
        ):
            write_coastline_mask(
                connection,
                tile_id,
                mask,
                "fixture_coastline",
                1,
                commit=False,
            )

        visible = [_MIXED, _NEAR_WATER, _FAR_WATER, _LAND, _NO_MASK]
        before = connection.total_changes
        first = eligible_fjord_jobs(connection, visible)
        read_only = connection.total_changes == before

        write_bathymetry(
            connection,
            _MIXED_JOB,
            np.full((65, 65), -20.0, dtype=np.float32),
            source="fixture_bathymetry",
            version=1,
            commit=False,
        )
        remaining = eligible_fjord_jobs(connection, visible)
        coarse = eligible_fjord_jobs(connection, ["11-800-800"])

        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="bathymetry complete", stderr=""
        )
        with (
            patch.dict(os.environ, {"GLACIER_ROOT": "/fixture/glacier"}),
            patch("pathlib.Path.is_file", return_value=True),
            patch(
                "dynamic_functions.Terrain.bathymetry_demand.subprocess.run",
                return_value=completed,
            ) as runner,
        ):
            worker_result = run_bathymetry_job(_MIXED_JOB)
        worker_call = runner.call_args
        worker_command = worker_call.args[0]
        worker_environment = worker_call.kwargs["env"]

        deferred = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="coverage incomplete: 3/4"
        )
        deferred_retryable = False
        with (
            patch.dict(os.environ, {"GLACIER_ROOT": "/fixture/glacier"}),
            patch("pathlib.Path.is_file", return_value=True),
            patch(
                "dynamic_functions.Terrain.bathymetry_demand.subprocess.run",
                return_value=deferred,
            ),
        ):
            try:
                run_bathymetry_job(_MIXED_JOB)
            except BathymetryDeferredError:
                deferred_retryable = True
        return {
            "mixedCoastEligible": _MIXED_JOB in first,
            "nearWaterEligible": _NEAR_JOB in first,
            "farWaterExcluded": _FAR_JOB not in first,
            "landAndMissingExcluded": first == {_MIXED_JOB, _NEAR_JOB},
            "existingCoverageExcluded": remaining == {_NEAR_JOB},
            "contractDepthRequired": coarse == set(),
            "readOnlySelection": read_only,
            "workerUsesTargetRuntime": bool(
                worker_result["written"]
                and worker_command[0] == "/fixture/glacier/runOnDemand"
                and worker_command[worker_command.index("--db") + 1]
                == str(DATABASE_PATH)
                and worker_command[worker_command.index("--python") + 1]
                == sys.executable
                and worker_environment["PYTHON_BIN"] == sys.executable
                and worker_environment["SERVER_DIR"]
                == str(DATABASE_PATH.parents[3])
            ),
            "coverageFailureRetryable": deferred_retryable,
        }
    finally:
        connection.execute("ROLLBACK TO bathymetry_demand_test")
        connection.execute("RELEASE bathymetry_demand_test")
