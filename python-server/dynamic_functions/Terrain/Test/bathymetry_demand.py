"""Rollback-only checks for coastal bathymetry demand eligibility."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from dynamic_functions.Terrain.bathymetry_demand import (
    BathymetryDeferredError,
    BathymetryWorkerError,
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

        with (
            patch("dynamic_functions.Terrain.bathymetry_demand.db", return_value=connection),
            patch("dynamic_functions.Terrain.bathymetry_demand.missing_inputs") as inputs,
            patch("dynamic_functions.Terrain.bathymetry_demand.subprocess.run") as runner,
        ):
            existing = run_bathymetry_job(_MIXED_JOB)
            assert existing == {"tileId": _MIXED_JOB, "written": False, "rows": 0}
            inputs.assert_not_called()
            runner.assert_not_called()

        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"tileId": _NEAR_JOB, "written": True, "rows": 1}), stderr=""
        )
        with (
            patch("dynamic_functions.Terrain.bathymetry_demand.db", return_value=connection),
            patch("dynamic_functions.Terrain.bathymetry_demand.missing_inputs",
                  return_value={"dem": [], "coastline": []}),
            patch(
                "dynamic_functions.Terrain.bathymetry_demand.subprocess.run",
                return_value=completed,
            ) as runner,
        ):
            worker_result = run_bathymetry_job(_NEAR_JOB)
        worker_call = runner.call_args
        worker_command = worker_call.args[0]
        worker_environment = worker_call.kwargs["env"]

        deferred = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="coverage incomplete: 3/4"
        )
        deferred_retryable = False
        with (
            patch("dynamic_functions.Terrain.bathymetry_demand.db", return_value=connection),
            patch("dynamic_functions.Terrain.bathymetry_demand.missing_inputs",
                  return_value={"dem": [], "coastline": []}),
            patch(
                "dynamic_functions.Terrain.bathymetry_demand.subprocess.run",
                return_value=deferred,
            ),
        ):
            try:
                run_bathymetry_job(_NEAR_JOB)
            except BathymetryDeferredError:
                deferred_retryable = True
        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="solving region",
            stderr="Traceback:\nValueError: invalid solver input",
        )
        diagnostics_visible = False
        with (
            patch("dynamic_functions.Terrain.bathymetry_demand.db", return_value=connection),
            patch("dynamic_functions.Terrain.bathymetry_demand.missing_inputs",
                  return_value={"dem": [], "coastline": []}),
            patch("dynamic_functions.Terrain.bathymetry_demand.subprocess.run", return_value=failed),
        ):
            try:
                run_bathymetry_job(_NEAR_JOB)
            except BathymetryWorkerError as exc:
                diagnostics_visible = "invalid solver input" in str(exc) and "solving region" in str(exc)
        with (
            patch("dynamic_functions.Terrain.bathymetry_demand.db", return_value=connection),
            patch("dynamic_functions.Terrain.bathymetry_demand.missing_inputs",
                  return_value={"dem": [_MIXED], "coastline": [_NEAR_WATER]}),
            patch("dynamic_functions.Terrain.demand._coordinator") as coordinator,
            patch("dynamic_functions.Terrain.bathymetry_demand.subprocess.run") as subprocess_run,
        ):
            try:
                run_bathymetry_job(_NEAR_JOB)
            except BathymetryDeferredError:
                pass
            else:
                raise AssertionError("incomplete inputs must defer bathymetry")
            coordinator.return_value.submit_bathymetry_inputs.assert_called_once_with(
                _NEAR_JOB, {"dem": [_MIXED], "coastline": [_NEAR_WATER]},
            )
            subprocess_run.assert_not_called()
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
                and worker_command[:3] == [sys.executable, "-m", "dynamic_functions.Terrain.Bathymetry.worker"]
                and worker_command[worker_command.index("--db") + 1]
                == str(DATABASE_PATH)
                and "--base" not in worker_command
                and worker_environment["PYTHONPATH"]
                == str(Path(__file__).resolve().parents[3])
            ),
            "coverageFailureRetryable": deferred_retryable,
            "workerDiagnosticsVisible": diagnostics_visible,
            "nativePrerequisitesScheduled": True,
        }
    finally:
        connection.execute("ROLLBACK TO bathymetry_demand_test")
        connection.execute("RELEASE bathymetry_demand_test")
