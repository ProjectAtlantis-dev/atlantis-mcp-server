"""Deterministic isolation gate for nonblocking terrain demand lanes."""

from __future__ import annotations

import datetime
import threading
import time

import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import write_dem
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.demand import (
    DemandCoordinator,
    DemandLane,
    demand_candidates,
)
from dynamic_functions.Terrain.hydrography import write_hydrography_mask
from dynamic_functions.Terrain.tile_address import ancestor_tile_ids


_READY = "10-700-300"
_MISSING = "10-701-300"
_FALLBACK = "9-350-150"


def _candidate_staging() -> tuple[bool, bool]:
    connection = db()
    fixture_ids: set[str] = set()
    for tile_id in (_READY, _MISSING, _FALLBACK):
        fixture_ids.update(ancestor_tile_ids(tile_id, include_self=True))
    marks = ",".join("?" for _ in fixture_ids)
    args = tuple(fixture_ids)
    connection.execute("SAVEPOINT demand_candidates_test")
    try:
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
        values = np.full((65, 65), 20.0, dtype=np.float32)
        for tile_id in (_READY, _FALLBACK):
            write_dem(
                connection,
                tile_id,
                values,
                "arcticdem_10m",
                "EGM2008",
                commit=False,
            )
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        connection.execute(
            "INSERT INTO textures (tile_id,source,texture,updated_at) "
            "VALUES (?, ?, ?, ?)",
            (_READY, "fixture_texture", b"fixture", now),
        )
        mask = np.zeros((65, 65), dtype=bool)
        mask[0, 0] = True
        write_coastline_mask(
            connection, _FALLBACK, mask, "fixture_coast", 1, commit=False
        )
        write_hydrography_mask(
            connection, _FALLBACK, mask, "fixture_hydro", 1, commit=False
        )
        before = connection.total_changes
        candidates = demand_candidates(
            connection,
            [_READY, _MISSING],
            [_FALLBACK],
        )
        expected = {
            "dem": [_MISSING],
            "texture": ["10-700-300", "9-348-148"],
            "coastline": [_READY],
            "hydrography": [_READY],
        }
        staged = all(candidates[name] == value for name, value in expected.items())
        staged = bool(
            staged
            and len(candidates["connectivity"]) == 1
            and candidates["connectivity"][0].startswith("9:")
        )
        return staged, connection.total_changes == before
    finally:
        connection.execute("ROLLBACK TO demand_candidates_test")
        connection.execute("RELEASE demand_candidates_test")


@visible
def demand_lanes_offline() -> dict:
    """Prove independent capacity, deduplication, and immediate submission."""

    slow_started = threading.Event()
    release_slow = threading.Event()
    fast_finished = threading.Event()
    slow_order = []

    def slow_worker(item_id: str) -> dict:
        slow_order.append(item_id)
        slow_started.set()
        if item_id == "slow-a" and not release_slow.wait(timeout=2.0):
            raise TimeoutError("fixture release was not signaled")
        return {"itemId": item_id}

    def fast_worker(item_id: str) -> dict:
        fast_finished.set()
        return {"itemId": item_id}

    def failing_worker(item_id: str) -> dict:
        raise RuntimeError(f"fixture failure: {item_id}")

    slow = DemandLane("fixture-slow", slow_worker, 1)
    fast = DemandLane("fixture-fast", fast_worker, 1)
    failing = DemandLane("fixture-failing", failing_worker, 1)
    coordinator = DemandCoordinator(
        {"slow": slow, "fast": fast, "failing": failing}
    )
    try:
        started = time.perf_counter()
        submitted = coordinator.submit(
            {
                "slow": ["slow-a", "slow-b"],
                "fast": ["fast-a"],
                "failing": ["bad-a"],
            }
        )
        submit_ms = (time.perf_counter() - started) * 1000.0
        slow_is_active = slow_started.wait(timeout=1.0)
        fast_is_independent = fast_finished.wait(timeout=1.0)
        before_release = slow.status()
        duplicate = slow.submit(["slow-a", "slow-b"])

        release_slow.set()
        all_idle = all(
            lane.wait_for_idle(timeout=2.0)
            for lane in (slow, fast, failing)
        )
        final = coordinator.status()
        failure_isolated = bool(
            "bad-a" in final["failing"]["failures"]
            and final["fast"]["completedCount"] == 1
            and final["slow"]["completedCount"] == 2
        )
        failed_not_resubmitted = failing.submit(["bad-a"])["acceptedCount"] == 0

        unknown_rejected = False
        try:
            coordinator.submit({"unknown": ["x"]})
        except ValueError:
            unknown_rejected = True

        staged, candidate_reads_are_read_only = _candidate_staging()
        return {
            "immediateReturn": submit_ms < 100.0,
            "slowStarted": slow_is_active,
            "independentCapacity": fast_is_independent,
            "boundedActive": bool(
                before_release["active"] == ["slow-a"]
                and before_release["pending"] == ["slow-b"]
            ),
            "deduplicated": duplicate["acceptedCount"] == 0,
            "allWorkCompleted": all_idle and slow_order == ["slow-a", "slow-b"],
            "failureIsolated": failure_isolated,
            "failedNotHotLooped": failed_not_resubmitted,
            "unknownLaneRejected": unknown_rejected,
            "dependencyStaged": staged,
            "candidateReadsAreReadOnly": candidate_reads_are_read_only,
            "waitedForWorkers": False,
            "submittedCounts": {
                name: result["acceptedCount"]
                for name, result in submitted.items()
            },
            "submitMs": round(submit_ms, 3),
        }
    finally:
        release_slow.set()
        for lane in (slow, fast, failing):
            lane.close()
