"""Deterministic gate for latest-camera pending-work replacement."""

from __future__ import annotations

import threading

from dynamic_functions.Terrain.demand import (
    DemandCoordinator,
    DemandLane,
    prioritized_selection_ids,
)


@visible
def demand_priority_offline() -> dict:
    """Prove newest nearest-first work replaces only unstarted work."""

    active_started = threading.Event()
    release_active = threading.Event()
    execution_order = []

    def worker(item_id: str) -> dict:
        execution_order.append(item_id)
        if item_id == "active-old":
            active_started.set()
            if not release_active.wait(timeout=2.0):
                raise TimeoutError("priority fixture release was not signaled")
        return {"itemId": item_id}

    lane = DemandLane("priority-fixture", worker, 1)
    coordinator = DemandCoordinator({"dem": lane})
    try:
        coordinator.refresh(
            {"dem": ["active-old", "stale-far", "retained-mid"]}
        )
        started = active_started.wait(timeout=1.0)
        refreshed = coordinator.refresh(
            {"dem": ["new-near", "retained-mid", "new-far"]}
        )["dem"]
        auxiliary = lane.submit_claimed(
            ["stale-http-repoll", "retained-mid"]
        )
        before_release = lane.status()
        monitor = lane.monitor_status()
        release_active.set()
        idle = lane.wait_for_idle(timeout=2.0)

        selection = {
            "tileIds": ["far", "near", "middle"],
            "tiles": [
                {"tileId": "far", "distance": 900.0},
                {"tileId": "near", "distance": 10.0},
                {"tileId": "middle", "distance": 100.0},
            ],
            "coverage": [
                {"tileId": "cover-far", "targetIds": ["far"]},
                {"tileId": "cover-near", "targetIds": ["near", "middle"]},
            ],
        }
        target_ids, coverage_ids = prioritized_selection_ids(selection)
        return {
            "activeAllowedToFinish": started
            and before_release["active"] == ["active-old"],
            "obsoletePendingDropped": refreshed["dropped"] == ["stale-far"],
            "retainedWithoutDuplication": refreshed["retained"]
            == ["retained-mid"],
            "newNearestFirst": before_release["pending"]
            == ["new-near", "retained-mid", "new-far"],
            "staleAuxiliaryRejected": bool(
                auxiliary["acceptedCount"] == 0
                and auxiliary["ignored"] == ["stale-http-repoll"]
                and "stale-http-repoll" not in before_release["pending"]
                and before_release["claimedCount"] == 3
            ),
            "boundedBacklogMonitoring": bool(
                monitor["refreshGeneration"] == 2
                and monitor["claimedActiveCount"] == 0
                and monitor["staleActiveCount"] == 1
                and monitor["pendingCount"] == 3
                and monitor["totals"]["dropped"] == 1
                and monitor["totals"]["ignored"] == 1
                and monitor["staleActiveSample"] == ["active-old"]
            ),
            "executionOrder": idle
            and execution_order
            == ["active-old", "new-near", "retained-mid", "new-far"],
            "cameraDistanceOrder": target_ids == ["near", "middle", "far"],
            "coverageFollowsTargets": coverage_ids
            == ["cover-near", "cover-far"],
        }
    finally:
        release_active.set()
        lane.close()
