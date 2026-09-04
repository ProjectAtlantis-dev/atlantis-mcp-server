"""Deterministic gate for bounded, non-sleeping demand retries."""

from __future__ import annotations

import time

from rasterio.errors import RasterioIOError

from dynamic_functions.Terrain.demand import DemandLane, retryable_failure


@visible
def demand_retry_offline() -> dict:
    """Prove deadlines, exhaustion, and terminal failure classification."""

    now = [100.0]
    transient_attempts = []

    def clock() -> float:
        return now[0]

    def transient_worker(item_id: str) -> dict:
        transient_attempts.append(item_id)
        if len(transient_attempts) < 3:
            raise TimeoutError("fixture provider timeout")
        return {"itemId": item_id}

    transient = DemandLane(
        "retry-transient",
        transient_worker,
        1,
        retry_delays=(5.0, 10.0),
        clock=clock,
    )
    terminal = DemandLane(
        "retry-terminal",
        lambda _item_id: (_ for _ in ()).throw(
            ValueError("fixture corrupt dimensions")
        ),
        1,
        retry_delays=(5.0,),
        clock=clock,
    )
    exhausting = DemandLane(
        "retry-exhausting",
        lambda _item_id: (_ for _ in ()).throw(
            ConnectionError("fixture connection reset")
        ),
        1,
        retry_delays=(3.0,),
        clock=clock,
    )
    try:
        transient.submit(["tile"])
        transient.wait_for_idle(timeout=1.0)
        first = transient.status()["failures"]["tile"]

        now[0] = 104.9
        started = time.perf_counter()
        too_early = transient.replace_pending(["tile"])
        refresh_ms = (time.perf_counter() - started) * 1000.0

        now[0] = 105.0
        second_submit = transient.replace_pending(["tile"])
        transient.wait_for_idle(timeout=1.0)
        second = transient.status()["failures"]["tile"]

        now[0] = 115.0
        third_submit = transient.replace_pending(["tile"])
        transient.wait_for_idle(timeout=1.0)
        transient_final = transient.status()

        terminal.submit(["bad"])
        terminal.wait_for_idle(timeout=1.0)
        terminal_failure = terminal.status()["failures"]["bad"]
        now[0] = 1000.0
        terminal_retry = terminal.replace_pending(["bad"])

        exhausting.submit(["flaky"])
        exhausting.wait_for_idle(timeout=1.0)
        now[0] = 1003.0
        exhausting.replace_pending(["flaky"])
        exhausting.wait_for_idle(timeout=1.0)
        exhausted = exhausting.status()["failures"]["flaky"]
        now[0] = 1063.0
        reclaimed = exhausting.replace_pending(["flaky"])
        exhausting.wait_for_idle(timeout=1.0)
        reclaimed_failure = exhausting.status()["failures"]["flaky"]

        return {
            "firstDeadline": bool(
                first["attempts"] == 1
                and first["retryable"]
                and first["retryAt"] == 105.0
            ),
            "noEarlyRetry": too_early["acceptedCount"] == 0
            and len(transient_attempts) == 3,
            "laterPassEligible": second_submit["acceptedCount"] == 1
            and second["attempts"] == 2
            and second["retryAt"] == 115.0,
            "boundedEventuallySucceeds": third_submit["acceptedCount"] == 1
            and transient_final["completedCount"] == 1
            and transient_final["failures"] == {},
            "refreshDoesNotSleep": refresh_ms < 100.0,
            "terminalNotRetried": bool(
                terminal_failure["retryable"] is False
                and terminal_retry["acceptedCount"] == 0
            ),
            "transientExhausted": bool(
                exhausted["attempts"] == 2
                and exhausted["retryable"] is False
                and exhausted["exhausted"] is True
                and exhausted["reclaimAt"] == 1063.0
            ),
            "exhaustedReclaimedAfterCooldown": bool(
                reclaimed["acceptedCount"] == 1
                and reclaimed_failure["attempts"] == 1
                and reclaimed_failure["retryable"] is True
            ),
            "classifierBoundaries": bool(
                retryable_failure(TimeoutError("x"))
                and retryable_failure(ConnectionError("x"))
                and retryable_failure(
                    RasterioIOError("HTTP response code: 503")
                )
                and not retryable_failure(
                    RasterioIOError("HTTP response code: 404")
                )
                and not retryable_failure(ValueError("bad payload"))
                and not retryable_failure(RuntimeError("missing token"))
            ),
            "refreshMs": round(refresh_ms, 3),
        }
    finally:
        for lane in (transient, terminal, exhausting):
            lane.close()
