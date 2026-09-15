"""Thread-safe browser GPU profiling command state."""

from __future__ import annotations

import copy
import datetime
import threading
import uuid
from typing import Any


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class GpuProfileControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._command_id = 0
        self._state = self._idle_state()

    def _idle_state(self) -> dict[str, Any]:
        return {
            "commandId": self._command_id,
            "profileId": None,
            "status": "idle",
            "desiredEnabled": False,
            "sampleInterval": 10,
            "startedAt": None,
            "stopRequestedAt": None,
            "completedAt": None,
            "client": None,
            "result": None,
            "error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def start(self, sample_interval: int = 60) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] in {"starting", "running", "stopping"}:
                raise RuntimeError("a GPU profile is already active")
            self._command_id += 1
            self._state = {
                "commandId": self._command_id,
                "profileId": uuid.uuid4().hex,
                "status": "starting",
                "desiredEnabled": True,
                "sampleInterval": sample_interval,
                "startedAt": _utc_now(),
                "stopRequestedAt": None,
                "completedAt": None,
                "client": None,
                "result": None,
                "error": None,
            }
            return copy.deepcopy(self._state)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._state["desiredEnabled"]:
                raise RuntimeError("no GPU profile is active")
            self._command_id += 1
            self._state["commandId"] = self._command_id
            self._state["status"] = "stopping"
            self._state["desiredEnabled"] = False
            self._state["stopRequestedAt"] = _utc_now()
            return copy.deepcopy(self._state)

    def report(
        self,
        *,
        profile_id: str,
        phase: str,
        client: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if profile_id != self._state["profileId"]:
                raise LookupError("profileId does not match the current profile")
            if client is not None:
                self._state["client"] = copy.deepcopy(client)
                self._state["client"]["lastSeenAt"] = _utc_now()
            if phase == "running":
                if not self._state["desiredEnabled"]:
                    raise RuntimeError("the profile has already been asked to stop")
                self._state["status"] = "running"
            elif phase == "complete":
                self._state["status"] = "complete"
                self._state["desiredEnabled"] = False
                self._state["completedAt"] = _utc_now()
                self._state["result"] = copy.deepcopy(result)
                self._state["error"] = None
            elif phase == "error":
                self._state["status"] = "error"
                self._state["desiredEnabled"] = False
                self._state["completedAt"] = _utc_now()
                self._state["result"] = copy.deepcopy(result)
                self._state["error"] = error or "browser GPU profiler failed"
            else:
                raise ValueError("phase must be running, complete, or error")
            return copy.deepcopy(self._state)
