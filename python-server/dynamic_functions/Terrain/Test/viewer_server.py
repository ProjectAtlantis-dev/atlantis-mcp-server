"""Offline lifecycle gate for the Terrain viewer HTTP sidecar."""

from __future__ import annotations

import json
import socket
import urllib.request

from dynamic_functions.Terrain.viewer_server import (
    server_start,
    server_status,
    server_stop,
)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@visible
async def viewer_server_offline() -> dict:
    """Prove explicit start, idempotence, health, and explicit stop."""

    await server_stop()
    initially_stopped = server_status()
    port = _unused_port()
    try:
        started = await server_start(port=port)
        running = server_status()
        duplicate = await server_start(port=port)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2.0
        ) as response:
            health = json.loads(response.read())
            health_status = response.status
    finally:
        stopped = await server_stop()
    already_stopped = await server_stop()
    finally_stopped = server_status()
    return {
        "statusBeforeStart": bool(
            not initially_stopped["running"]
            and not initially_stopped["threadAlive"]
            and initially_stopped["host"] is None
            and initially_stopped["port"] is None
            and initially_stopped["url"] is None
            and initially_stopped["error"] is None
        ),
        "starts": bool(
            started["started"]
            and started["running"]
            and started["port"] == port
        ),
        "statusWhileRunning": bool(
            running["running"]
            and running["threadAlive"]
            and running["host"] == "127.0.0.1"
            and running["port"] == port
            and running["url"] == f"http://127.0.0.1:{port}"
            and running["error"] is None
        ),
        "idempotentStart": bool(
            duplicate["started"] is False
            and duplicate["alreadyRunning"]
            and duplicate["port"] == port
        ),
        "health": bool(
            health_status == 200
            and health["status"] == "healthy"
            and health["service"] == "terrain-viewer"
            and health["running"]
        ),
        "stops": bool(stopped["stopped"] and not stopped["running"]),
        "idempotentStop": bool(
            already_stopped["alreadyStopped"]
            and not already_stopped["running"]
        ),
        "statusAfterStop": bool(
            not finally_stopped["running"]
            and not finally_stopped["threadAlive"]
            and finally_stopped["host"] is None
            and finally_stopped["port"] is None
            and finally_stopped["url"] is None
            and finally_stopped["error"] is None
        ),
    }
