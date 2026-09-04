"""Visible lifecycle controls for the Terrain viewer HTTP sidecar."""

import logging
import time

import atlantis

from dynamic_functions.Terrain.Database.database import _update_dashboard
from dynamic_functions.Terrain.viewer_server import (
    _RUNTIME_KEY,
    _ViewerRuntime,
)


_START_TIMEOUT_SECONDS = 5.0
log = logging.getLogger("terrain.viewer_http")


def _validated_bind(host: str, port: int) -> tuple[str, int]:
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    if isinstance(port, bool):
        raise ValueError("port must be an integer")
    try:
        parsed_port = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError("port must be an integer") from exc
    if not 1 <= parsed_port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return host.strip(), parsed_port


@visible
async def start(host: str = "127.0.0.1", port: int = 5180) -> dict:
    """Start the viewer HTTP sidecar without modifying the MCP host."""

    bind_host, bind_port = _validated_bind(host, port)
    current = atlantis.server_shared.get(_RUNTIME_KEY)
    if current is not None and current.thread.is_alive():
        current_status = current.status()
        if (current_status["host"], current_status["port"]) != (
            bind_host,
            bind_port,
        ):
            raise RuntimeError(
                "terrain viewer server is already running at "
                f"{current_status['url']}"
            )
        log.info(
            "viewer sidecar already running at %s",
            current_status["url"],
        )
        await _update_dashboard()
        return {
            "started": False,
            "alreadyRunning": True,
            **current_status,
        }
    if current is not None:
        atlantis.server_shared.remove(_RUNTIME_KEY)

    runtime = _ViewerRuntime(bind_host, bind_port)
    log.info("starting viewer sidecar at http://%s:%d", bind_host, bind_port)
    atlantis.server_shared.set(_RUNTIME_KEY, runtime)
    runtime.thread.start()
    deadline = time.monotonic() + _START_TIMEOUT_SECONDS
    while (
        runtime.thread.is_alive()
        and not runtime.server.started
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    current_status = runtime.status()
    if not current_status["running"]:
        atlantis.server_shared.remove(_RUNTIME_KEY)
        detail = current_status["error"] or "startup timed out or bind failed"
        raise RuntimeError(f"terrain viewer server failed to start: {detail}")
    log.info("viewer sidecar started at %s", current_status["url"])
    await _update_dashboard()
    return {
        "started": True,
        "alreadyRunning": False,
        **current_status,
    }


@visible
def status() -> dict:
    """Return the current viewer sidecar state without changing it."""

    runtime = atlantis.server_shared.get(_RUNTIME_KEY)
    if runtime is None:
        return {
            "running": False,
            "host": None,
            "port": None,
            "url": None,
            "threadAlive": False,
            "error": None,
        }
    return runtime.status()


@visible
async def stop() -> dict:
    """Stop the Terrain viewer sidecar and release its listening port."""

    runtime = atlantis.server_shared.get(_RUNTIME_KEY)
    if runtime is None:
        log.info("viewer sidecar already stopped")
        await _update_dashboard()
        return {"stopped": False, "alreadyStopped": True, "running": False}
    log.info("stopping viewer sidecar at http://%s:%d", runtime.host, runtime.port)
    runtime.server.should_exit = True
    runtime.thread.join(timeout=5.0)
    if runtime.thread.is_alive():
        runtime.server.force_exit = True
        runtime.thread.join(timeout=1.0)
    current_status = runtime.status()
    if not current_status["threadAlive"]:
        atlantis.server_shared.remove(_RUNTIME_KEY)
        log.info("viewer sidecar stopped")
    else:
        log.error("viewer sidecar thread did not stop")
    await _update_dashboard()
    return {
        "stopped": not current_status["threadAlive"],
        "alreadyStopped": False,
        **current_status,
    }
