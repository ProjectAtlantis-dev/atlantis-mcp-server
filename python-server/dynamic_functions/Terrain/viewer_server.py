"""Lifecycle for the Terrain-owned HTTP compatibility sidecar."""

from __future__ import annotations

import json
import logging
import threading
import time

import atlantis
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from dynamic_functions.Terrain.http_adapter import (
    compose_tiles_response,
    parse_tiles_request,
    serve_texture,
)


_RUNTIME_KEY = "Terrain.viewer_http.runtime.v1"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5180
_START_TIMEOUT_SECONDS = 5.0
log = logging.getLogger("terrain.viewer_http")


class _ViewerRuntime:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.server = uvicorn.Server(
            uvicorn.Config(
                _viewer_app(),
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="terrain-viewer-http",
            daemon=True,
        )

    def _run(self) -> None:
        try:
            self.server.run()
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("Terrain viewer HTTP sidecar stopped unexpectedly")

    def status(self) -> dict:
        alive = self.thread.is_alive()
        return {
            "running": bool(alive and self.server.started),
            "host": self.host,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}",
            "threadAlive": alive,
            "error": self.error,
        }


async def _tiles(request: Request) -> Response:
    started_at = time.perf_counter()
    try:
        body = await request.json() if request.method == "POST" else {}
        arguments = parse_tiles_request(request.query_params, body)
        log.info(
            "[/api/tiles] request method=%s qx=%.1f qy=%.1f agl=%.1f "
            "range=%.1f previous_depth=%s known=%d",
            request.method,
            arguments["camera_x"],
            arguments["camera_y"],
            arguments["altitude"],
            arguments["max_range"],
            arguments["previous_depth"],
            len(arguments["known_digests"]),
        )
        response = await run_in_threadpool(compose_tiles_response, arguments)
        log.info(
            "[/api/tiles] response status=%d bytes=%d elapsed_ms=%.1f",
            response.status_code,
            len(response.body),
            (time.perf_counter() - started_at) * 1000.0,
        )
        return response
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        log.warning(
            "[/api/tiles] invalid method=%s error=%s elapsed_ms=%.1f",
            request.method,
            exc,
            (time.perf_counter() - started_at) * 1000.0,
        )
        return JSONResponse(
            {"error": "invalid_terrain_request", "message": str(exc)},
            status_code=400,
        )
    except Exception as exc:
        log.exception("Terrain tile request failed")
        return JSONResponse(
            {"error": "terrain_request_failed", "message": str(exc)},
            status_code=500,
        )


async def _texture(request: Request) -> Response:
    started_at = time.perf_counter()
    tile_id = request.path_params["tile_id"]
    try:
        log.info("[/api/texture] request tile_id=%s", tile_id)
        response = await run_in_threadpool(
            serve_texture,
            tile_id,
            request.headers.get("if-none-match"),
        )
        log.info(
            "[/api/texture] response tile_id=%s status=%d tex_status=%s "
            "source=%s ancestor=%s bytes=%d elapsed_ms=%.1f",
            tile_id,
            response.status_code,
            response.headers.get("x-tex-status", ""),
            response.headers.get("x-tex-source", ""),
            response.headers.get("x-tex-ancestor", ""),
            len(response.body),
            (time.perf_counter() - started_at) * 1000.0,
        )
        return response
    except (TypeError, ValueError) as exc:
        log.warning(
            "[/api/texture] invalid tile_id=%s error=%s elapsed_ms=%.1f",
            tile_id,
            exc,
            (time.perf_counter() - started_at) * 1000.0,
        )
        return JSONResponse(
            {"error": "invalid_texture_request", "message": str(exc)},
            status_code=400,
        )
    except Exception as exc:
        log.exception("Terrain texture request failed")
        return JSONResponse(
            {"error": "texture_request_failed", "message": str(exc)},
            status_code=500,
        )


async def _health(_request: Request) -> JSONResponse:
    runtime = atlantis.server_shared.get(_RUNTIME_KEY)
    status = runtime.status() if runtime is not None else {"running": False}
    log.info("[/health] running=%s", status.get("running", False))
    return JSONResponse({"status": "healthy", "service": "terrain-viewer", **status})


def _viewer_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/health", _health, methods=["GET"]),
            Route("/api/tiles", _tiles, methods=["GET", "POST"]),
            Route(
                "/api/texture/{tile_id}.jpg",
                _texture,
                methods=["GET"],
            ),
        ]
    )


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
def server_start(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
) -> dict:
    """Start the viewer HTTP sidecar without modifying the MCP host."""

    bind_host, bind_port = _validated_bind(host, port)
    current = atlantis.server_shared.get(_RUNTIME_KEY)
    if current is not None and current.thread.is_alive():
        status = current.status()
        if (status["host"], status["port"]) != (bind_host, bind_port):
            raise RuntimeError(
                "terrain viewer server is already running at "
                f"{status['url']}"
            )
        log.info("viewer sidecar already running at %s", status["url"])
        return {"started": False, "alreadyRunning": True, **status}
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
    status = runtime.status()
    if not status["running"]:
        atlantis.server_shared.remove(_RUNTIME_KEY)
        detail = status["error"] or "startup timed out or bind failed"
        raise RuntimeError(f"terrain viewer server failed to start: {detail}")
    log.info("viewer sidecar started at %s", status["url"])
    return {"started": True, "alreadyRunning": False, **status}


@visible
def server_status() -> dict:
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
def server_stop() -> dict:
    """Stop the Terrain viewer sidecar and release its listening port."""

    runtime = atlantis.server_shared.get(_RUNTIME_KEY)
    if runtime is None:
        log.info("viewer sidecar already stopped")
        return {"stopped": False, "alreadyStopped": True, "running": False}
    log.info("stopping viewer sidecar at http://%s:%d", runtime.host, runtime.port)
    runtime.server.should_exit = True
    runtime.thread.join(timeout=5.0)
    if runtime.thread.is_alive():
        runtime.server.force_exit = True
        runtime.thread.join(timeout=1.0)
    status = runtime.status()
    if not status["threadAlive"]:
        atlantis.server_shared.remove(_RUNTIME_KEY)
        log.info("viewer sidecar stopped")
    else:
        log.error("viewer sidecar thread did not stop")
    return {
        "stopped": not status["threadAlive"],
        "alreadyStopped": False,
        **status,
    }
