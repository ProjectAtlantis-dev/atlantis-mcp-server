"""Lifecycle for the Terrain-owned HTTP compatibility sidecar."""

from __future__ import annotations

import importlib
import json
import logging
import math
import threading
import time
from collections import deque
from logging import FileHandler
from pathlib import Path

import atlantis
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, request_response

from dynamic_functions.Terrain.Database.database import _update_dashboard
from dynamic_functions.Terrain.Database.database import connection_lock, db
from dynamic_functions.Terrain.bathymetry_map import query_bathymetry_map
from dynamic_functions.Terrain.coords import to_stereo
from dynamic_functions.Terrain.gpu_profile_control import GpuProfileControl
from dynamic_functions.Terrain.http_adapter import (
    compose_tiles_response,
    parse_tiles_request,
    serve_texture,
)
from dynamic_functions.Terrain.serve_flask import CLIENT_LOG_PATH
from dynamic_functions.Terrain.tile_address import require_tile_id
from dynamic_functions.Terrain.viewer_assets import (
    encode_buildings_response,
    query_buildings,
    startup_assets,
)


_RUNTIME_KEY = "Terrain.viewer_http.runtime.v1"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5180
_START_TIMEOUT_SECONDS = 5.0
log = logging.getLogger("terrain.viewer_http")
client_log = logging.getLogger("terrain.client")
_CLIENT_LOG_MAX_ENTRIES = 200
_CLIENT_LOG_RING_SIZE = 2000
_CLIENT_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
_gpu_profile_control = GpuProfileControl()
_FAVICON = b"""<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 64\"><rect width=\"64\" height=\"64\" rx=\"12\" fill=\"#071b2b\"/><path d=\"M7 48 23 19l10 18 7-11 17 22Z\" fill=\"#5ec7d7\"/><path d=\"M7 48h50\" stroke=\"#d9f7ff\" stroke-width=\"4\"/></svg>"""


def _configure_client_log() -> None:
    """Keep browser telemetry out of the main MCP/server log."""

    log_path = CLIENT_LOG_PATH.resolve()
    has_target_handler = any(
        isinstance(handler, FileHandler)
        and Path(handler.baseFilename).resolve() == log_path
        for handler in client_log.handlers
    )
    if not has_target_handler:
        CLIENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = FileHandler(
            str(CLIENT_LOG_PATH),
            mode="a",
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            )
        )
        client_log.addHandler(handler)
    client_log.setLevel(logging.DEBUG)
    client_log.propagate = False


_configure_client_log()


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
        self.client_log_ring: deque[dict] = deque(
            maxlen=_CLIENT_LOG_RING_SIZE
        )
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
        log.debug(
            "[/api/tiles] request method=%s qx=%.1f qy=%.1f agl=%.1f "
            "range=%.1f previous_depth=%s known=%d origin=%s",
            request.method,
            arguments["camera_x"],
            arguments["camera_y"],
            arguments["altitude"],
            arguments["max_range"],
            arguments["previous_depth"],
            len(arguments["known_digests"]),
            arguments["demand_origin"],
        )
        response = await run_in_threadpool(compose_tiles_response, arguments)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        response_log = log.warning if elapsed_ms >= 300.0 else log.debug
        response_log(
            "[/api/tiles] response method=%s origin=%s status=%d bytes=%d "
            "elapsed_ms=%.1f",
            request.method,
            arguments["demand_origin"],
            response.status_code,
            len(response.body),
            elapsed_ms,
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


async def _client_log(request: Request) -> JSONResponse:
    """Ingest the existing viewer's bounded structured-log batches."""

    try:
        data = await request.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"error": "invalid_client_log", "message": str(exc)},
            status_code=400,
        )
    if not isinstance(data, dict):
        return JSONResponse(
            {"error": "invalid_client_log", "message": "body must be an object"},
            status_code=400,
        )
    raw_entries = data.get("entries")
    entries = raw_entries if isinstance(raw_entries, list) else [data]
    incoming_count = len(entries)
    dropped = max(0, incoming_count - _CLIENT_LOG_MAX_ENTRIES)
    scene_mode = data.get("sceneMode")
    runtime = atlantis.server_shared.get(_RUNTIME_KEY)
    written = 0
    for item in entries[:_CLIENT_LOG_MAX_ENTRIES]:
        if not isinstance(item, dict):
            dropped += 1
            continue
        payload = {
            "ts": item.get("ts"),
            "sceneMode": item.get("sceneMode") or scene_mode,
            "phase": item.get("phase") or item.get("event") or "client.log",
            "elapsedMs": item.get("elapsedMs"),
            "memory": item.get("memory"),
            "details": item.get("details"),
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        level_name = str(item.get("level", "info")).strip().lower()
        line = json.dumps(payload, ensure_ascii=False, default=str)
        if len(line) > 20000:
            line = line[:20000] + "...<truncated>"
        client_log.log(_CLIENT_LOG_LEVELS.get(level_name, logging.INFO), line)
        if runtime is not None:
            runtime.client_log_ring.append({"level": level_name, **payload})
        written += 1
    return JSONResponse(
        {
            "ok": True,
            "written": written,
            "dropped": dropped,
            "logPath": str(CLIENT_LOG_PATH),
        }
    )


async def _client_log_ring(request: Request) -> JSONResponse:
    runtime = atlantis.server_shared.get(_RUNTIME_KEY)
    try:
        limit = int(request.query_params.get("limit", "200"))
    except (TypeError, ValueError):
        limit = 200
    limit = max(1, min(limit, _CLIENT_LOG_RING_SIZE))
    retained = list(runtime.client_log_ring) if runtime is not None else []
    entries = retained[-limit:]
    return JSONResponse(
        {
            "count": len(entries),
            "retainedCount": len(retained),
            "entries": entries,
        }
    )


def _query_float(request: Request, name: str, default: float) -> float:
    raw = request.query_params.get(name)
    try:
        value = default if raw is None else float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _query_position(request: Request) -> tuple[float, float]:
    if "sx" in request.query_params or "sy" in request.query_params:
        if "sx" not in request.query_params or "sy" not in request.query_params:
            raise ValueError("sx and sy must be supplied together")
        return _query_float(request, "sx", 0.0), _query_float(request, "sy", 0.0)
    latitude = _query_float(request, "lat", 64.175)
    longitude = _query_float(request, "lon", -51.7388)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("lat/lon are outside their valid ranges")
    x, y = to_stereo(latitude, longitude)
    return float(x), float(y)


async def _assets(_request: Request) -> JSONResponse:
    try:
        payload = await run_in_threadpool(startup_assets)
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})
    except Exception as exc:
        log.exception("Viewer startup asset request failed")
        return JSONResponse(
            {"error": "asset_catalog_failed", "message": str(exc)}, status_code=500
        )


async def _buildings(request: Request) -> Response:
    try:
        qx, qy = _query_position(request)
        max_range = _query_float(request, "range", 9000.0)
        if max_range <= 0:
            raise ValueError("range must be greater than zero")
        ox = _query_float(request, "ox", qx)
        oy = _query_float(request, "oy", qy)
        buildings, source = await run_in_threadpool(
            query_buildings, qx, qy, max_range, ox, oy
        )
        payload = encode_buildings_response(
            buildings, qx=qx, qy=qy, ox=ox, oy=oy, source=source
        )
        return Response(
            payload,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store", "X-Terrain-Format": "binary-v1"},
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            {"error": "invalid_buildings_request", "message": str(exc)},
            status_code=400,
        )
    except Exception as exc:
        log.exception("Viewer building request failed")
        return JSONResponse(
            {"error": "buildings_request_failed", "message": str(exc)}, status_code=500
        )


async def _bathymetry_map(request: Request) -> JSONResponse:
    try:
        qx, qy = _query_position(request)
        max_range = _query_float(request, "range", 50000.0)
        if max_range < 0:
            raise ValueError("range must be non-negative")
        ox = _query_float(request, "ox", qx)
        oy = _query_float(request, "oy", qy)
        with connection_lock():
            payload = query_bathymetry_map(
                db(), qx, qy, max_range, ox=ox, oy=oy
            )
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            {"error": "invalid_bathymetry_map_request", "message": str(exc)},
            status_code=400,
        )
    except Exception as exc:
        log.exception("Bathymetry map request failed")
        return JSONResponse(
            {"error": "bathymetry_map_failed", "message": str(exc)}, status_code=500
        )


async def _classifier(request: Request) -> Response:
    tile_id = request.path_params["tile_id"]
    try:
        require_tile_id(tile_id)
        resolution = int(request.query_params.get("res", "512"))
        if not 16 <= resolution <= 2048:
            raise ValueError("res must be between 16 and 2048")
    except (TypeError, ValueError):
        return Response(b"", status_code=400, headers={"Cache-Control": "no-store"})
    # This port intentionally has no classifier storage yet. 204 is the
    # established viewer contract for a known route with no applicable map.
    return Response(
        b"",
        status_code=204,
        headers={"Cache-Control": "no-store", "X-Classifier-Status": "missing"},
    )


async def _gpu_profile(_request: Request) -> JSONResponse:
    return JSONResponse(
        _gpu_profile_control.snapshot(), headers={"Cache-Control": "no-store"}
    )


async def _gpu_profile_start(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = {}
    raw_interval = payload.get("sampleInterval", 60) if isinstance(payload, dict) else 60
    try:
        if isinstance(raw_interval, bool):
            raise ValueError
        sample_interval = int(raw_interval)
    except (TypeError, ValueError):
        sample_interval = 0
    if not 1 <= sample_interval <= 600:
        return JSONResponse(
            {"ok": False, "error": "sampleInterval must be an integer from 1 to 600"},
            status_code=400,
        )
    try:
        state = _gpu_profile_control.start(sample_interval)
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    return JSONResponse(state, status_code=202)


async def _gpu_profile_stop(_request: Request) -> JSONResponse:
    try:
        state = _gpu_profile_control.stop()
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    return JSONResponse(state, status_code=202)


async def _gpu_profile_report(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        state = _gpu_profile_control.report(
            profile_id=str(payload.get("profileId") or ""),
            phase=str(payload.get("phase") or ""),
            client=payload.get("client") if isinstance(payload.get("client"), dict) else None,
            result=payload.get("result") if isinstance(payload.get("result"), dict) else None,
            error=str(payload.get("error") or "") or None,
        )
    except LookupError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse(state)


async def _favicon(_request: Request) -> Response:
    return Response(
        _FAVICON,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


_HOTLOAD_ROUTE_ENDPOINTS = {
    "/health": "_health",
    "/favicon.ico": "_favicon",
    "/api/client_log": "_client_log",
    "/api/client_log/ring": "_client_log_ring",
    "/api/assets": "_assets",
    "/api/buildings": "_buildings",
    "/api/bathymetry-map": "_bathymetry_map",
    "/api/classifier/{tile_id}.png": "_classifier",
    "/api/gpu-profile": "_gpu_profile",
    "/api/gpu-profile/start": "_gpu_profile_start",
    "/api/gpu-profile/stop": "_gpu_profile_stop",
    "/api/gpu-profile/report": "_gpu_profile_report",
    "/api/tiles": "_tiles",
    "/api/texture/{tile_id}.jpg": "_texture",
}


def _hotload_dispatch(endpoint_name: str):
    """Resolve a sidecar endpoint from the latest dynamic module instance."""

    async def dispatch(request: Request) -> Response:
        module = importlib.import_module(__name__)
        endpoint = getattr(module, endpoint_name)
        return await endpoint(request)

    dispatch.__name__ = f"hotload_{endpoint_name.lstrip('_')}"
    return dispatch


def _bind_hotload_routes(app: Starlette) -> int:
    """Make long-lived Starlette routes follow MCP dynamic-module reloads."""

    rebound = 0
    for route in app.routes:
        endpoint_name = _HOTLOAD_ROUTE_ENDPOINTS.get(route.path)
        if endpoint_name is None:
            continue
        endpoint = _hotload_dispatch(endpoint_name)
        route.endpoint = endpoint
        route.app = request_response(endpoint)
        rebound += 1
    return rebound


def _viewer_app() -> Starlette:
    app = Starlette(
        routes=[
            Route("/health", _health, methods=["GET"]),
            Route("/favicon.ico", _favicon, methods=["GET"]),
            Route("/api/client_log", _client_log, methods=["POST"]),
            Route("/api/client_log/ring", _client_log_ring, methods=["GET"]),
            Route("/api/assets", _assets, methods=["GET"]),
            Route("/api/buildings", _buildings, methods=["GET"]),
            Route("/api/bathymetry-map", _bathymetry_map, methods=["GET"]),
            Route(
                "/api/classifier/{tile_id}.png", _classifier, methods=["GET"]
            ),
            Route("/api/gpu-profile", _gpu_profile, methods=["GET"]),
            Route(
                "/api/gpu-profile/start", _gpu_profile_start, methods=["POST"]
            ),
            Route(
                "/api/gpu-profile/stop", _gpu_profile_stop, methods=["POST"]
            ),
            Route(
                "/api/gpu-profile/report", _gpu_profile_report, methods=["POST"]
            ),
            Route("/api/tiles", _tiles, methods=["GET", "POST"]),
            Route(
                "/api/texture/{tile_id}.jpg",
                _texture,
                methods=["GET"],
            ),
        ]
    )
    _bind_hotload_routes(app)
    return app


def _rebind_running_sidecar() -> int:
    """Upgrade an existing pre-dispatch sidecar without restarting it."""

    runtime = atlantis.server_shared.get(_RUNTIME_KEY)
    if runtime is None:
        return 0
    config = getattr(getattr(runtime, "server", None), "config", None)
    app = getattr(config, "app", None)
    if not isinstance(app, Starlette):
        return 0
    return _bind_hotload_routes(app)


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
async def server_start(
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
        await _update_dashboard()
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
    await _update_dashboard()
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
async def server_stop() -> dict:
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
    status = runtime.status()
    if not status["threadAlive"]:
        atlantis.server_shared.remove(_RUNTIME_KEY)
        log.info("viewer sidecar stopped")
    else:
        log.error("viewer sidecar thread did not stop")
    await _update_dashboard()
    return {
        "stopped": not status["threadAlive"],
        "alreadyStopped": False,
        **status,
    }


# DynamicFunctionManager removes modules from sys.modules on source changes,
# while the HTTP sidecar intentionally survives in server_shared. Rebind any
# pre-dispatch runtime as soon as this module is loaded again.
_rebind_running_sidecar()
