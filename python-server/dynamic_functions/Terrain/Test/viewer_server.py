"""Offline lifecycle gate for the Terrain viewer HTTP sidecar."""

from __future__ import annotations

import json
import socket
import urllib.request
from unittest.mock import AsyncMock, patch

from starlette.routing import Route

from dynamic_functions.Terrain.Server.server import start, status, stop
from dynamic_functions.Terrain.viewer_server import (
    CLIENT_LOG_PATH,
    _HOTLOAD_ROUTE_ENDPOINTS,
    _viewer_app,
    client_log,
)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@visible
async def viewer_server_offline() -> dict:
    """Prove explicit start, idempotence, health, and explicit stop."""

    route_app = _viewer_app()
    hotload_routes = {
        route.path: route.endpoint.__name__
        for route in route_app.routes
        if isinstance(route, Route)
    }
    with (
        patch(
            "dynamic_functions.Terrain.Server.server._update_dashboard",
            new=AsyncMock(),
        ),
        patch(
            "dynamic_functions.Terrain.viewer_server.startup_assets",
            return_value={
                "ok": True,
                "source": "asset_catalog",
                "catalogStatus": "ready",
                "catalogPath": "/fixture/assets.db",
                "schemaVersion": 4,
                "vehicle_definition": {
                    "url": "/models/vehicle.glb",
                    "realLengthM": 7.7,
                    "tireDiameterM": 1.27,
                    "altOffsetM": 0.05,
                },
                "structure_definition": {},
                "vehicle_instances": [{
                    "id": "vehicle-1",
                    "lat": 64.1,
                    "lon": -51.7,
                    "headingDeg": 0,
                    "z": 10,
                    "headlightsOn": True,
                    "savedAt": None,
                }],
                "structure_instances": [],
            },
        ),
        patch(
            "dynamic_functions.Terrain.viewer_server.query_buildings",
            return_value=([], "asset_catalog"),
        ),
    ):
        await stop()
        initially_stopped = status()
        port = _unused_port()
        try:
            started = await start(port=port)
            running = status()
            duplicate = await start(port=port)
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2.0
            ) as response:
                health = json.loads(response.read())
                health_status = response.status
            log_request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/client_log",
                data=json.dumps(
                    {
                        "sceneMode": "test",
                        "entries": [
                            {
                                "level": "error",
                                "phase": "terrain.residency.overlap",
                                "details": {
                                    "pairs": [
                                        {
                                            "ancestorId": "9-175-103",
                                            "descendantId": "12-1401-825",
                                            "depthGap": 3,
                                        }
                                    ]
                                },
                            }
                        ],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(log_request, timeout=2.0) as response:
                log_result = json.loads(response.read())
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/client_log/ring", timeout=2.0
            ) as response:
                log_ring = json.loads(response.read())
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/assets", timeout=2.0
            ) as response:
                assets = json.loads(response.read())
                assets_status = response.status
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/buildings"
                "?sx=-299570&sy=-2883110&range=9000"
                "&ox=-299570&oy=-2883110",
                timeout=5.0,
            ) as response:
                buildings = response.read()
                buildings_status = response.status
                buildings_format = response.headers.get("X-Terrain-Format")
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/gpu-profile", timeout=2.0
            ) as response:
                gpu_profile = json.loads(response.read())
                gpu_profile_status = response.status
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/classifier/12-1379-765.png"
                "?raw=1&res=256&v=11",
                timeout=2.0,
            ) as response:
                classifier_status = response.status
                classifier_state = response.headers.get("X-Classifier-Status")
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/bathymetry-map"
                "?sx=-299570&sy=-2883110&range=50000"
                "&ox=-299570&oy=-2883110",
                timeout=2.0,
            ) as response:
                bathymetry_map = json.loads(response.read())
                bathymetry_map_status = response.status
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/favicon.ico", timeout=2.0
            ) as response:
                favicon = response.read()
                favicon_status = response.status
        finally:
            stopped = await stop()
        already_stopped = await stop()
        finally_stopped = status()
    return {
        "hotloadRoutes": bool(
            set(hotload_routes) == set(_HOTLOAD_ROUTE_ENDPOINTS)
            and all(
                name.startswith("hotload_")
                for name in hotload_routes.values()
            )
        ),
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
        "clientLogIngested": bool(
            log_result
            == {
                "ok": True,
                "written": 1,
                "dropped": 0,
                "logPath": str(CLIENT_LOG_PATH),
            }
            and log_ring["count"] == 1
            and log_ring["retainedCount"] == 1
            and log_ring["entries"][0]["level"] == "error"
            and log_ring["entries"][0]["phase"]
            == "terrain.residency.overlap"
        ),
        "clientLogIsolated": bool(
            client_log.propagate is False
            and client_log.level == 10
            and any(
                getattr(handler, "baseFilename", None)
                == str(CLIENT_LOG_PATH)
                for handler in client_log.handlers
            )
        ),
        "compatibilityRoutes": bool(
            assets_status == 200
            and assets["ok"]
            and isinstance(assets["vehicle_instances"], list)
            and buildings_status == 200
            and buildings_format == "binary-v1"
            and len(buildings) >= 4
            and gpu_profile_status == 200
            and gpu_profile["status"] == "idle"
            and classifier_status == 204
            and classifier_state == "missing"
            and bathymetry_map_status == 200
            and isinstance(bathymetry_map["coverage"], list)
            and bathymetry_map["soundingStatus"] == "not_imported"
            and favicon_status == 200
            and favicon.startswith(b"<svg")
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
