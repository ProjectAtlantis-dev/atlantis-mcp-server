"""Lifecycle owner and local HTTP client for the headless simulation service.

The fixed-step loop lives in the supervised Node child process. Dynamic
functions call this module as a strategic API; they never own or schedule its
tick.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


log = logging.getLogger("atlantis.simulation_host")
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 5190
_START_TIMEOUT_SECONDS = 5.0


class SimulationHost:
    """Supervise one local simulation child and authenticate every request."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._host = _DEFAULT_HOST
        self._port = _DEFAULT_PORT
        self._token: str | None = None
        self._started_at: float | None = None
        self._database_path: Path | None = None
        self._terrain_worker = None

    @property
    def base_url(self) -> str:
        address = f"[{self._host}]" if ":" in self._host else self._host
        return f"http://{address}:{self._port}"

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        authenticated: bool = True,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if authenticated:
            if self._token is None:
                raise RuntimeError("simulation service has not been started")
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 409:
                try:
                    conflict = json.loads(detail)
                except json.JSONDecodeError:
                    conflict = None
                if isinstance(conflict, dict):
                    return conflict
            raise RuntimeError(f"simulation request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"simulation service unavailable: {exc.reason}") from exc

    def start(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        database_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string")
        if isinstance(port, bool) or not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        configured_database = database_path or os.environ.get('ATLANTIS_SIM_DB_PATH')
        if not configured_database:
            raise RuntimeError('ATLANTIS_SIM_DB_PATH must explicitly select module runtime data outside source code')
        selected_database = Path(configured_database).expanduser().resolve()
        if host.strip() not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('simulation child must bind to loopback; expose it through an authenticated host adapter')
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                if (self._host, self._port, self._database_path) != (host.strip(), int(port), selected_database):
                    raise RuntimeError(f"simulation already runs at {self.base_url}")
                return {"started": False, "alreadyRunning": True, **self.status()}
            node = shutil.which("node")
            if node is None:
                raise RuntimeError("node executable is required for the simulation service")
            entrypoint = Path(__file__).resolve().parent / 'runtime' / 'src' / 'server.mjs'
            if not entrypoint.is_file():
                raise RuntimeError(f"simulation entrypoint is missing: {entrypoint}")
            self._host = host.strip()
            self._port = int(port)
            self._database_path = selected_database
            self._token = secrets.token_urlsafe(32)
            self._process = subprocess.Popen(
                [
                    node,
                    str(entrypoint),
                    "--host", self._host,
                    "--port", str(self._port),
                    "--token", self._token,
                    "--database", str(self._database_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=None,
            )
            self._started_at = time.monotonic()
            deadline = time.monotonic() + _START_TIMEOUT_SECONDS
            last_start_error = None
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    exit_code = self._process.returncode
                    self._clear_process()
                    raise RuntimeError(f"simulation child exited during startup with code {exit_code}")
                try:
                    # Verify authenticated ownership too: an unrelated process on
                    # this port must never be mistaken for our child.
                    health = self._request("GET", "/health", authenticated=False, timeout=0.25)
                    self._request('GET', '/games/startup-check/component-contract', timeout=0.25)
                    from .mission_terrain import MissionTerrainWorker
                    self._terrain_worker = MissionTerrainWorker(self)
                    self._terrain_worker.start()
                    log.info("headless simulation started at %s", self.base_url)
                    return {"started": True, "alreadyRunning": False, **self.status(), "health": health}
                except RuntimeError as error:
                    last_start_error = error
                    time.sleep(0.025)
            self.stop()
            raise RuntimeError("simulation child did not become healthy before timeout") from last_start_error

    def _clear_process(self) -> None:
        if self._terrain_worker is not None:
            self._terrain_worker.stop()
            self._terrain_worker = None
        self._process = None
        self._token = None
        self._started_at = None
        self._database_path = None

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._terrain_worker is not None:
                self._terrain_worker.stop()
            process = self._process
            if process is None or process.poll() is not None:
                self._clear_process()
                return {"stopped": False, "alreadyStopped": True, "running": False}
            process.terminate()
            try:
                process.wait(timeout=3.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
            exit_code = process.returncode
            log.info("headless simulation stopped with code %s", exit_code)
            self._clear_process()
            return {"stopped": True, "alreadyStopped": False, "running": False, "exitCode": exit_code}

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            result: dict[str, Any] = {
                "running": running,
                "host": self._host if running else None,
                "port": self._port if running else None,
                "url": self.base_url if running else None,
                "pid": self._process.pid if running else None,
                "uptimeSeconds": time.monotonic() - self._started_at if running and self._started_at is not None else None,
                "databasePath": str(self._database_path) if running and self._database_path is not None else None,
            }
            if running:
                try:
                    result["health"] = self._request("GET", "/health", authenticated=False)
                except RuntimeError as exc:
                    result["healthError"] = str(exc)
            return result

    def command(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("simulation service is not running; call ArcticSimulation.start first")
            return self._request(method, path, payload)


simulation_host = SimulationHost()
atexit.register(simulation_host.stop)


def shutdown_simulation_host() -> None:
    simulation_host.stop()
