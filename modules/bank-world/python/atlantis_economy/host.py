"""Own the local persistent bank process without owning Terrain's simulation tick."""
import atexit
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


class BankHost:
    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._url = None
        self._token = None
        self._database = None

    def _health(self):
        request = Request(self._url + "/health", headers={"Authorization": "Bearer " + self._token})
        with urlopen(request, timeout=0.5) as response:
            return json.load(response)

    def status(self):
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            result = {"running": running, "databasePath": self._database,
                      "pid": self._process.pid if running else None}
            if running:
                result["health"] = self._health()
            return result

    def start(self):
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.status()
            if self._process is not None:
                self.stop()  # Clear only this supervisor's stale credential after a child crash.
            root = os.environ.get("ATLANTIS_BANK_MODULE_ROOT")
            db = os.environ.get("ATLANTIS_BANK_DB_PATH")
            port = int(os.environ.get("ATLANTIS_BANK_PORT", "5192"))
            if not root or not db:
                raise RuntimeError("ATLANTIS_BANK_MODULE_ROOT and ATLANTIS_BANK_DB_PATH must be configured")
            entry = Path(root).resolve() / "apps/bank/src/server.js"
            database = Path(db)
            if not entry.is_file() or not database.is_absolute() or database.suffix != ".sqlite":
                raise ValueError("An existing bank module and absolute .sqlite database path are required")
            if not 1 <= port <= 65535:
                raise ValueError("Invalid bank port")
            node = shutil.which("node")
            if not node:
                raise RuntimeError("Node is required for the bank")
            if os.environ.get("GAME_BANK_URL") or os.environ.get("GAME_BANK_AUTHORITY_TOKEN"):
                raise RuntimeError("An external bank is already configured; local start will not replace it")
            self._token = secrets.token_urlsafe(48)
            self._url = f"http://127.0.0.1:{port}"
            self._database = str(database)
            env = {**os.environ, "ATLANTIS_BANK_PORT": str(port), "BANK_AUTHORITY_TOKEN": self._token}
            self._process = subprocess.Popen([node, str(entry)], env=env,
                                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    if self._process.poll() is not None:
                        raise RuntimeError(f"Bank exited at startup: {self._process.returncode}")
                    try:
                        health = self._health()
                    except (URLError, TimeoutError):
                        time.sleep(0.05)
                        continue
                    if health.get("service") != "atlantis-bank":
                        raise RuntimeError("Wrong service on bank port")
                    os.environ["GAME_BANK_URL"] = self._url + "/api/bank"
                    os.environ["GAME_BANK_AUTHORITY_TOKEN"] = self._token
                    return self.status()
                raise RuntimeError("Bank startup timed out")
            except Exception:
                self.stop()
                raise

    def stop(self):
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if self._token and os.environ.get("GAME_BANK_AUTHORITY_TOKEN") == self._token:
                os.environ.pop("GAME_BANK_AUTHORITY_TOKEN", None)
                os.environ.pop("GAME_BANK_URL", None)
            self._process = None
            self._token = None
            return {"running": False}


bank_host = BankHost()
atexit.register(bank_host.stop)
