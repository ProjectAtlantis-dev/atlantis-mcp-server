"""Incremental Atlantis port of the terrain HTTP service.

The active code at the top of this module is deliberately framework-neutral.
The complete source ``flaskserver/serve_flask.py`` snapshot follows as
comments, in its original order. Port one dependency slice at a time by moving
code out of that snapshot only after its imports and tests work in Atlantis.

Port order is owned by the source module:

1. wire-level query/result contracts and binary encoding (active);
2. configuration, logging, and database access (active);
3. DEM and coastline demand schedulers;
4. texture acquisition and retry scheduling;
5. framework-neutral ``collect_tiles`` orchestration;
6. Atlantis-visible entry points replacing Flask routes;
7. auxiliary building, texture, classifier, and diagnostic endpoints.

Do not expose a partial tile tool: the demand coordinator must remain disabled
until its database and scheduler dependencies have crossed together.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dynamic_functions.Terrain.terrain_config import (
    BOOTSTRAP_SEED_DEPTH,
    GREENLAND_BBOX,
    MAX_TILE_DEPTH,
    WMS_CONTRACT_DEPTH,
    WMS_TEXTURE_INSPECT_MIN_DEPTH,
    WMS_TEXTURE_PROBE_MAX_DEPTH,
)
from dynamic_functions.Terrain.tile_address import parse_tile_id as _parse_tile_id


log = logging.getLogger("terrain")
log_db = logging.getLogger("terrain.db")
log_tex = logging.getLogger("terrain.tex")
log_cog = logging.getLogger("terrain.cog")
log_d13 = logging.getLogger("terrain.d13")


TERRAIN_DIR = Path(__file__).resolve().parent
DATABASE_DIR = TERRAIN_DIR / "Database"
LOCAL_DB_PATH = DATABASE_DIR / "terrain.db"
CLIENT_LOG_PATH = TERRAIN_DIR / os.environ.get(
    "CLIENT_LOG_FILE", "client_debug.log"
)
DB_PATH = LOCAL_DB_PATH

# The old asset catalog and browser client-log viewer have not crossed yet.
# Keep their source configuration disabled until the corresponding consumers
# are ported; guessing new paths would silently point terrain work at the wrong
# database or frontend.
# TERRAIN_DB_PATH overrides are intentionally not ported: the Database tool
# owns both the fixed path and the one connection stored in server_shared.
# ASSETS_DB_PATH = Path(os.environ["ASSET_DB_PATH"]).expanduser().resolve()
# CLIENT_LOG_HTML_PATH = ...


def _env_int(name: str, default: int) -> int:
    """Read an integer environment setting with the source fallback behavior."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def shared_database_connection() -> sqlite3.Connection:
    """Return the connection owned by the existing Terrain Database tool.

    Schema initialization, commits, shutdown, and reload-safe lifetime all
    remain owned by ``Terrain/Database/database.py`` through
    ``atlantis.server_shared``. This module never opens a second connection.
    """

    from dynamic_functions.Terrain.Database.database import db

    return db()


# Lazily populated backend symbols. Keeping these declarations active is
# side-effect free and preserves the source bootstrap contract while their
# producer modules cross into Terrain one at a time.
_backend_ready = False
_backend_error: str | None = None
_np: Any = None
_Image: Any = None
_to_stereo: Any = None
_query_tiles_stereo: Any = None
_load_no_data_cache: Any = None
_GRID_N: int = 65
_tile_bbox: Any = None
_texture_ids_in: Any = None
_texture_sources_in: Any = None
_metatile_is_upsampled: Any = None
_read_texture: Any = None
_write_texture: Any = None
_fetch_sentinel2_texture: Any = None
_fetch_dataforsyningen_texture: Any = None
_split_texture_metatile: Any = None
_harmonize_texture_metatile: Any = None
_init_textures: Any = None
_init_classifier_tiles: Any = None


@dataclass(frozen=True)
class TileQuery:
    """Framework-neutral inputs for one terrain tile demand pass."""

    qx: float
    qy: float
    ox: float
    oy: float
    error: float
    max_depth: int
    demand_origin: str
    max_range: float
    lod_altitude: float = 0.0
    heading: float = 0.0
    binary_heightmaps: bool = False
    bathymetry: bool = True
    known_digests: dict[str, str] = field(default_factory=dict)
    previous_depth_cap: int | None = None


@dataclass
class TileResult:
    """Framework-neutral tile result before JSON or binary transport."""

    payload: dict[str, Any]
    heightmap_blobs: list[bytes] = field(default_factory=list)
    stages: list[tuple[str, float]] = field(default_factory=list)
    staged_until: float = 0.0


class TileQueryError(RuntimeError):
    """The terrain traversal failed before a result could be produced."""


def encode_tiles_binary(result: TileResult) -> bytes:
    """Encode the existing binary-v1 terrain response without Flask.

    Layout: ``uint32_le header_length``, padded JSON header, then float32
    sample blocks in ``payload["tiles"]`` order.
    """

    header = json.dumps(result.payload, separators=(",", ":")).encode("utf-8")
    header += b" " * (-(len(header) + 4) % 4)
    return b"".join(
        [struct.pack("<I", len(header)), header, *result.heightmap_blobs]
    )


# ---------------------------------------------------------------------------
# Disabled source snapshot
# ---------------------------------------------------------------------------
# Everything below is intentionally commented out. It is the migration queue,
# not executable code. Keep its order aligned with the source service so each
# newly enabled block reveals the next real dependency instead of inventing a
# parallel architecture.

# from __future__ import annotations
# 
# import base64
# import binascii
# import datetime
# import io
# import json
# import struct
# import logging
# import math
# import os
# import re
# import socket
# import sqlite3
# import threading
# import time
# import zlib
# import zipfile
# from concurrent.futures import ThreadPoolExecutor
# from dataclasses import dataclass, field
# from logging import FileHandler
# from pathlib import Path
# from typing import Any, cast
# 
# import asyncio
# 
# from colored_log import get_logger
# from buildings_query import buildings_for_tile_query, pack_buildings
# from classifier_job_control import (
#   ClassifierJobControl,
#   classifier_inventory,
#   regression_case_summaries,
# )
# from classifier_training_control import ClassifierTrainingControl
# from classifier_routes import (
#   ClassifierRouteDependencies,
#   classifier_bp,
#   configure_classifier_routes,
#   ensure_d12_class_map as _classifier_ensure_d12_class_map,
# )
# from gpu_profile_control import GpuProfileControl
# from terrain_config import (
#   BOOTSTRAP_SEED_DEPTH, MAX_TILE_DEPTH, WMS_CONTRACT_DEPTH,
#   WMS_TEXTURE_INSPECT_MIN_DEPTH, WMS_TEXTURE_PROBE_MAX_DEPTH,
# )
# from tile_address import parse_tile_id as _parse_tile_id
# from bathymetry_demand import (
#   BathymetryDemandScheduler,
#   terrain_request_max_depth,
#   terrain_request_origin,
# )
# 
# log = get_logger("terrain")
# log_db = get_logger("terrain.db")
# log_tex = get_logger("terrain.tex")
# log_cog = get_logger("terrain.cog")
# # Everything past WMS_CONTRACT_DEPTH: blowup inspection verdicts, procedural
# # cooks, deferrals. Every line carries running totals so detector-floor and
# # rate-limit tuning can be read straight off a tail of this channel.
# log_d13 = get_logger("terrain.d13")
# 
# ROOT = Path(__file__).resolve().parent.parent
# 
# from flask import Flask, Response, g, jsonify, request, send_from_directory
# DIST_DIR = ROOT / "webserver" / "dist"
# STATIC_DIR = str(DIST_DIR)
# CLIENT_LOG_HTML_PATH = ROOT / "webserver" / "client_log.html"
# 
# FLASK_DIR = Path(__file__).resolve().parent
# LOCAL_DB_PATH = FLASK_DIR / "terrain.db"
# CLIENT_LOG_PATH = FLASK_DIR / os.environ.get("CLIENT_LOG_FILE", "client_debug.log")
# ASSETS_DB_PATH = Path(
#   os.environ.get("ASSET_DB_PATH", ROOT / "assetserver" / "assets.db")
# ).expanduser().resolve()
# 
# 
# def _resolve_db_path() -> Path:
#   explicit = os.environ.get("TERRAIN_DB_PATH", "").strip()
#   if explicit:
#     return Path(explicit).expanduser().resolve()
#   return LOCAL_DB_PATH
# 
# 
# DB_PATH = _resolve_db_path()
# 
# app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
# _gpu_profile_control = GpuProfileControl()
# _classifier_job_control = ClassifierJobControl(DB_PATH)
# _classifier_training_control = ClassifierTrainingControl()
# 
# 
# def _env_int(name: str, default: int) -> int:
#   raw = os.environ.get(name)
#   if raw is None:
#     return default
#   try:
#     return int(raw)
#   except (TypeError, ValueError):
#     return default
# 
# 
# _bathymetry_demand = BathymetryDemandScheduler(
#   DB_PATH,
#   glacier_root=os.environ.get("GLACIER_ROOT") or None,
#   workers=max(1, _env_int("BATHYMETRY_WORKERS", 1)),
#   retry_seconds=max(1, _env_int("BATHYMETRY_RETRY_SECONDS", 300)),
#   retry_max_seconds=max(
#     1, _env_int("BATHYMETRY_RETRY_MAX_SECONDS", 3600)
#   ),
#   logger=log,
# )
# 
# 
# def _require_available_port(host: str, port: int) -> None:
#   """Abort startup before backend initialization when Flask cannot bind."""
#   family = socket.AF_INET6 if ":" in host else socket.AF_INET
#   try:
#     with socket.socket(family, socket.SOCK_STREAM) as probe:
#       probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#       probe.bind((host, port))
#   except OSError as exc:
#     raise SystemExit(
#       f"Flask startup aborted: cannot bind to {host}:{port}: {exc}"
#     ) from exc
# 
# 
# # In-memory ring buffer of raw client log entries for the HTML viewer.
# _client_log_ring = []        # list of dicts
# _CLIENT_LOG_RING_MAX = 2000
# _ws_clients: set = set()       # active async websocket connections
# _ws_loop: asyncio.AbstractEventLoop | None = None  # set when ws server starts
# _ws_queue: asyncio.Queue | None = None             # set when ws server starts
# 
# _client_log = logging.getLogger("terrain.client")
# if not _client_log.handlers:
#   CLIENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
#   _client_handler = FileHandler(
#     str(CLIENT_LOG_PATH),
#     mode="w",
#     encoding="utf-8",
#   )
#   class _ClientColorFormatter(logging.Formatter):
#     """Pretty-print JSON client logs with ANSI colors."""
#     _LEVEL_COLORS = {
#       logging.DEBUG:    "\033[36m",    # cyan
#       logging.INFO:     "\033[32m",    # green
#       logging.WARNING:  "\033[33m",    # yellow
#       logging.ERROR:    "\033[31m",    # red
#       logging.CRITICAL: "\033[1;31m",  # bold red
#     }
#     _RESET = "\033[0m"
#     _DIM = "\033[2m"
#     _BOLD = "\033[1m"
#     _PHASE_COLOR = "\033[35m"  # magenta
#     _KEY_COLOR = "\033[36m"    # cyan for JSON keys
#     _STR_COLOR = "\033[33m"    # yellow for string values
#     _NUM_COLOR = "\033[32m"    # green for numbers
#     _BOOL_COLOR = "\033[35m"   # magenta for booleans/null
#     # Regex to colorize JSON tokens in pretty-printed output
#     _JSON_TOKEN_RE = re.compile(
#       r'("(?:[^"\\]|\\.)*")\s*:'       # key followed by colon
#       r'|("(?:[^"\\]|\\.)*")'          # string value
#       r'|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)' # number
#       r'|\b(true|false|null)\b'         # boolean/null
#     )
# 
#     def _colorize_json(self, text):
#       def _repl(m):
#         if m.group(1) is not None:    # JSON key
#           return f"{self._KEY_COLOR}{m.group(1)}{self._RESET}:"
#         if m.group(2) is not None:    # string value
#           return f"{self._STR_COLOR}{m.group(2)}{self._RESET}"
#         if m.group(3) is not None:    # number
#           return f"{self._NUM_COLOR}{m.group(3)}{self._RESET}"
#         if m.group(4) is not None:    # bool/null
#           return f"{self._BOOL_COLOR}{m.group(4)}{self._RESET}"
#         return m.group(0)
#       return self._JSON_TOKEN_RE.sub(_repl, text)
# 
#     def format(self, record):
#       ts = self.formatTime(record, self.datefmt)
#       color = self._LEVEL_COLORS.get(record.levelno, "")
#       level = record.levelname
#       # Try to pretty-print the JSON message
#       msg = record.getMessage()
#       try:
#         obj = json.loads(msg)
#         phase = obj.pop("phase", "client.log")
#         pretty = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
#         pretty = self._colorize_json(pretty)
#         return (
#           f"{self._DIM}{ts}{self._RESET} "
#           f"{color}[{level}]{self._RESET} "
#           f"{self._PHASE_COLOR}{phase}{self._RESET} "
#           f"{pretty}"
#         )
#       except Exception as exc:
#         return (
#           f"{self._DIM}{ts}{self._RESET} {color}[{level}]{self._RESET} "
#           f"{msg} [formatter fallback: {type(exc).__name__}: {exc}]"
#         )
# 
#   _client_handler.setFormatter(_ClientColorFormatter())
#   _client_log.addHandler(_client_handler)
#   _client_log.setLevel(logging.DEBUG)
#   _client_log.propagate = False
# 
# 
# 
# # Lazily imported terrain backend symbols.
# _backend_ready = False
# _backend_error: str | None = None
# _np: Any = None
# _Image: Any = None
# _to_stereo: Any = None
# _query_tiles_stereo: Any = None
# _load_no_data_cache: Any = None
# _GRID_N: int = 65
# _tile_bbox: Any = None
# _texture_ids_in: Any = None
# _texture_sources_in: Any = None
# _metatile_is_upsampled: Any = None
# _read_texture: Any = None
# _write_texture: Any = None
# _fetch_sentinel2_texture: Any = None
# _fetch_dataforsyningen_texture: Any = None
# _split_texture_metatile: Any = None
# _harmonize_texture_metatile: Any = None
# _init_textures: Any = None
# _init_classifier_tiles: Any = None
# 
# _tex_pool = ThreadPoolExecutor(max_workers=4)
# _tex_fetching: set[str] = set()
# _tex_fetching_lock = threading.Lock()
# 
# # Coastline rasterization can take seconds across a fresh camera footprint.
# # It must never run in /api/tiles: camera topology owns request latency, while
# # derived masks are allowed to trail and trigger an ordinary polling refresh.
# _coastline_demand_lock = threading.Lock()
# _coastline_demand_pending: dict[str, tuple] = {}
# _coastline_demand_active: str | None = None
# _coastline_demand_thread: threading.Thread | None = None
# 
# 
# # Background mask repair yields to the interactive tile query between tiles.
# _COASTLINE_WORKER_YIELD_S = 0.05
# 
# 
# def _coastline_demand_worker() -> None:
#   global _coastline_demand_active, _coastline_demand_thread
#   from coastline import cache_official_water_mask
# 
#   while True:
#     with _coastline_demand_lock:
#       if not _coastline_demand_pending:
#         _coastline_demand_active = None
#         _coastline_demand_thread = None
#         return
#       tile_id = next(iter(_coastline_demand_pending))
#       bbox = _coastline_demand_pending.pop(tile_id)
#       _coastline_demand_active = tile_id
# 
#     db = sqlite3.connect(
#       str(DB_PATH), timeout=30.0, check_same_thread=False,
#     )
#     db.execute("PRAGMA busy_timeout=30000")
#     try:
#       cache_official_water_mask(db, tile_id, bbox, _GRID_N)
#       db.commit()
#     except sqlite3.OperationalError as exc:
#       # The next camera poll will requeue a still-visible tile. Do not let a
#       # transient bulk-ingest writer kill this worker or block topology.
#       log.warning(
#         f"[coastline-demand] {tile_id}: {exc}; deferred to next camera poll"
#       )
#       db.rollback()
#     finally:
#       db.close()
#       with _coastline_demand_lock:
#         if _coastline_demand_active == tile_id:
#           _coastline_demand_active = None
#     # Yield between masks. This worker is background repair; the camera's tile
#     # query is interactive and shares the same database. Draining flat out put
#     # 12+ mask writes into the gap between two polls and pushed query_tiles
#     # from ~50ms to 968ms.
#     time.sleep(_COASTLINE_WORKER_YIELD_S)
# 
# 
# def _schedule_coastline_demand(
#   db, targets: dict[str, tuple], camera: tuple[float, float] | None = None,
# ) -> tuple[int, int]:
#   """Replace unstarted mask work with the latest visible camera footprint.
# 
#   ``camera`` orders the queue nearest-first. Without it the worker drained an
#   arbitrary dict order, so it would grind through coarse depth-8 masks on the
#   far rim while the tiles under the aircraft waited behind them.
#   """
#   global _coastline_demand_pending, _coastline_demand_thread
#   if not targets:
#     with _coastline_demand_lock:
#       _coastline_demand_pending = {}
#       return int(_coastline_demand_active is not None), 0
# 
#   marks = ",".join("?" for _ in targets)
#   existing = {
#     row[0] for row in db.execute(
#       f"SELECT tile_id FROM coastline_masks WHERE tile_id IN ({marks})",
#       tuple(targets),
#     )
#   }
#   needed = [
#     (tile_id, bbox) for tile_id, bbox in targets.items()
#     if tile_id not in existing
#   ]
#   if camera is not None:
#     # Nearest first. The worker pops in insertion order, so ordering here is
#     # what decides whether the ground under the aircraft is masked before the
#     # far rim. Coarse tiles are enormous, so measure to the nearest point of
#     # the bbox rather than its centre — a depth-8 tile the camera sits inside
#     # has a centre kilometres away.
#     cx, cy = camera
# 
#     def _proximity(item):
#       _, (x0, y0, x1, y1) = item
#       dx = max(x0 - cx, 0.0, cx - x1)
#       dy = max(y0 - cy, 0.0, cy - y1)
#       return math.hypot(dx, dy)
# 
#     needed.sort(key=_proximity)
#   with _coastline_demand_lock:
#     _coastline_demand_pending = {
#       tile_id: bbox for tile_id, bbox in needed
#       if tile_id != _coastline_demand_active
#     }
#     if (
#       _coastline_demand_pending
#       and (_coastline_demand_thread is None
#            or not _coastline_demand_thread.is_alive())
#     ):
#       _coastline_demand_thread = threading.Thread(
#         target=_coastline_demand_worker,
#         name="coastline-demand",
#         daemon=True,
#       )
#       _coastline_demand_thread.start()
#     return (
#       int(_coastline_demand_active is not None),
#       len(_coastline_demand_pending),
#     )
# 
# # Cumulative inspected texture pipeline counters (process lifetime).
# _d13_stats = {
#   "inspected": 0,       # D11+ metatiles scored by the detector
#   "genuine": 0,         # verdict: real provider detail, stored as usual
#   "blowup": 0,          # verdict: provider upsample, routed to procedural cook
#   "cooked": 0,          # procedural cooks completed (parent quads)
#   "cook_children": 0,   # child textures written by cooks
#   "cook_skipped": 0,    # cook children skipped (terminal source already present)
#   "deferred": 0,        # cooks deferred because the parent texture is not final
#   "cook_failed": 0,     # cooks that raised (bad parent JPEG etc.)
# }
# _d13_stats_lock = threading.Lock()
# 
# 
# def _d13_count(**deltas) -> str:
#   """Apply counter deltas and return the totals string for log lines."""
#   with _d13_stats_lock:
#     for key, delta in deltas.items():
#       _d13_stats[key] += delta
#     return (
#       "totals: "
#       f"{_d13_stats['inspected']} inspected, "
#       f"{_d13_stats['genuine']} genuine, "
#       f"{_d13_stats['blowup']} blowups, "
#       f"{_d13_stats['cooked']} cooked "
#       f"({_d13_stats['cook_children']} children, {_d13_stats['cook_skipped']} skipped), "
#       f"{_d13_stats['deferred']} deferred, "
#       f"{_d13_stats['cook_failed']} failed"
#     )
# 
# 
# # A past-contract texture row still on a temporary source after this long,
# # with no worker on it and no retry queued, has fallen out of the workflow.
# _D13_STUCK_AFTER_S = 300.0
# _D13_WATCHDOG_INTERVAL_S = 120.0
# _D13_WATCHDOG_REQUEUE_CAP = 64
# _d13_watchdog_thread: threading.Thread | None = None
# _d13_last_distribution: dict[str, int] | None = None
# 
# 
# def _d13_texture_audit(db, now=None):
#   """Workflow-invariant audit of every past-contract texture row.
# 
#   Returns the source distribution plus the rows that violate the pipeline's
#   liveness invariant: temporary source, older than _D13_STUCK_AFTER_S, and
#   not in the fetching set or the retry queue — i.e. nothing will ever
#   upscale them unless someone re-demands the tile. These are the
#   "forgot to upscale" bugs; the watchdog requeues them.
#   """
#   if now is None:
#     now = datetime.datetime.now(datetime.timezone.utc)
#   rows = db.execute(
#     "SELECT tile_id, source, updated_at FROM textures "
#     "WHERE CAST(substr(tile_id, 1, instr(tile_id, '-') - 1) AS INTEGER) > ?",
#     (WMS_CONTRACT_DEPTH,),
#   ).fetchall()
#   with _tex_fetching_lock:
#     fetching = set(_tex_fetching)
#   with _tex_retry_lock:
#     retrying = set(_tex_retry_tiles)
# 
#   distribution: dict[str, int] = {}
#   stuck = []
#   for tile_id, source, updated_at in rows:
#     distribution[source] = distribution.get(source, 0) + 1
#     if source not in _TEX_TEMPORARY:
#       continue
#     if tile_id in fetching or tile_id in retrying:
#       continue
#     try:
#       age_s = (now - datetime.datetime.fromisoformat(updated_at)).total_seconds()
#     except (TypeError, ValueError):
#       age_s = float("inf")
#     if age_s >= _D13_STUCK_AFTER_S:
#       stuck.append((tile_id, source, age_s))
#   stuck.sort(key=lambda item: -item[2])
#   return {
#     "rows": len(rows),
#     "distribution": distribution,
#     "stuck": stuck,
#     "fetching": len(fetching),
#     "retrying": len(retrying),
#   }
# 
# 
# def _d13_watchdog_sweep(db) -> None:
#   global _d13_last_distribution
#   audit = _d13_texture_audit(db)
#   if audit["rows"] == 0:
#     return
# 
#   if audit["stuck"]:
#     requeue = audit["stuck"][:_D13_WATCHDOG_REQUEUE_CAP]
#     for tile_id, _source, _age_s in requeue:
#       parsed = _parse_tile_id(tile_id)
#       if parsed is not None:
#         _queue_texture_fetch(tile_id, _tile_bbox(*parsed))
#     sample = ", ".join(
#       f"{tile_id}({source}, {age_s / 60.0:.0f}min)"
#       for tile_id, source, age_s in requeue[:8]
#     )
#     log_d13.warning(
#       f"[watchdog] {len(audit['stuck'])} past-contract tiles STUCK on temporary "
#       f"sources with no in-flight work — requeued {len(requeue)} "
#       f"(cap {_D13_WATCHDOG_REQUEUE_CAP}): {sample}"
#       + ("…" if len(requeue) > 8 else "")
#     )
# 
#   if audit["distribution"] != _d13_last_distribution:
#     _d13_last_distribution = dict(audit["distribution"])
#     breakdown = ", ".join(
#       f"{source}={count}"
#       for source, count in sorted(audit["distribution"].items())
#     )
#     log_d13.info(
#       f"[watchdog] past-contract textures: {audit['rows']} rows ({breakdown}); "
#       f"{audit['fetching']} fetching, {audit['retrying']} in retry, "
#       f"{len(audit['stuck'])} stuck"
#     )
# 
# 
# def _d13_watchdog_worker() -> None:
#   while True:
#     time.sleep(_D13_WATCHDOG_INTERVAL_S)
#     db = None
#     try:
#       db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#       _d13_watchdog_sweep(db)
#     except Exception as exc:  # pragma: no cover - background sweep
#       log_d13.error(f"[watchdog] sweep FAILED: {type(exc).__name__}: {exc}")
#     finally:
#       if db is not None:
#         db.close()
# 
# 
# def _ensure_d13_watchdog() -> None:
#   global _d13_watchdog_thread
#   if _d13_watchdog_thread is not None and _d13_watchdog_thread.is_alive():
#     return
#   _d13_watchdog_thread = threading.Thread(target=_d13_watchdog_worker, daemon=True)
#   _d13_watchdog_thread.start()
#   log_d13.info(
#     f"[watchdog] started: sweep every {_D13_WATCHDOG_INTERVAL_S:.0f}s, "
#     f"stuck after {_D13_STUCK_AFTER_S:.0f}s, requeue cap {_D13_WATCHDOG_REQUEUE_CAP}"
#   )
# _tex_metatile_locks: dict[str, threading.Lock] = {}
# _tex_metatile_locks_guard = threading.Lock()
# 
# # --- Texture retry queue (transient Dataforsyningen failures) ---
# # A provider throttle is never evidence that coverage is absent. Retries stay
# # live with capped backoff until the provider succeeds or explicitly returns a
# # valid no-coverage frame.
# _TEX_RETRY_DELAYS = [30, 60, 120, 300]  # seconds between retries, then cap
# _tex_retry_queue: list[tuple[str, tuple, int]] = []  # (tile_id, bbox, attempt)
# _tex_retry_lock = threading.Lock()
# _tex_retry_tiles: set[str] = set()  # queued or currently sleeping/fetching
# _tex_retry_thread: threading.Thread | None = None
# 
# 
# def _tex_retry_enqueue(tile_id: str, bbox: tuple, attempt: int = 0) -> None:
#   with _tex_retry_lock:
#     if tile_id in _tex_retry_tiles:
#       return
#     _tex_retry_tiles.add(tile_id)
#     _tex_retry_queue.append((tile_id, bbox, attempt))
#     log_tex.debug(f"[tex-retry] enqueued {tile_id} attempt={attempt}")
#   _ensure_tex_retry_thread()
# 
# 
# def _tex_retry_finish(tile_id: str) -> None:
#   with _tex_retry_lock:
#     _tex_retry_tiles.discard(tile_id)
# 
# 
# def _tex_retry_again(tile_id: str, bbox: tuple, attempt: int) -> None:
#   """Requeue an active transient retry without opening a duplicate window."""
#   with _tex_retry_lock:
#     _tex_retry_queue.append((tile_id, bbox, attempt))
# 
# 
# def _ensure_tex_retry_thread() -> None:
#   global _tex_retry_thread
#   if _tex_retry_thread is not None and _tex_retry_thread.is_alive():
#     return
#   _tex_retry_thread = threading.Thread(target=_tex_retry_worker, daemon=True)
#   _tex_retry_thread.start()
# 
# 
# def _tex_retry_worker() -> None:
#   """Background thread that retries rate-limited Dataforsyningen fetches."""
#   while True:
#     with _tex_retry_lock:
#       if not _tex_retry_queue:
#         return  # thread exits, will be restarted when new items enqueue
#       tile_id, bbox, attempt = _tex_retry_queue.pop(0)
# 
#     delay = _TEX_RETRY_DELAYS[min(attempt, len(_TEX_RETRY_DELAYS) - 1)]
#     log_tex.info(
#       f"[tex-retry] {tile_id}: waiting {delay}s before provider attempt "
#       f"{attempt + 1}"
#     )
#     time.sleep(delay)
# 
#     db = None
#     try:
#       db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#       db.execute("PRAGMA journal_mode=WAL")
#       if _init_textures is not None:
#         _init_textures(db)
# 
#       # Check tile hasn't been upgraded while we waited
#       cur_row = db.execute(
#         "SELECT source, texture FROM textures WHERE tile_id = ?", (tile_id,)
#       ).fetchone()
#       if cur_row and cur_row[0] not in _METATILE_UPGRADEABLE_SOURCES:
#         log_tex.debug(f"[tex-retry] {tile_id}: already upgraded to {cur_row[0]}, skipping")
#         _tex_retry_finish(tile_id)
#         continue
# 
#       metatile_id, _, _, _ = _texture_metatile_spec(tile_id)
#       with _texture_metatile_lock(metatile_id):
#         children, fail_reason = _fetch_texture_metatile(tile_id)
#         if children is not None:
#           written, no_coverage = _store_texture_metatile(db, children)
#           if tile_id in written:
#             log_tex.info(f"[tex-retry] {tile_id}: SUCCESS on attempt {attempt + 1}")
#             _tex_retry_finish(tile_id)
#           elif tile_id in no_coverage:
#             _resolve_no_coverage(db, tile_id, cur_row[1] if cur_row else None, "[tex-retry]")
#             _tex_retry_finish(tile_id)
#           else:
#             # A sibling request may have filled the aligned metatile while
#             # this retry waited. The next demand will inspect the final row.
#             _tex_retry_finish(tile_id)
#         elif fail_reason == 'no_coverage':
#           _resolve_no_coverage(db, tile_id, cur_row[1] if cur_row else None, "[tex-retry]")
#           _tex_retry_finish(tile_id)
#         else:
#           # Still transient. Never convert a throttle/timeout into terminal
#           # no-coverage; retain the ancestor crop and capped retry loop.
#           _tex_retry_again(tile_id, bbox, attempt + 1)
#           log_tex.warning(
#             f"[tex-retry] {tile_id}: provider still transient; "
#             f"re-queued attempt {attempt + 2}"
#           )
#     except Exception as exc:
#       log_tex.error(f"[tex-retry] {tile_id}: FAILED: {type(exc).__name__}: {exc}")
#       _tex_retry_again(tile_id, bbox, attempt + 1)
#     finally:
#       if db is not None:
#         db.close()
# 
# 
# _cog_scheduler_lock = threading.RLock()
# _cog_fetching_tiles: set[str] = set()
# _cog_pending_tiles: dict[
#   str, tuple[tuple[float, float, float, float], str]
# ] = {}
# _cog_demand_ids: set[str] = set()
# _cog_fetched_total = 0      # lifetime count of COG tiles fetched from S3
# _cog_skipped_total = 0      # lifetime count of tiles skipped (already had data)
# _cog_already_fetched: set[str] = set()  # tile IDs we've already fetched this session
# _cog_synthetic_retry_at: dict[str, float] = {}
# _cog_failure_retry_at: dict[str, float] = {}
# _cog_failure_attempts: dict[str, int] = {}
# _COG_TILE_WORKERS = max(1, _env_int("COG_TILE_WORKERS", 6))
# _COG_SYNTHETIC_RETRY_SECONDS = max(
#   1, _env_int("COG_SYNTHETIC_RETRY_SECONDS", 300)
# )
# _COG_FAILURE_RETRY_BASE_SECONDS = max(
#   1, _env_int("COG_FAILURE_RETRY_BASE_SECONDS", 5)
# )
# _COG_FAILURE_RETRY_MAX_SECONDS = max(
#   _COG_FAILURE_RETRY_BASE_SECONDS,
#   _env_int("COG_FAILURE_RETRY_MAX_SECONDS", 300),
# )
# _cog_pool = ThreadPoolExecutor(max_workers=_COG_TILE_WORKERS)
# _cog_audit_lock = threading.Lock()
# 
# 
# def _request_timestamp() -> str:
#   return datetime.datetime.now(datetime.timezone.utc).isoformat()
# 
# 
# def _record_dem_requests(
#   db, missing, requested_at: str | None = None, *, origin: str = "viewer"
# ) -> int:
#   """Record that the terrain algorithm demanded these tile heightmaps."""
#   tile_ids = list(dict.fromkeys(tile_id for tile_id, _ in missing))
#   if not tile_ids:
#     return 0
#   timestamp = requested_at or _request_timestamp()
#   db.executemany(
#     "UPDATE tiles SET dem_demanded_at = ? WHERE tile_id = ?",
#     ((timestamp, tile_id) for tile_id in tile_ids),
#   )
#   db.commit()
#   for tile_id in tile_ids:
#     log_cog.info(
#       f"[tile-audit] tile={tile_id} stage=dem_demanded "
#       f"origin={origin} at={timestamp}"
#     )
#   return len(tile_ids)
# 
# 
# def _record_dem_request(
#   db, tile_id: str, requested_at: str | None = None, *, origin: str = "viewer"
# ) -> str:
#   """Persist proof that a DEM worker began processing this tile."""
#   timestamp = requested_at or _request_timestamp()
#   db.execute(
#     "UPDATE tiles SET dem_requested_at = ? WHERE tile_id = ?",
#     (timestamp, tile_id),
#   )
#   db.commit()
#   log_cog.info(
#     f"[tile-audit] tile={tile_id} stage=dem_requested "
#     f"origin={origin} at={timestamp}"
#   )
#   return timestamp
# 
# 
# def _record_cog_request(
#   tile_id: str, event: dict, *, origin: str = "viewer"
# ) -> None:
#   """Record every concrete provider COG attempt in the existing server log."""
#   timestamp = event.get("at") or _request_timestamp()
#   stage = event.get("stage", "cog_event")
#   provider = event.get("provider", "unknown")
#   outcome = event.get("outcome")
#   detail = event.get("detail")
#   parts = [
#     f"tile={tile_id}", f"stage={stage}", f"provider={provider}",
#     f"origin={origin}", f"at={timestamp}",
#   ]
#   if outcome is not None:
#     parts.append(f"outcome={outcome}")
#   if detail is not None:
#     parts.append(f"detail={json.dumps(detail, sort_keys=True, separators=(',', ':'))}")
#   log_cog.info(f"[tile-audit] {' '.join(parts)}")
#   if stage != "cog_requested":
#     return
#   # The provider callback can run in an ArcticDEM sub-worker. Use its own
#   # short connection and serialize this single-row proof update.
#   with _cog_audit_lock:
#     audit_db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#     try:
#       audit_db.execute(
#         "UPDATE tiles SET cog_requested_at = ? WHERE tile_id = ?",
#         (timestamp, tile_id),
#       )
#       audit_db.commit()
#     finally:
#       audit_db.close()
# 
# 
# def _tile_ancestor_ids(tile_id: str):
#   parsed = _parse_tile_id(tile_id)
#   if parsed is None:
#     return
#   depth, column, row = parsed
#   while depth > 0:
#     depth -= 1
#     column //= 2
#     row //= 2
#     yield f"{depth}-{column}-{row}"
# 
# 
# def _fetch_one_cog_tile(tile_id, bbox, origin="viewer"):
#   """Fetch and persist one tile; completion never waits for sibling tiles."""
#   from ingest import _read_cog_heightmap, _resample_from_parent
#   from database import CONFIDENCE, TileClobberError, write_tile
#   from serve import (
#     mark_no_data, _UPGRADEABLE_SOURCES, _cache_coastline,
#     _mark_official_ocean,
#   )
# 
#   db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   db.execute("PRAGMA journal_mode=WAL")
#   try:
#     # Past-contract heightmaps are DERIVED, never read from COG — a raw
#     # 10m/30m read on a 41-165m tile is interpolation mush whose edges
#     # ignore every neighbor (visible as raised square tile corners). This
#     # gate mirrors serve._fetch_tile; the cook defers until the measured
#     # parent chain exists, riding the scheduler's normal retry path.
#     parsed = _parse_tile_id(tile_id)
#     if parsed is not None and parsed[0] > WMS_CONTRACT_DEPTH:
#       from serve import _cook_cooked_dem_quad
# 
#       _record_dem_request(db, tile_id, origin=origin)
#       if _cook_cooked_dem_quad(
#         db, tile_id,
#       ):
#         log_cog.info(
#           f"[tile-audit] tile={tile_id} stage=dem_completed "
#           f"origin={origin} outcome=cooked"
#         )
#         return "fetched"
#       log_cog.info(
#         f"[tile-audit] tile={tile_id} stage=dem_completed "
#         f"origin={origin} outcome=cook_deferred"
#       )
#       # NOT "defer": that requeues and redispatches immediately, and a
#       # cook-defer is millisecond-fast — it would hot-loop a worker until
#       # the parent lands. This outcome waits for the next demand refresh.
#       return "cook_deferred"
#     row = db.execute(
#       "SELECT source FROM tiles WHERE tile_id = ?", (tile_id,)
#     ).fetchone()
#     upgrade = bool(row and row[0] in _UPGRADEABLE_SOURCES)
#     log_cog.debug(
#       f"[cog-worker] {tile_id}: reading "
#       f"bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
#     )
#     _record_dem_request(db, tile_id, origin=origin)
# 
#     # Coastline classification is cheaper and more authoritative than either
#     # elevation provider.  In particular, ArcticDEM publishes a technically
#     # valid flat sheet over parts of the ocean, so waiting until both COGs
#     # return nodata causes needless network traffic and stores false terrain.
#     try:
#       water = _cache_coastline(db, tile_id, bbox)
#     except Exception as exc:
#       water = None
#       log_cog.warning(
#         f"[cog-worker] {tile_id}: coastline preflight failed: "
#         f"{type(exc).__name__}: {exc}"
#       )
#     if water is not None and water.all():
#       log_cog.info(
#         f"[tile-audit] tile={tile_id} stage=dem_completed "
#         f"origin={origin} outcome=official_ocean"
#       )
#       _mark_official_ocean(db, tile_id)
#       return "fetched"
# 
#     data, src_name = _read_cog_heightmap(
#       bbox,
#       _GRID_N,
#       audit=lambda event: _record_cog_request(
#         tile_id, event, origin=origin
#       ),
#     )
# 
#     if data is None and src_name == "official_ocean":
#       log_cog.info(
#         f"[tile-audit] tile={tile_id} stage=dem_completed "
#         f"origin={origin} outcome=official_ocean provider=arcticdem"
#       )
#       _mark_official_ocean(db, tile_id)
#       return "fetched"
# 
#     if data is None:
#       # A child whose source COG is empty may depend on a parent currently in
#       # flight. Put it back into the live queue instead of blocking a worker or
#       # prematurely caching it as no-data.
#       with _cog_scheduler_lock:
#         ancestor_work = any(
#           ancestor in _cog_fetching_tiles or ancestor in _cog_pending_tiles
#           for ancestor in _tile_ancestor_ids(tile_id)
#         )
#       if ancestor_work:
#         log_cog.info(
#           f"[tile-audit] tile={tile_id} stage=dem_completed "
#           f"origin={origin} outcome=deferred"
#         )
#         return "defer"
# 
#       data, src_name = _resample_from_parent(
#         db, tile_id, bbox=None, resolution=_GRID_N
#       )
#       if data is None:
#         log_cog.info(
#           f"[tile-audit] tile={tile_id} stage=dem_completed "
#           f"origin={origin} outcome=no_data"
#         )
#         mark_no_data(db, tile_id)
#         return "no_data"
# 
#     source_name = src_name if isinstance(src_name, str) else "parent_resampled"
#     conf = CONFIDENCE.get(source_name, CONFIDENCE["arcticdem"])
#     cm = _np.where(_np.isnan(data), _np.uint8(0), _np.uint8(conf))
#     hm = _np.where(_np.isnan(data), 0.0, data).astype(_np.float32)
#     try:
#       write_tile(
#         db, tile_id, hm, cm, source_name, reconcile=False,
#         allow_overwrite=upgrade,
#       )
#       if source_name == "parent_resampled":
#         log_cog.info(f"[PARENT RESAMPLE] {tile_id}: filled from parent")
#       log_cog.info(
#         f"[tile-audit] tile={tile_id} stage=dem_completed "
#         f"origin={origin} outcome=stored source={source_name}"
#       )
#       return "synthetic" if source_name == "parent_resampled" else "fetched"
#     except TileClobberError:
#       log_cog.info(
#         f"[tile-audit] tile={tile_id} stage=dem_completed "
#         f"origin={origin} outcome=clobber_skipped"
#       )
#       return "skipped"
#   except Exception as exc:
#     log_cog.error(
#       f"[COG FETCH] {tile_id}: {type(exc).__name__}: {exc}"
#     )
#     log_cog.info(
#       f"[tile-audit] tile={tile_id} stage=dem_completed "
#       f"origin={origin} outcome=error error={type(exc).__name__}"
#     )
#     # An operational failure is not evidence that the provider has no data.
#     # Keep the tile eligible, but let the scheduler apply bounded backoff.
#     return "retry"
#   finally:
#     db.close()
# 
# 
# def _fill_cog_workers_locked():
#   while _cog_pending_tiles and len(_cog_fetching_tiles) < _COG_TILE_WORKERS:
#     tile_id = next(iter(_cog_pending_tiles))
#     bbox, origin = _cog_pending_tiles.pop(tile_id)
#     _cog_fetching_tiles.add(tile_id)
#     _cog_already_fetched.add(tile_id)
#     log_cog.info(
#       f"[tile-audit] tile={tile_id} stage=dem_dispatched origin={origin}"
#     )
#     future = _cog_pool.submit(_fetch_one_cog_tile, tile_id, bbox, origin)
#     future.add_done_callback(
#       lambda completed, tid=tile_id, tile_bbox=bbox, demand_origin=origin:
#         _finish_cog_tile(tid, tile_bbox, demand_origin, completed)
#     )
# 
# 
# def _finish_cog_tile(tile_id, bbox, origin, future):
#   global _cog_fetched_total, _cog_skipped_total
#   try:
#     outcome = future.result()
#   except Exception as exc:  # pragma: no cover - defensive executor boundary
#     log_cog.error(f"[COG worker] {tile_id}: {type(exc).__name__}: {exc}")
#     outcome = "retry"
# 
#   with _cog_scheduler_lock:
#     _cog_fetching_tiles.discard(tile_id)
#     if outcome == "defer":
#       # Dependency deferrals become eligible on the next demand refresh. Do
#       # not resubmit inside this completion callback: the dependency may still
#       # be running, which otherwise creates a millisecond-fast loop.
#       _cog_already_fetched.discard(tile_id)
#     elif outcome == "retry":
#       attempts = _cog_failure_attempts.get(tile_id, 0) + 1
#       delay = min(
#         _COG_FAILURE_RETRY_BASE_SECONDS * (2 ** min(attempts - 1, 20)),
#         _COG_FAILURE_RETRY_MAX_SECONDS,
#       )
#       _cog_failure_attempts[tile_id] = attempts
#       _cog_failure_retry_at[tile_id] = time.time() + delay
#       log_cog.warning(
#         f"[tile-audit] tile={tile_id} stage=dem_retry_scheduled "
#         f"origin={origin} attempt={attempts} delay_s={delay}"
#       )
#     elif outcome == "cook_deferred":
#       # Eligible again, but only the next demand refresh re-queues it —
#       # by then the measured parent this cook was waiting on may exist.
#       _cog_already_fetched.discard(tile_id)
#     elif outcome == "fetched":
#       _cog_synthetic_retry_at.pop(tile_id, None)
#       _cog_failure_retry_at.pop(tile_id, None)
#       _cog_failure_attempts.pop(tile_id, None)
#       _cog_fetched_total += 1
#     elif outcome == "synthetic":
#       _cog_failure_retry_at.pop(tile_id, None)
#       _cog_failure_attempts.pop(tile_id, None)
#       _cog_fetched_total += 1
#       _cog_synthetic_retry_at[tile_id] = (
#         time.time() + _COG_SYNTHETIC_RETRY_SECONDS
#       )
#       log_cog.info(
#         f"[tile-audit] tile={tile_id} stage=dem_retry_scheduled "
#         f"origin={origin} delay_s={_COG_SYNTHETIC_RETRY_SECONDS}"
#       )
#     elif outcome == "skipped":
#       _cog_failure_retry_at.pop(tile_id, None)
#       _cog_failure_attempts.pop(tile_id, None)
#       _cog_skipped_total += 1
#     elif outcome == "no_data":
#       _cog_failure_retry_at.pop(tile_id, None)
#       _cog_failure_attempts.pop(tile_id, None)
#     _fill_cog_workers_locked()
# 
# 
# def _schedule_cog_demand(
#   missing, visible_synthetic=(), *, origin="viewer"
# ):
#   """Replace unstarted work with the latest camera-priority ordering."""
#   global _cog_pending_tiles, _cog_demand_ids
#   missing_ids = {tile_id for tile_id, _ in missing}
#   with _cog_scheduler_lock:
#     now = time.time()
#     missing_ids = {tile_id for tile_id, _ in missing}
#     for tile_id, retry_at in list(_cog_failure_retry_at.items()):
#       if tile_id not in missing_ids or retry_at > now:
#         continue
#       _cog_failure_retry_at.pop(tile_id, None)
#       _cog_already_fetched.discard(tile_id)
#       log_cog.info(
#         f"[tile-audit] tile={tile_id} stage=dem_retry_due origin={origin}"
#       )
#     visible_retry_bboxes = dict(visible_synthetic)
#     due_retries = []
#     for tile_id, retry_at in list(_cog_synthetic_retry_at.items()):
#       if retry_at > now or tile_id not in visible_retry_bboxes:
#         continue
#       _cog_synthetic_retry_at.pop(tile_id, None)
#       _cog_already_fetched.discard(tile_id)
#       due_retries.append((tile_id, visible_retry_bboxes[tile_id]))
#       log_cog.info(
#         f"[tile-audit] tile={tile_id} stage=dem_retry_due origin={origin}"
#       )
#     demand = list(missing) + due_retries
#     _cog_demand_ids = {tile_id for tile_id, _ in demand}
#     _cog_pending_tiles = {
#       tile_id: (bbox, origin) for tile_id, bbox in demand
#       if tile_id not in _cog_already_fetched
#       and tile_id not in _cog_fetching_tiles
#       and _cog_failure_retry_at.get(tile_id, 0) <= now
#     }
#     _fill_cog_workers_locked()
#     return len(_cog_fetching_tiles), len(_cog_pending_tiles)
# 
# def _bootstrap_backend() -> None:
#   global _backend_ready, _backend_error
#   global _np, _Image, _to_stereo, _query_tiles_stereo, _load_no_data_cache
#   global _GRID_N, _tile_bbox, _texture_ids_in, _read_texture, _write_texture
#   global _fetch_sentinel2_texture, _fetch_dataforsyningen_texture, _split_texture_metatile
#   global _harmonize_texture_metatile
#   global _init_textures, _init_classifier_tiles
#   global _texture_sources_in, _metatile_is_upsampled
# 
#   if _backend_ready or _backend_error is not None:
#     return
# 
#   try:
#     import numpy as np  # type: ignore
#     from PIL import Image  # type: ignore
# 
#     from coords import to_stereo
#     from classifier.storage import init_classifier_tiles
#     from database import GRID_N, _tile_bbox as terrain_tile_bbox, seed_tiles, open_db
#     from serve import load_no_data_cache, query_tiles_stereo
#     from texture import (
#       fetch_dataforsyningen_texture,
#       fetch_sentinel2_texture,
#       harmonize_texture_metatile,
#       init_textures,
#       metatile_is_upsampled,
#       read_texture,
#       split_texture_metatile,
#       texture_ids_in,
#       texture_sources_in,
#       write_texture,
#     )
# 
#     db = open_db(str(DB_PATH))
# 
#     tile_count = db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
#     if tile_count == 0:
#       seed_depth = max(0, min(BOOTSTRAP_SEED_DEPTH, MAX_TILE_DEPTH))
#       log_db.info(
#         f"Empty tiles table — fast bootstrap seed to depth {seed_depth} "
#         f"(target ceiling depth {MAX_TILE_DEPTH})..."
#       )
#       seed_tiles(db, max_depth=seed_depth)
#       tile_count = db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
#       log_db.info(f"Seeded {tile_count} tile skeletons at bootstrap depth {seed_depth}.")
#     else:
#       hm_count = db.execute("SELECT COUNT(*) FROM tiles WHERE heightmap IS NOT NULL").fetchone()[0]
#       log_db.info(f"Tile grid already seeded: {tile_count} skeletons, {hm_count} populated with heightmaps")
# 
#     # Keep traversal ceiling metadata at full target depth, even if bootstrap
#     # seeded fewer levels initially.
#     cur_max = db.execute("SELECT value FROM metadata WHERE key = 'max_depth'").fetchone()
#     if cur_max is None or int(cur_max[0]) < MAX_TILE_DEPTH:
#       db.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('max_depth', ?)", (str(MAX_TILE_DEPTH),))
#       db.commit()
#       log_db.info(f"Updated max_depth metadata to {MAX_TILE_DEPTH}")
# 
#     # Source-name migration (2026-07-23): the cook pipeline lost its noise
#     # painter, so the legacy 'fractal_*' provenance strings were renamed to
#     # 'cooked_*'. Rewrite old rows in place — idempotent, and it must run
#     # BEFORE the recipe-version gate below so its queries see one spelling.
#     db.execute(
#       "UPDATE tiles SET source = 'cooked_dem' WHERE source = 'fractal_dem'"
#     )
#     try:
#       db.execute(
#         "UPDATE textures SET source = 'cooked_upscale' "
#         "WHERE source = 'fractal_upscale'"
#       )
#     except sqlite3.OperationalError as exc:
#       if "no such table" not in str(exc).lower():
#         raise
#       # Fresh database — textures table not created yet.
#     db.commit()
# 
#     # Cooked DEMs and cooked deep textures are derived artifacts: when the
#     # derived terrain / texture recipe changes, every cooked_dem tile
#     # and cooked_upscale texture is stale by definition. Reset DEMs to
#     # pending skeletons (payload dropped, seams invalidated) and drop the
#     # texture rows so normal demand recooks both with the current recipe —
#     # d13 recooks from the measured d12 surface, deeper depths re-cascade.
#     from terrain_upscale import MACRO_TERRAIN_VERSION
#     from terrain_seams import invalidate_tile_seams
#     macro_version = db.execute(
#       "SELECT value FROM metadata WHERE key = 'macro_terrain_version'"
#     ).fetchone()
#     if macro_version is None or int(macro_version[0]) != MACRO_TERRAIN_VERSION:
#       stale = [row[0] for row in db.execute(
#         "SELECT tile_id FROM tiles WHERE source = 'cooked_dem'"
#       ).fetchall()]
#       if stale:
#         now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
#         db.execute(
#           "UPDATE tiles SET source = 'pending', heightmap = NULL, "
#           "confidence_map = NULL, geometric_error = 0, updated_at = ? "
#           "WHERE source = 'cooked_dem'",
#           (now_iso,),
#         )
#         for stale_id in stale:
#           invalidate_tile_seams(db, stale_id)
#       stale_textures = 0
#       try:
#         stale_textures = db.execute(
#           "SELECT COUNT(*) FROM textures WHERE source = 'cooked_upscale'"
#         ).fetchone()[0]
#         if stale_textures:
#           db.execute("DELETE FROM textures WHERE source = 'cooked_upscale'")
#       except sqlite3.OperationalError as exc:
#         if "no such table" not in str(exc).lower():
#           raise
#         # Fresh database — textures table not created yet.
#       db.execute(
#         "INSERT OR REPLACE INTO metadata (key, value) "
#         "VALUES ('macro_terrain_version', ?)",
#         (str(MACRO_TERRAIN_VERSION),),
#       )
#       db.commit()
#       log_db.info(
#         f"Derived terrain recipe v{MACRO_TERRAIN_VERSION}: reset {len(stale)} "
#         f"cooked_dem tiles and dropped {stale_textures} "
#         f"cooked_upscale textures for recook"
#       )
# 
#     no_data_count = load_no_data_cache(db)
# 
#     db.close()
# 
#     _np = np
#     _Image = Image
#     _to_stereo = to_stereo
#     _GRID_N = GRID_N
#     _tile_bbox = terrain_tile_bbox
#     _query_tiles_stereo = query_tiles_stereo
#     _load_no_data_cache = load_no_data_cache
#     _texture_ids_in = texture_ids_in
#     _texture_sources_in = texture_sources_in
#     _metatile_is_upsampled = metatile_is_upsampled
#     _read_texture = read_texture
#     _write_texture = write_texture
#     _fetch_sentinel2_texture = fetch_sentinel2_texture
#     _fetch_dataforsyningen_texture = fetch_dataforsyningen_texture
#     _split_texture_metatile = split_texture_metatile
#     _harmonize_texture_metatile = harmonize_texture_metatile
#     _init_textures = init_textures
#     _init_classifier_tiles = init_classifier_tiles
# 
#     _backend_ready = True
#     log.info(f"Terrain backend ready. DB={DB_PATH}")
#     log_db.info(f"No-data cache: {no_data_count} tiles")
# 
#     _ensure_d13_watchdog()
# 
#     import grundkort
#     grundkort.ensure_grundkort_async()
#     grundkort.enable_demand()
# 
#     from ingest_coastline import ensure_gtk50_blocks
#     threading.Thread(target=ensure_gtk50_blocks, daemon=True).start()
# 
#     # Blocks beyond the seed set are acquired as tiles ask for them, so flying
#     # into new territory no longer needs anyone to predeclare its coastline.
#     import gtk50_demand
#     gtk50_demand.enable()
# 
#   except Exception as exc:  # pragma: no cover - runtime setup path
#     _backend_error = (
#       f"Terrain backend unavailable: {type(exc).__name__}: {exc}. "
#       "Install dependencies into ./venv and ensure pipeline modules are present."
#     )
#     log.error(_backend_error)
# 
# 
# def _terrain_unavailable_response(status: int = 503):
#   _bootstrap_backend()
#   if _backend_ready:
#     return None
#   return jsonify({"error": _backend_error or "Terrain backend unavailable"}), status
# 
# 
# 
# def _get_db() -> sqlite3.Connection:
#   db = cast(sqlite3.Connection | None, g.get("terrain_db"))
#   if db is None:
#     # WAL is established once during backend bootstrap. Reissuing the journal
#     # mode pragma on every request takes a schema/write lock and made camera
#     # LOD requests fail while GTK50 was committing derived masks.
#     db = sqlite3.connect(
#       str(DB_PATH), timeout=30.0, check_same_thread=False,
#     )
#     db.execute("PRAGMA busy_timeout=30000")
#     g.terrain_db = db
#   return db
# 
# 
# @app.teardown_appcontext
# def _close_db(exc: BaseException | None) -> None:
#   db = cast(sqlite3.Connection | None, g.pop("terrain_db", None))
#   if db is not None:
#     db.close()
#   assets_db = cast(sqlite3.Connection | None, g.pop("assets_db", None))
#   if assets_db is not None:
#     assets_db.close()
# 
# 
# def _get_assets_db() -> sqlite3.Connection:
#   db = cast(sqlite3.Connection | None, g.get("assets_db"))
#   if db is None:
#     from asset_catalog import connect
#     db = connect(ASSETS_DB_PATH)
#     g.assets_db = db
#   return db
# 
# 
# 
# def _arg_float(name: str, default: float) -> float:
#   raw = request.args.get(name)
#   if raw is None:
#     return default
#   try:
#     return float(raw)
#   except (TypeError, ValueError):
#     return default
# 
# 
# 
# OCEAN_MAX_ELEV_M = 0.5
# 
# 
# def _is_ocean_tile(db, tile_id: str) -> bool:
#   """True when the tile's heightmap is entirely at/below sea level.
# 
#   Tiles where both DEMs came up empty ('no_data') are open ocean too.
#   Unknown (no row / no heightmap yet) → False; a later texture request
#   re-checks once the heightmap has been ingested.
#   """
#   row = db.execute(
#     "SELECT source, heightmap FROM tiles WHERE tile_id = ?", (tile_id,)
#   ).fetchone()
#   if row is None:
#     return False
#   source, hm_blob = row
#   from coastline import read_water_mask
#   official_water = read_water_mask(db, tile_id)
#   if official_water is not None:
#     return bool(_np.all(official_water))
#   if source == 'no_data':
#     return True
#   if hm_blob is None:
#     return False
#   hm = _np.frombuffer(zlib.decompress(hm_blob), dtype=_np.float32)
#   if _np.all(_np.isnan(hm)):
#     return True
#   return float(_np.nanmax(hm)) <= OCEAN_MAX_ELEV_M
# 
# 
# def _repair_white_ocean_jpeg(db, tile_id: str, jpeg: bytes) -> bytes:
#   """Fill WMS white no-data pixels over ocean (per heightmap) with OCEAN_RGB.
# 
#   Coastal frames arrive with real land + white sea fill and pass the
#   whole-frame white check. No-op when the tile has no heightmap yet.
#   """
#   from texture import repair_white_ocean
#   row = db.execute(
#     "SELECT heightmap FROM tiles WHERE tile_id = ?", (tile_id,)
#   ).fetchone()
#   if not row or row[0] is None:
#     return jpeg
#   hm = _np.frombuffer(zlib.decompress(row[0]), dtype=_np.float32).reshape(_GRID_N, _GRID_N)
#   from coastline import effective_heightmap
#   hm = effective_heightmap(db, tile_id, hm)
#   repaired = repair_white_ocean(jpeg, hm)
#   if repaired is not None:
#     log_tex.info(f"[tex-repair] {tile_id}: filled white ocean pixels with OCEAN_RGB")
#     return repaired
#   return jpeg
# 
# 
# def _resolve_no_coverage(db, tile_id: str, existing_jpeg, log_prefix: str) -> None:
#   """Terminal-state a tile Dataforsyningen has no imagery for.
# 
#   Ocean tiles get a flat deep-water texture. Land tiles keep their ancestor
#   crop, stamped ancestor_crop_nodata. A white ancestor crop is poison (a
#   pre-filter WMS no-data frame that got cached as imagery) — delete it so it
#   can't keep getting served.
#   """
#   from texture import is_white_fill_jpeg, ocean_texture_jpeg
# 
#   if _is_ocean_tile(db, tile_id):
#     _write_texture(db, tile_id, ocean_texture_jpeg(), "ocean_nodata")
#     log_tex.info(f"{log_prefix} {tile_id}: no coverage + ocean heightmap → ocean_nodata")
#   elif existing_jpeg and not is_white_fill_jpeg(existing_jpeg):
#     _write_texture(db, tile_id, existing_jpeg, "ancestor_crop_nodata")
#     log_tex.info(f"{log_prefix} {tile_id}: no coverage → ancestor_crop_nodata")
#   elif existing_jpeg:
#     db.execute("DELETE FROM textures WHERE tile_id = ?", (tile_id,))
#     db.commit()
#     log_tex.warning(f"{log_prefix} {tile_id}: no coverage, dropped white-fill ancestor crop")
#   else:
#     log_tex.info(f"{log_prefix} {tile_id}: no coverage, no fallback available yet")
# 
# 
# def _nearest_ancestor_texture(tile_id: str, texture_ids: set[str]) -> str | None:
#   parsed = _parse_tile_id(tile_id)
#   if parsed is None:
#     return None
#   d, c, r = parsed
#   while d > 0:
#     d -= 1
#     c //= 2
#     r //= 2
#     ancestor_id = f"{d}-{c}-{r}"
#     if ancestor_id in texture_ids:
#       return ancestor_id
#   return None
# 
# 
# def _crop_child_from_parent(parent_img, parent_depth, child_d, child_c, child_r):
#   """Crop a single child tile's quadrant from a parent/ancestor image.
# 
#   Returns JPEG bytes.
#   """
#   w, h = parent_img.size
#   depth_diff = child_d - parent_depth
#   sub_c = child_c % (1 << depth_diff)
#   sub_r = child_r % (1 << depth_diff)
#   n = 1 << depth_diff
# 
#   x0 = sub_c * w // n
#   x1 = (sub_c + 1) * w // n
#   y0 = (n - 1 - sub_r) * h // n
#   y1 = (n - sub_r) * h // n
# 
#   cropped = parent_img.crop((x0, y0, x1, y1)).resize((256, 256), _Image.Resampling.BILINEAR)
#   buf = io.BytesIO()
#   cropped.save(buf, format="JPEG", quality=85)
#   return buf.getvalue()
# 
# 
# def _seed_children_from_parent(db, parent_id: str) -> int:
#   """Crop a parent texture into 4 child quadrants and write as ancestor_crop.
# 
#   Only writes children that have no texture yet. Returns count of children seeded.
#   """
#   parent_tex = _read_texture(db, parent_id)
#   if parent_tex is None:
#     return 0
# 
#   parsed = _parse_tile_id(parent_id)
#   if parsed is None:
#     return 0
#   p_d, p_c, p_r = parsed
# 
#   parent_img = _Image.open(io.BytesIO(parent_tex))
#   from texture import is_white_fill
#   if is_white_fill(_np.asarray(parent_img.convert("RGB"), dtype=_np.uint8)):
#     log_tex.warning(f"[tex-seed] {parent_id}: white-fill parent, refusing to seed children")
#     return 0
#   child_depth = p_d + 1
#   seeded = 0
# 
#   for dc in range(2):
#     for dr in range(2):
#       child_c = p_c * 2 + dc
#       child_r = p_r * 2 + dr
#       child_id = f"{child_depth}-{child_c}-{child_r}"
# 
#       existing = db.execute(
#         "SELECT source FROM textures WHERE tile_id = ?", (child_id,)
#       ).fetchone()
#       if existing:
#         continue
# 
#       jpeg = _crop_child_from_parent(parent_img, p_d, child_depth, child_c, child_r)
#       jpeg = _repair_white_ocean_jpeg(db, child_id, jpeg)
#       _write_texture(db, child_id, jpeg, "ancestor_crop")
#       seeded += 1
# 
#   if seeded:
#     log_tex.debug(f"[tex-seed] {parent_id}: seeded {seeded}/4 children with ancestor_crop")
#   return seeded
# 
# 
# 
# def _texture_flags(
#   tile_id: str,
#   texture_ids: set[str],
#   fetching_ids: set[str],
# ) -> dict[str, str | bool | None]:
#   has_exact = tile_id in texture_ids
#   ancestor_id = None if has_exact else _nearest_ancestor_texture(tile_id, texture_ids)
#   has_any = has_exact or ancestor_id is not None
#   is_fetching = tile_id in fetching_ids
# 
#   if has_exact:
#     status = "ready"
#   elif ancestor_id is not None:
#     status = "ancestor_fallback"
#   elif is_fetching:
#     status = "fetching"
#   else:
#     status = "missing"
# 
#   return {
#     "status": status,
#     "has_texture": has_exact,
#     "available": has_any,
#     "is_placeholder": ancestor_id is not None,
#     "ancestor_id": ancestor_id,
#     "is_fetching": is_fetching,
#   }
# 
# 
# 
# _METATILE_FINAL_SOURCE = "dataforsyningen_metatile4h2"
# # Past WMS_CONTRACT_DEPTH, tiles whose metatile came back as a provider
# # blowup are cooked from the parent's final texture instead. Terminal, same
# # standing as _METATILE_FINAL_SOURCE; the value itself records provenance.
# _COOKED_SOURCE = "cooked_upscale"
# _METATILE_UPGRADEABLE_SOURCES = {
#   "sentinel2_crop",
#   "ancestor_crop",
#   "ancestor_crop_ratelimit",
#   "dataforsyningen",
#   "dataforsyningen_metatile",
#   "dataforsyningen_metatile4",
#   "dataforsyningen_metatile4h",
# }
# 
# # Sources that are temporary placeholders — serve them but let client re-fetch.
# _TEX_TEMPORARY = {
#   "sentinel2_crop",
#   "ancestor_crop",
#   "ancestor_crop_ratelimit",
#   "dataforsyningen",  # legacy independent fetch; upgrade to aligned metatile
#   "dataforsyningen_metatile",  # legacy 2x2 group; upgrade to aligned 4x4
#   "dataforsyningen_metatile4",  # legacy 4x4 group without harmonization
#   "dataforsyningen_metatile4h",  # legacy conservative harmonization
# }
# 
# 
# def _texture_metatile_spec(tile_id: str) -> tuple[str, tuple, int, dict[str, tuple[int, int]]]:
#   """Return aligned fetch information for the 4x4 group containing tile_id."""
#   parsed = _parse_tile_id(tile_id)
#   if parsed is None:
#     raise ValueError(f"invalid tile id: {tile_id}")
#   depth, column, row = parsed
#   if depth == 0:
#     return tile_id, tuple(_tile_bbox(depth, column, row)), 256, {tile_id: (0, 0)}
# 
#   depth_step = min(2, depth)
#   grid_size = 1 << depth_step
#   parent_depth = depth - depth_step
#   parent_column = column // grid_size
#   parent_row = row // grid_size
#   metatile_id = f"{parent_depth}-{parent_column}-{parent_row}"
#   children = {
#     f"{depth}-{parent_column * grid_size + column_bit}-{parent_row * grid_size + row_bit}":
#       (column_bit, row_bit)
#     for column_bit in range(grid_size)
#     for row_bit in range(grid_size)
#   }
#   return (
#     metatile_id,
#     tuple(_tile_bbox(parent_depth, parent_column, parent_row)),
#     256 * grid_size,
#     children,
#   )
# 
# 
# def _texture_metatile_lock(metatile_id: str) -> threading.Lock:
#   with _tex_metatile_locks_guard:
#     return _tex_metatile_locks.setdefault(metatile_id, threading.Lock())
# 
# 
# def _fetch_texture_metatile(tile_id: str) -> tuple[dict[str, bytes] | None, str | None]:
#   """Fetch and split one aligned metatile, without touching the database."""
#   _, bbox, resolution, child_offsets = _texture_metatile_spec(tile_id)
#   jpeg, fail_reason = _fetch_dataforsyningen_texture(
#     list(bbox), resolution=resolution, lossless=resolution > 256
#   )
#   if jpeg is None:
#     return None, fail_reason
# 
#   # From D11 onward the provider may fill a nominally finer request with a
#   # blowup of a D9/D10 image. Inspect the finest-octave energy before trusting
#   # it; a blowup routes the caller to the explicit cooked_upscale state.
#   parsed = _parse_tile_id(tile_id)
#   if parsed is not None and parsed[0] >= WMS_TEXTURE_INSPECT_MIN_DEPTH:
#     from texture import METATILE_RECON_MIN, METATILE_SPECTRAL_MIN
# 
#     inspect_started = time.perf_counter()
#     cheated, reconstruction, spectral = _metatile_is_upsampled(jpeg)
#     inspect_ms = (time.perf_counter() - inspect_started) * 1000.0
#     if cheated:
#       tripped = []
#       if reconstruction < METATILE_RECON_MIN:
#         tripped.append(
#           f"recon {reconstruction:.3f} < {METATILE_RECON_MIN} (reconstructible from half-res)"
#         )
#       if spectral < METATILE_SPECTRAL_MIN:
#         tripped.append(
#           f"spectral {spectral:.3f} < {METATILE_SPECTRAL_MIN} (outer frequency band empty)"
#         )
#       totals = _d13_count(inspected=1, blowup=1)
#       log_d13.info(
#         f"{tile_id}: BLOWUP metatile {bbox[2] - bbox[0]:.0f}m/{resolution}px "
#         f"({len(jpeg)}b, {inspect_ms:.0f}ms) — {'; '.join(tripped)} "
#         f"→ procedural cook | {totals}"
#       )
#       return None, "wms_upsampled"
#     totals = _d13_count(inspected=1, genuine=1)
#     log_d13.info(
#       f"{tile_id}: GENUINE metatile {bbox[2] - bbox[0]:.0f}m/{resolution}px "
#       f"({len(jpeg)}b, {inspect_ms:.0f}ms) recon={reconstruction:.3f} "
#       f"spectral={spectral:.3f} (floors {METATILE_RECON_MIN}/{METATILE_SPECTRAL_MIN}) "
#       f"→ storing provider detail | {totals}"
#     )
# 
#   if resolution == 256:
#     return {tile_id: jpeg}, None
#   jpeg = _harmonize_texture_metatile(
#     jpeg,
#     child_resolution=256,
#     grid_size=resolution // 256,
#   )
#   quadrants = _split_texture_metatile(
#     jpeg,
#     child_resolution=256,
#     grid_size=resolution // 256,
#   )
#   return {
#     child_id: quadrants[offset]
#     for child_id, offset in child_offsets.items()
#   }, None
# 
# 
# def _store_texture_metatile(db, children: dict[str, bytes]) -> tuple[set[str], set[str]]:
#   """Store valid metatile children without clobbering terminal/final states."""
#   from texture import is_no_coverage_fill_jpeg
# 
#   written = set()
#   no_coverage = set()
#   for child_id, jpeg in children.items():
#     if is_no_coverage_fill_jpeg(jpeg):
#       no_coverage.add(child_id)
#       continue
#     existing = db.execute(
#       "SELECT source FROM textures WHERE tile_id = ?", (child_id,)
#     ).fetchone()
#     if existing and existing[0] not in _METATILE_UPGRADEABLE_SOURCES:
#       continue
#     jpeg = _repair_white_ocean_jpeg(db, child_id, jpeg)
#     _write_texture(db, child_id, jpeg, _METATILE_FINAL_SOURCE)
#     written.add(child_id)
#   return written, no_coverage
# 
# 
# def _partition_texture_parent_blowups(db, children: dict[str, bytes]):
#   """Separate complete child quads that add no detail over their exact parent."""
#   from texture import PARENT_DETAIL_MIN, texture_quad_adds_parent_detail
# 
#   accepted = dict(children)
#   groups: dict[str, dict[tuple[int, int], tuple[str, bytes]]] = {}
#   for child_id, jpeg in children.items():
#     parsed = _parse_tile_id(child_id)
#     if parsed is None or parsed[0] < WMS_TEXTURE_INSPECT_MIN_DEPTH:
#       continue
#     depth, column, row = parsed
#     parent_id = f"{depth - 1}-{column // 2}-{row // 2}"
#     groups.setdefault(parent_id, {})[(column % 2, row % 2)] = (child_id, jpeg)
# 
#   blowups = []
#   for parent_id, group in groups.items():
#     if len(group) != 4:
#       continue
#     parent = db.execute(
#       "SELECT texture, source FROM textures WHERE tile_id = ?", (parent_id,)
#     ).fetchone()
#     if parent is None or parent[0] is None or parent[1] in _TEX_TEMPORARY:
#       continue
#     adds_detail, scores = texture_quad_adds_parent_detail(
#       {offset: child[1] for offset, child in group.items()}, parent[0]
#     )
#     score_text = ", ".join(
#       f"{group[offset][0]}={scores[offset]:.4f}" for offset in sorted(scores)
#     )
#     if adds_detail:
#       log_d13.info(
#         f"{parent_id}: PARENT DETAIL genuine — {score_text}; "
#         f"at least one child >= {PARENT_DETAIL_MIN:.2f}"
#       )
#       continue
#     child_ids = [child[0] for child in group.values()]
#     for child_id in child_ids:
#       accepted.pop(child_id, None)
#     representative = min(child_ids)
#     blowups.append((representative, parent_id, scores))
#     log_d13.info(
#       f"{parent_id}: PARENT BLOWUP — {score_text}; "
#       f"no child >= {PARENT_DETAIL_MIN:.2f} → procedural cook"
#     )
#   return accepted, blowups
# 
# 
# 
# def _cook_texture_quad(db, tile_id: str) -> bool:
#   """Cook tile_id's parent quad from the parent's final texture.
# 
#   Enlarges the depth-(d-1) parent texture (plain deterministic Lanczos — the
#   noise painter is REMOVED, it damaged tiles with dark shadow
#   artifacts) and writes all four depth-d children with source
#   "cooked_upscale" — the provenance record the client and DB keep. Returns
#   False when the parent texture is missing or still temporary; the parent
#   fetch is queued and the caller should retry this tile later.
#   """
#   from database import read_tile
#   from terrain_upscale import upscale_texture
# 
#   parsed = _parse_tile_id(tile_id)
#   if parsed is None:
#     log_d13.warning(f"{tile_id}: procedural cook refused — unparseable tile id")
#     return False
#   depth, column, row = parsed
#   if depth == 0:
#     log_d13.warning(f"{tile_id}: procedural cook refused — root tile has no parent")
#     return False
#   if depth > MAX_TILE_DEPTH:
#     log_d13.info(
#       f"{tile_id}: procedural cook refused — depth {depth} beyond "
#       f"MAX_TILE_DEPTH={MAX_TILE_DEPTH} (upscaling disabled)"
#     )
#     return False
#   cook_started = time.perf_counter()
#   parent_column, parent_row = column // 2, row // 2
#   parent_id = f"{depth - 1}-{parent_column}-{parent_row}"
#   parent = db.execute(
#     "SELECT texture, source FROM textures WHERE tile_id = ?", (parent_id,)
#   ).fetchone()
#   parent_bbox = list(_tile_bbox(depth - 1, parent_column, parent_row))
#   if parent is None or parent[0] is None or parent[1] in _TEX_TEMPORARY:
#     reason = (
#       "no parent texture row"
#       if parent is None
#       else "parent texture blob is empty"
#       if parent[0] is None
#       else f"parent source '{parent[1]}' is temporary"
#     )
#     totals = _d13_count(deferred=1)
#     log_d13.info(
#       f"{tile_id}: cook DEFERRED — {reason}; queued parent {parent_id} fetch, "
#       f"child retries later | {totals}"
#     )
#     _queue_texture_fetch(parent_id, tuple(parent_bbox))
#     return False
# 
#   upscale_started = time.perf_counter()
#   # Plain Lanczos enlarge of the parent photo. Procedural painters are absent
#   # because they quantized continuous source evidence into square decisions.
#   upscaled, _size = upscale_texture(parent[0], factor=2)
# 
#   shading = "shade=none"
# 
#   # Classification does NOT happen at every cook depth (user directive,
#   # 2026-07-22): there's no ground truth past the WMS contract depth, and
#   # d13-d16 imagery is procedurally upscaled — increasingly synthetic, with no
#   # new real information for a fresh classification to key off. The design
#   # follows the CLASSIFIER LADDER through real imagery (d8-d12, via
#   # _ensure_d12_class_map's live classification), and let every deeper
#   # depth INHERIT that label via the ancestor-walk already built into
#   # /api/classifier/<id>.png (crop + nearest-neighbor resize down from the
#   # nearest real ancestor). Writing a fresh classifier_tiles row at every
#   # cook depth — which this function used to do — defeats that walk before
#   # it starts: an exact-depth row always wins over an inherited one, so
#   # every deep tile was silently reclassifying itself from blurry
#   # synthetic imagery instead of inheriting the one classification that
#   # was ever backed by real ground truth.
#   upscale_ms = (time.perf_counter() - upscale_started) * 1000.0
#   quadrants = _split_texture_metatile(
#     upscaled, child_resolution=256, grid_size=2
#   )
#   written, skipped = [], []
#   for column_bit in range(2):
#     for row_bit in range(2):
#       child_id = f"{depth}-{parent_column * 2 + column_bit}-{parent_row * 2 + row_bit}"
#       existing = db.execute(
#         "SELECT source FROM textures WHERE tile_id = ?", (child_id,)
#       ).fetchone()
#       if existing and existing[0] not in _METATILE_UPGRADEABLE_SOURCES:
#         skipped.append(f"{child_id}({existing[0]})")
#         continue
#       _write_texture(db, child_id, quadrants[(column_bit, row_bit)], _COOKED_SOURCE)
#       written.append(child_id)
#       # No classifier_tiles write here — see the ladder-inheritance
#       # note above. Deep tiles read their classification through the
#       # ancestor walk in /api/classifier/<id>.png instead of storing their
#       # own.
#   cook_ms = (time.perf_counter() - cook_started) * 1000.0
#   totals = _d13_count(
#     cooked=1, cook_children=len(written), cook_skipped=len(skipped)
#   )
#   log_d13.info(
#     f"{tile_id}: COOKED quad from {parent_id} "
#     f"({parent[1]}, {len(parent[0])}b, lanczos {shading}) — "
#     f"wrote {len(written)} as {_COOKED_SOURCE}"
#     + (f", kept {', '.join(skipped)}" if skipped else "")
#     + f" in {cook_ms:.0f}ms (upscale {upscale_ms:.0f}ms) | {totals}"
#   )
#   return True
# 
# 
# def _queue_texture_fetch(
#   tile_id: str,
#   bbox: tuple[float, float, float, float],
# ) -> None:
#   with _tex_fetching_lock:
#     if tile_id in _tex_fetching:
#       return
#     _tex_fetching.add(tile_id)
# 
#   _RE_FETCHABLE = _METATILE_UPGRADEABLE_SOURCES - {"ancestor_crop_ratelimit"}
# 
#   def _worker() -> None:
#     db = None
#     try:
#       db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#       db.execute("PRAGMA journal_mode=WAL")
#       if _init_textures is not None:
#         _init_textures(db)
# 
#       # Skip if already in a non-refetchable state.
#       existing = db.execute(
#         "SELECT source FROM textures WHERE tile_id = ?", (tile_id,)
#       ).fetchone()
#       if existing and existing[0] not in _RE_FETCHABLE:
#         log_tex.debug(f"[tex-worker] {tile_id}: already cached ({existing[0]}), skipping")
#         return
# 
#       # Step 1: seed this tile + siblings from parent texture so all 4
#       # children get an immediate ancestor_crop baseline.
#       parsed = _parse_tile_id(tile_id)
#       if parsed is not None:
#         d, c, r = parsed
#         if d > 0:
#           parent_id = f"{d - 1}-{c // 2}-{r // 2}"
#           _seed_children_from_parent(db, parent_id)
# 
#       # Step 2: fetch the aligned grandparent extent once and split all exact
#       # children from the same reprojection. Concurrent sibling workers share
#       # this lock and observe the first worker's writes instead of refetching.
#       metatile_id, metatile_bbox, _, _ = _texture_metatile_spec(tile_id)
#       with _texture_metatile_lock(metatile_id):
#         existing = db.execute(
#           "SELECT source FROM textures WHERE tile_id = ?", (tile_id,)
#         ).fetchone()
#         if existing and existing[0] not in _RE_FETCHABLE:
#           # A sibling holding this lock already finished the group — via
#           # metatile store or procedural cook. Without this check every queued
#           # sibling re-downloads the 2MB metatile and re-cooks for nothing.
#           log_tex.debug(
#             f"[tex-worker] {tile_id}: sibling already filled it ({existing[0]})"
#           )
#           return
# 
#         if parsed is not None and parsed[0] > WMS_TEXTURE_PROBE_MAX_DEPTH:
#           # Beyond the measured WMS ceiling every metatile comes back as a
#           # blowup; skip the provider round-trip and take the cook branch
#           # directly, same as an inspected blowup would.
#           log_tex.debug(
#             f"[tex-worker] {tile_id}: depth {parsed[0]} beyond WMS probe "
#             f"ceiling {WMS_TEXTURE_PROBE_MAX_DEPTH} — cooking directly"
#           )
#           children, fail_reason = None, "wms_upsampled"
#         else:
#           log_tex.debug(
#             f"[tex-worker] {tile_id}: fetching metatile {metatile_id} bbox={metatile_bbox}"
#           )
#           children, fail_reason = _fetch_texture_metatile(tile_id)
#         if children is not None:
#           children, parent_blowups = _partition_texture_parent_blowups(db, children)
#           written, no_coverage = _store_texture_metatile(db, children)
#           log_tex.debug(
#             f"[tex-worker] {tile_id}: metatile wrote {len(written)} children"
#           )
#           for representative, _parent_id, _scores in parent_blowups:
#             try:
#               cooked = _cook_texture_quad(db, representative)
#             except Exception as exc:
#               totals = _d13_count(cook_failed=1)
#               log_d13.error(
#                 f"{representative}: parent-relative cook FAILED "
#                 f"({type(exc).__name__}: {exc}) — queued for retry | {totals}"
#               )
#               cooked = False
#             if not cooked:
#               parsed_child = _parse_tile_id(representative)
#               retry_bbox = (
#                 tuple(_tile_bbox(*parsed_child)) if parsed_child is not None else bbox
#               )
#               _tex_retry_enqueue(representative, retry_bbox, attempt=0)
#           if tile_id in no_coverage:
#             existing = db.execute(
#               "SELECT texture FROM textures WHERE tile_id = ?", (tile_id,)
#             ).fetchone()
#             _resolve_no_coverage(
#               db, tile_id, existing[0] if existing else None, "[tex-worker]"
#             )
#         elif fail_reason == 'no_coverage':
#           # Permanent for the requested child. Sibling requests can still try
#           # their group if provider coverage behavior changes at the boundary.
#           existing = db.execute(
#             "SELECT texture FROM textures WHERE tile_id = ?", (tile_id,)
#           ).fetchone()
#           _resolve_no_coverage(db, tile_id, existing[0] if existing else None, "[tex-worker]")
#         elif fail_reason == 'wms_upsampled':
#           # Provider blowup at an inspected depth — cook the parent quad
#           # with the procedural upscaler at the same finality as a real fetch.
#           # Every non-cooked outcome re-enters the retry queue: a tile must
#           # never leave this branch with no future work scheduled.
#           try:
#             cooked = _cook_texture_quad(db, tile_id)
#           except Exception as exc:
#             totals = _d13_count(cook_failed=1)
#             log_d13.error(
#               f"{tile_id}: procedural cook FAILED "
#               f"({type(exc).__name__}: {exc}) — queued for retry | {totals}"
#             )
#             cooked = False
#           if not cooked:
#             _tex_retry_enqueue(tile_id, bbox, attempt=0)
#         else:
#           # Transient (rate limit / timeout). Mark and enqueue for retry.
#           existing = db.execute(
#             "SELECT texture FROM textures WHERE tile_id = ?", (tile_id,)
#           ).fetchone()
#           if existing and existing[0]:
#             _write_texture(db, tile_id, existing[0], "ancestor_crop_ratelimit")
#           _tex_retry_enqueue(tile_id, bbox, attempt=0)
#           log_tex.debug(
#             f"[tex-worker] {tile_id}: transient → ancestor_crop_ratelimit, queued for retry"
#           )
#     except Exception as exc:  # pragma: no cover - runtime fetch path
#       log_tex.error(f"[tex-worker] {tile_id} FAILED: {type(exc).__name__}: {exc}")
#     finally:
#       if db is not None:
#         db.close()
#       with _tex_fetching_lock:
#         _tex_fetching.discard(tile_id)
# 
#   _tex_pool.submit(_worker)
# 
# 
# _api_tiles_state: dict[str, str | None] = {"last_result": None}
# # Only explain requests that are actually slow; normal flight is ~50ms.
# _TILES_SLOW_LOG_MS = 300.0
# # Downloading GTK50 packages is slow, so keep prefetch well ahead of what is
# # rendered — the data should already be cached by the time you fly into it.
# _GTK50_PREFETCH_RANGE_M = 25000.0
# 
# 
# @dataclass(frozen=True)
# class TileQuery:
#   """Everything the tile answer depends on, already decoded from the wire."""
# 
#   qx: float
#   qy: float
#   # Render origin the client wants bboxes rebased against; defaults to qx/qy.
#   ox: float
#   oy: float
#   error: float
#   # Only an untagged viewer request may drive terrain/coastline below the
#   # depth-12 data contract. Bathymetry's synthetic camera sweep exists solely
#   # to acquire solve/carve prerequisites and must never populate d13+ tiles.
#   max_depth: int
#   demand_origin: str
#   max_range: float
#   # LOD uses height above local terrain, not absolute altitude.
#   lod_altitude: float = 0.0
#   # Sent by the client and carried here as part of the request contract; the
#   # traversal below is a heading-independent circle and does not read it.
#   heading: float = 0.0
#   binary_heightmaps: bool = False
#   bathymetry: bool = True
#   # Residency map: tile id -> digest the client already holds and can rebuild
#   # from. Matching tiles ship as metadata only. Measured at 80% of transferred
#   # elevation bytes on a real flight, all of it re-decoded and re-allocated by
#   # the browser once a second.
#   known_digests: dict[str, str] = field(default_factory=dict)
#   # Echoed back from the previous response so the altitude cap has something
#   # to be hysteretic against.
#   previous_depth_cap: int | None = None
# 
# 
# @dataclass
# class TileResult:
#   """A tile answer, before any particular encoding is chosen."""
# 
#   payload: dict
#   # Float32 sample blocks, in ``payload["tiles"]`` order. Only populated for
#   # binary queries; JSON queries carry their samples inside the payload.
#   heightmap_blobs: list[bytes] = field(default_factory=list)
#   stages: list[tuple[str, float]] = field(default_factory=list)
#   # ``perf_counter`` at the last recorded stage, so a caller that encodes the
#   # payload can time that step continuously with the stages above.
#   staged_until: float = 0.0
# 
# 
# class TileQueryError(RuntimeError):
#   """The LOD traversal failed; the caller decides how to report that."""
# 
# 
# def collect_tiles(db, query: TileQuery) -> TileResult:
#   """Answer a camera position: the visible tiles plus the demand they schedule.
# 
#   Free of Flask so tools, tests, and synthetic sweeps can drive the same
#   traversal and scheduling without standing up a request context.
#   """
#   # Stage timings, reported only when a request is slow enough to be worth
#   # explaining. Normal flight is ~50ms and logs nothing; the occasional
#   # multi-second outlier says where it went instead of leaving us guessing.
#   _t_stages: list[tuple[str, float]] = []
#   _t_last = [time.perf_counter()]
# 
#   def _mark(name: str) -> None:
#     now = time.perf_counter()
#     _t_stages.append((name, (now - _t_last[0]) * 1000.0))
#     _t_last[0] = now
# 
#   qx, qy = query.qx, query.qy
#   ox, oy = query.ox, query.oy
#   error = query.error
#   max_depth = query.max_depth
#   demand_origin = query.demand_origin
#   max_range = query.max_range
#   lod_altitude = query.lod_altitude
#   binary_heightmaps = query.binary_heightmaps
#   known_digests = query.known_digests
# 
#   try:
#     tiles, missing = _query_tiles_stereo(
#       db,
#       qx,
#       qy,
#       error_threshold=error,
#       max_depth=max_depth,
#       max_range=max_range,
#       altitude=lod_altitude,
#       previous_depth=query.previous_depth_cap,
#       log=lambda msg: log.debug(f"[/api/tiles] {msg}"),
#     )
#   except Exception as exc:
#     log.error(f"[/api/tiles] query FAILED: {type(exc).__name__}: {exc}")
#     raise TileQueryError(
#       f"tile query failed: {type(exc).__name__}: {exc}"
#     ) from exc
# 
#   _mark("query_tiles")
#   # A global derived reset retains origin DEMs, so those tiles do not pass
#   # through the COG worker that normally performs coastline preflight. Make
#   # visible camera demand restore the missing contract-depth masks explicitly.
#   # Deep derived tiles still preflight their exact mask when they recook.
#   coastline_targets = {}
#   for tile in tiles:
#     if tile.get("heightmap") is None:
#       continue
#     depth, column, row = _parse_tile_id(tile["id"]) or (0, 0, 0)
#     if depth > WMS_CONTRACT_DEPTH:
#       shift = depth - WMS_CONTRACT_DEPTH
#       depth, column, row = (
#         WMS_CONTRACT_DEPTH, column >> shift, row >> shift,
#       )
#     target_id = f"{depth}-{column}-{row}"
#     coastline_targets[target_id] = tuple(_tile_bbox(depth, column, row))
# 
#   coastline_active, coastline_pending = _schedule_coastline_demand(
#     db, coastline_targets, camera=(qx, qy),
#   )
#   _mark("coastline_sched")
# 
#   _tiles_result_key = f"{len(tiles)}:{len(missing)}:{qx:.0f}:{qy:.0f}"
#   if _api_tiles_state["last_result"] != _tiles_result_key:
#     _api_tiles_state["last_result"] = _tiles_result_key
#     log.info(f"[/api/tiles] result: {len(tiles)} tiles, {len(missing)} missing (qx={qx:.0f} qy={qy:.0f})")
#   else:
#     log.debug(f"[/api/tiles] result: {len(tiles)} tiles, {len(missing)} missing")
# 
#   all_tile_ids = [t["id"] for t in tiles if t["heightmap"] is not None]
#   check_ids = set(all_tile_ids)
#   for tid in all_tile_ids:
#     d, c, r = _parse_tile_id(tid) or (0, 0, 0)
#     while d > 0:
#       d -= 1
#       c //= 2
#       r //= 2
#       check_ids.add(f"{d}-{c}-{r}")
# 
#   _mark("ancestor_ids")
#   texture_sources = _texture_sources_in(db, list(check_ids))
#   _mark("texture_sources")
#   texture_ids = set(texture_sources)
#   with _tex_fetching_lock:
#     tex_fetching = list(_tex_fetching)
#   tex_fetching_set = set(tex_fetching)
# 
#   # Base64 inflates float32 elevations by a third and forces the browser to
#   # parse megabytes of JSON on its main thread before a frame can render.
#   # Binary clients get a length-prefixed header plus one concatenated block of
#   # float32 samples, which they can view without copying.
#   heightmap_blobs = []
#   tile_reused = 0
#   tile_data = []
#   tex_status_counts = {
#     "ready": 0,
#     "ancestor_fallback": 0,
#     "fetching": 0,
#     "missing": 0,
#   }
#   for tile in tiles:
#     hm = tile["heightmap"]
#     if hm is None:
#       continue
# 
#     bbox = tile["bbox"]
#     tid = tile["id"]
#     tex_flags = _texture_flags(tid, texture_ids, tex_fetching_set)
#     tex_status = str(tex_flags["status"])
#     if tex_status in tex_status_counts:
#       tex_status_counts[tex_status] += 1
# 
#     hm_bytes = hm.astype(_np.float32).tobytes()
#     if binary_heightmaps:
#       # The client keys geometry reuse and seam-repair detection off heightmap
#       # identity. Base64 gave that for free as a string compare; binary needs
#       # an explicit digest, and it must cover the repaired bytes actually sent
#       # rather than any stored revision — edge repair depends on which
#       # neighbours are in *this* response.
#       heightmap_field = f"{zlib.crc32(hm_bytes) & 0xFFFFFFFF:08x}"
#       # The digest must cover the repaired bytes for *this* response, so the
#       # work above still happens; what is saved is the transfer and the
#       # browser-side allocation, not the server-side read.
#       if known_digests.get(tid) == heightmap_field:
#         heightmap_bytes_field = 0
#         tile_reused += 1
#       else:
#         heightmap_bytes_field = len(hm_bytes)
#         heightmap_blobs.append(hm_bytes)
#     else:
#       heightmap_field = base64.b64encode(hm_bytes).decode("ascii")
#       heightmap_bytes_field = None
# 
#     tile_data.append(
#       {
#         "id": tid,
#         "bbox": [bbox[0] - ox, bbox[1] - oy, bbox[2] - ox, bbox[3] - oy],
#         "depth": tile["depth"],
#         "source": tile["source"],
#         "resolution": _GRID_N,
#         "heightmap": heightmap_field,
#         "heightmapBytes": heightmap_bytes_field if binary_heightmaps else None,
#         "hasTexture": bool(tex_flags["has_texture"]),
#         "texAvailable": bool(tex_flags["available"]),
#         "texStatus": tex_status,
#         "texSource": texture_sources.get(tid),
#         "texIsPlaceholder": bool(tex_flags["is_placeholder"]),
#         "texAncestorId": tex_flags["ancestor_id"],
#         "texIsFetching": bool(tex_flags["is_fetching"]),
#       }
#     )
# 
# 
#   _mark("pack_tiles")
#   missing_data = []
#   for tid, bbox in missing:
#     missing_data.append(
#       {
#         "id": tid,
#         "bbox": [bbox[0] - ox, bbox[1] - oy, bbox[2] - ox, bbox[3] - oy],
#       }
#     )
# 
#   # Continuously feed free workers. Unstarted work is replaced by every new
#   # camera-priority list, so there is neither a wave barrier nor a stale queue.
#   _record_dem_requests(db, missing, origin=demand_origin)
#   visible_synthetic = [
#     (tile["id"], tile["bbox"])
#     for tile in tiles
#     if tile.get("source") == "parent_resampled"
#   ]
#   active_cog, pending_cog = _schedule_cog_demand(
#     missing, visible_synthetic, origin=demand_origin
#   )
#   _mark("dem_cog_sched")
#   # A parent fallback is useful immediately but is only eligible for another
#   # provider attempt after the scheduler cooldown. Tell the browser whether
#   # DEM work is actionable now and, if not, when it should wake. Without this
#   # distinction it polls every few seconds throughout the 300-second backoff.
#   missing_ids = {tile_id for tile_id, _ in missing}
#   with _cog_scheduler_lock:
#     now = time.time()
#     visible_synthetic_ids = {tile_id for tile_id, _ in visible_synthetic}
#     delayed_visible_retries = {
#       tile_id: retry_at
#       for tile_id, retry_at in _cog_synthetic_retry_at.items()
#       if tile_id in visible_synthetic_ids and retry_at > now
#     }
#     delayed_failure_retries = {
#       tile_id: retry_at
#       for tile_id, retry_at in _cog_failure_retry_at.items()
#       if tile_id in missing_ids and retry_at > now
#     }
#   dem_actionable = bool(active_cog or pending_cog) or any(
#     tile_id not in delayed_failure_retries for tile_id in missing_ids
#   )
#   delayed_retry_times = [
#     *delayed_visible_retries.values(), *delayed_failure_retries.values(),
#   ]
#   dem_retry_after_ms = (
#     max(1_000, int((min(delayed_retry_times) - now) * 1_000))
#     if delayed_retry_times and not dem_actionable
#     else None
#   )
#   if missing:
#     log_cog.debug(
#       f"[COG scheduler] {active_cog} active, {pending_cog} priority-queued"
#     )
#   with _cog_scheduler_lock:
#     downloading = list(_cog_fetching_tiles)
# 
#   # Bathymetry follows the same visible-demand loop, but only after a real
#   # depth-12 DEM and authoritative coastline mask exist. The scheduler
#   # coalesces eligible fjord/coastal tiles at depth 8 and ignores open ocean.
#   bathymetry_submitted = []
#   if query.bathymetry:
#     bathymetry_submitted = _bathymetry_demand.schedule(db, all_tile_ids)
#   bathymetry_status = _bathymetry_demand.status()
#   _mark("bathymetry_sched")
# 
#   # Asiaq publishes one vector archive per settlement. Only real viewer
#   # movement may acquire one; synthetic bathymetry sweeps must not populate
#   # unrelated villages. The background worker ingests and syncs the package,
#   # and the browser's normal polling observes the changed building hash.
#   grundkort_scheduled = []
#   grundkort_inflight = []
#   if demand_origin == "viewer":
#     import grundkort
#     grundkort_scheduled = grundkort.request_for_point(qx, qy)
#     grundkort_inflight = grundkort.demand_status()["inflight"]
#     import gtk50_demand
#     gtk50_demand.request_for_bbox((
#       qx - _GTK50_PREFETCH_RANGE_M, qy - _GTK50_PREFETCH_RANGE_M,
#       qx + _GTK50_PREFETCH_RANGE_M, qy + _GTK50_PREFETCH_RANGE_M,
#     ))
# 
#   _mark("grundkort_sched")
#   payload = {
#       "tiles": tile_data,
#       "missing": missing_data,
#       "downloading": downloading,
#       "demActionable": dem_actionable,
#       "demRetryAfterMs": dem_retry_after_ms,
#       "qx": qx,
#       "qy": qy,
#       "ox": ox,
#       "oy": oy,
#       "texCached": len(texture_ids),
#       "texFetching": len(tex_fetching),
#       "texQueued": len(tex_fetching),
#       "texRetryQueue": len(_tex_retry_queue),
#       "coastlineQueued": coastline_active + coastline_pending,
#       "texStatusCounts": tex_status_counts,
#       "bathymetryQueued": bathymetry_submitted,
#       "bathymetryStatus": bathymetry_status,
#       "grundkortQueued": sorted(set(grundkort_scheduled + grundkort_inflight)),
#       "tilesReused": tile_reused,
#       # Echoed back next request so the altitude cap has something to be
#       # hysteretic against; without it every response re-decides from scratch.
#       "depthCap": max((t["depth"] for t in tile_data), default=None),
#   }
#   return TileResult(payload, heightmap_blobs, _t_stages, _t_last[0])
# 
# 
# def encode_tiles_binary(result: TileResult) -> bytes:
#   """[uint32 LE header length][header JSON + pad][float32 samples, tile order]
# 
#   The header is padded to a 4-byte boundary so the sample block starts
#   aligned: a browser cannot create a Float32Array view over an unaligned
#   offset, and copying to realign would give back the win. Trailing spaces
#   are insignificant whitespace to any JSON parser. Every block is a whole
#   number of float32s, so each following block stays aligned too.
#   """
#   header = json.dumps(result.payload, separators=(",", ":")).encode("utf-8")
#   header += b" " * (-(len(header) + 4) % 4)
#   return b"".join([
#     struct.pack("<I", len(header)), header, *result.heightmap_blobs,
#   ])
# 
# 
# def _tile_query_from_request() -> TileQuery:
#   """Decode a /api/tiles request. The only place that touches ``request``."""
#   if "sx" in request.args and "sy" in request.args:
#     qx = _arg_float("sx", 0.0)
#     qy = _arg_float("sy", 0.0)
#   else:
#     lat = _arg_float("lat", 64.175)
#     lon = _arg_float("lon", -51.7388)
#     qx, qy = _to_stereo(lat, lon)
# 
#   known_digests: dict[str, str] = {}
#   previous_depth_cap = None
#   if request.method == "POST":
#     body = request.get_json(silent=True) or {}
#     candidate = body.get("known")
#     if isinstance(candidate, dict):
#       known_digests = candidate
#     held = body.get("depthCap")
#     if isinstance(held, int):
#       previous_depth_cap = held
# 
#   return TileQuery(
#     qx=qx,
#     qy=qy,
#     ox=_arg_float("ox", qx),
#     oy=_arg_float("oy", qy),
#     error=_arg_float("error", 0.0005),
#     max_depth=terrain_request_max_depth(request.args, MAX_TILE_DEPTH),
#     demand_origin=terrain_request_origin(request.args),
#     max_range=_arg_float("range", 16000.0),
#     # ``alt`` is retained as a fallback for older clients, but current
#     # clients send the unambiguous ``agl``.
#     lod_altitude=_arg_float("agl", _arg_float("alt", 0.0)),
#     heading=_arg_float("heading", 0.0),
#     binary_heightmaps=request.args.get("format") == "binary",
#     bathymetry=request.args.get("bathymetry", "1") != "0",
#     known_digests=known_digests,
#     previous_depth_cap=previous_depth_cap,
#   )
# 
# 
# @app.get("/api/tiles")
# @app.post("/api/tiles")
# def api_tiles():
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
# 
#   query = _tile_query_from_request()
#   log.debug(
#     f"[/api/tiles] qx={query.qx:.0f} qy={query.qy:.0f} error={query.error} "
#     f"maxDepth={query.max_depth} range={query.max_range:.0f} "
#     f"params={dict(request.args)}"
#   )
# 
#   try:
#     result = collect_tiles(_get_db(), query)
#   except TileQueryError as exc:
#     return jsonify({"error": str(exc)}), 500
# 
#   if not query.binary_heightmaps:
#     return jsonify(result.payload)
# 
#   body = encode_tiles_binary(result)
#   sample_bytes = sum(len(b) for b in result.heightmap_blobs)
#   header_bytes = len(body) - sample_bytes - 4
#   stages = [
#     *result.stages,
#     ("serialize", (time.perf_counter() - result.staged_until) * 1000.0),
#   ]
#   _total_ms = sum(ms for _, ms in stages)
#   if _total_ms >= _TILES_SLOW_LOG_MS:
#     log.warning(
#       "[/api/tiles SLOW] total=%.0fms  %s  | tiles=%d reused=%d "
#       "header=%.2fMB samples=%.2fMB",
#       _total_ms,
#       "  ".join(f"{n}={ms:.0f}" for n, ms in stages if ms >= 1.0),
#       len(result.payload["tiles"]), result.payload["tilesReused"],
#       header_bytes / 1e6, sample_bytes / 1e6,
#     )
#   return app.response_class(
#     body,
#     mimetype="application/octet-stream",
#     headers={"X-Terrain-Format": "binary-v1"},
#   )
# 
# 
# @app.get("/api/buildings")
# def api_buildings():
#   """Binary building footprints, fetched independently of terrain motion."""
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
#   if "sx" in request.args and "sy" in request.args:
#     qx = _arg_float("sx", 0.0)
#     qy = _arg_float("sy", 0.0)
#   else:
#     lat = _arg_float("lat", 64.175)
#     lon = _arg_float("lon", -51.7388)
#     qx, qy = _to_stereo(lat, lon)
#   ox = _arg_float("ox", qx)
#   oy = _arg_float("oy", qy)
#   buildings = buildings_for_tile_query(
#     _get_db(), _get_assets_db(), qx, qy, ox, oy,
#   )
#   entries, ring_blobs, buildings_hash = pack_buildings(buildings, True)
#   payload = {
#     "tiles": [],
#     "buildings": entries or [],
#     "count": len(entries or []),
#     "buildingsHash": buildings_hash,
#     "qx": qx,
#     "qy": qy,
#     "ox": ox,
#     "oy": oy,
#   }
#   header = json.dumps(payload, separators=(",", ":")).encode("utf-8")
#   header += b" " * (-(len(header) + 4) % 4)
#   body = b"".join([struct.pack("<I", len(header)), header, *ring_blobs])
#   return app.response_class(
#     body,
#     mimetype="application/octet-stream",
#     headers={"X-Terrain-Format": "binary-v1"},
#   )
# 
# 
# @app.get("/api/assets")
# def api_assets():
#   """Viewer-facing startup assets, read by Flask from the shared catalog."""
#   from asset_catalog import get_assets_response
#   return jsonify(get_assets_response(_get_assets_db()))
# 
# 
# @app.get("/api/bathymetry-map")
# def api_bathymetry_map():
#   """Mapped bathymetry footprints and sounding markers near the camera."""
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
#   if "sx" in request.args and "sy" in request.args:
#     qx = _arg_float("sx", 0.0)
#     qy = _arg_float("sy", 0.0)
#   else:
#     lat = _arg_float("lat", 64.175)
#     lon = _arg_float("lon", -51.7388)
#     qx, qy = _to_stereo(lat, lon)
#   max_range = _arg_float("range", 50000.0)
#   ox = _arg_float("ox", qx)
#   oy = _arg_float("oy", qy)
#   from bathymetry_map import query_bathymetry_map
#   return jsonify(
#     query_bathymetry_map(
#       _get_db(), qx, qy, max_range, ox=ox, oy=oy,
#     )
#   )
# 
# 
# @app.post("/api/vehicle_state")
# def api_vehicle_state():
#   """Persist vehicle state without exposing the asset service to the viewer."""
#   from vehicle_catalog import save_vehicle_state
#   payload, status = save_vehicle_state(_get_assets_db(), request.get_json(silent=True) or {})
#   return jsonify(payload), status
# 
# 
# @app.get("/api/roads")
# def api_roads():
#   """Compatibility endpoint; road rendering now comes from painted textures."""
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
#   if "sx" in request.args and "sy" in request.args:
#     qx = _arg_float("sx", 0.0)
#     qy = _arg_float("sy", 0.0)
#   else:
#     lat = _arg_float("lat", 64.175)
#     lon = _arg_float("lon", -51.7388)
#     qx, qy = _to_stereo(lat, lon)
#   max_range = _arg_float("range", 20000.0)
#   ox = _arg_float("ox", qx)
#   oy = _arg_float("oy", qy)
# 
#   from asset_catalog import query_roads
#   bbox = (qx - max_range, qy - max_range, qx + max_range, qy + max_range)
#   rows = query_roads(_get_assets_db(), bbox)
#   roads = [{
#     **road,
#     "path": [[x - ox, y - oy, z] for x, y, z in road["path"]],
#   } for road in rows]
#   log.debug(f"[/api/roads] {len(roads)} polylines near qx={qx:.0f} qy={qy:.0f} range={max_range:.0f}")
#   return jsonify({"roads": roads, "count": len(roads), "qx": qx, "qy": qy})
# 
# 
# def _painted_texture_response(
#   jpeg: bytes,
#   bbox: tuple[float, float, float, float],
#   *,
#   tile_id: str,
#   headers: dict[str, str],
#   road_debug: bool = False,
#   water_debug: bool = False,
#   hydro_debug: bool = False,
# ) -> Response:
#   """Apply terrain-coupled catalog overlays to a clean cached texture copy."""
#   bake_asset = None
#   cacheable_bake = (
#     not road_debug
#     and not water_debug
#     and not hydro_debug
#     and headers.get("X-Tex-Temporary") == "0"
#   )
#   if cacheable_bake:
#     from road_texture_cache import get_or_create_road_texture
#     bake_asset = get_or_create_road_texture(
#       _get_db(), tile_id, bbox, ASSETS_DB_PATH,
#     )
#   if bake_asset is not None:
#     painted = bake_asset["texture"]
#     road_count = bake_asset["road_count"]
#   else:
#     from asset_catalog import paint_roads
#     painted, road_count = paint_roads(
#       jpeg, bbox, ASSETS_DB_PATH, debug=road_debug,
#     )
#   if hydro_debug or water_debug:
#     from classifier.rendering import smooth_effective_water_mask
#     from coastline import read_hydrography_mask, read_water_mask
#     db = _get_db()
#     hydro = read_hydrography_mask(db, tile_id) if hydro_debug else None
#     water = read_water_mask(db, tile_id) if water_debug else None
#     if hydro is not None or water is not None:
#       image = _Image.open(io.BytesIO(painted)).convert("RGB")
#       pixels = _np.asarray(image).copy()
#       if hydro is not None:
#         render_hydro = smooth_effective_water_mask(
#           hydro, image.width, image.height,
#         )
#         pixels[render_hydro] = (0, 140, 255)
#       # Paint authoritative tidal sea last so simultaneous diagnostics read as
#       # pink sea plus blue WMS-only inland hydrography.
#       if water is not None:
#         render_water = smooth_effective_water_mask(
#           water, image.width, image.height,
#         )
#         pixels[render_water] = (255, 42, 161)
#       output = io.BytesIO()
#       _Image.fromarray(pixels, mode="RGB").save(output, format="JPEG", quality=92)
#       painted = output.getvalue()
#   response_headers = dict(headers)
#   response_headers["X-Road-Overlay-Count"] = str(road_count)
#   response_headers["X-Road-Debug"] = "1" if road_debug else "0"
#   response_headers["X-Water-Debug"] = "1" if water_debug else "0"
#   response_headers["X-Hydrography-Debug"] = "1" if hydro_debug else "0"
#   if bake_asset is not None:
#     response_headers["X-Road-Bake-Cache"] = (
#       "miss" if bake_asset["generated"] else "hit"
#     )
#     response_headers["X-Road-Bake-Recipe"] = str(
#       bake_asset["recipe_version"]
#     )
#   else:
#     response_headers["X-Road-Bake-Cache"] = (
#       "bypass" if road_debug or water_debug or hydro_debug else "temporary"
#     )
#   if road_count:
#     # Asset edits must be visible on the next texture request; the canonical
#     # imagery remains cached in terrain.db and is never painted in place.
#     response_headers["Cache-Control"] = "no-cache"
#   response = Response(
#     painted, mimetype="image/jpeg", headers=response_headers,
#   )
#   if bake_asset is not None:
#     response.set_etag(bake_asset["fingerprint"])
#     response.make_conditional(request)
#   return response
# 
# 
# @app.get("/api/d13/status")
# def api_d13_status():
#   """Workflow-accuracy audit for everything past WMS_CONTRACT_DEPTH.
# 
#   Live process counters (inspections, verdicts, cooks, deferrals) plus a DB
#   audit: texture source distribution, in-flight/retry sizes, and any rows
#   stuck outside the workflow (the watchdog requeues these on its own sweep).
#   """
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
#   with _d13_stats_lock:
#     counters = dict(_d13_stats)
#   audit = _d13_texture_audit(_get_db())
#   return jsonify({
#     "contractDepth": WMS_CONTRACT_DEPTH,
#     "maxDepth": MAX_TILE_DEPTH,
#     "counters": counters,
#     "textureRows": audit["rows"],
#     "sourceDistribution": audit["distribution"],
#     "fetching": audit["fetching"],
#     "retrying": audit["retrying"],
#     "stuck": [
#       {"id": tile_id, "source": source, "ageSeconds": round(age_s, 1)}
#       for tile_id, source, age_s in audit["stuck"][:50]
#     ],
#     "stuckTotal": len(audit["stuck"]),
#   })
# 
# 
# @app.get("/api/texture/<tile_id>.jpg")
# def api_texture(tile_id: str):
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
# 
#   log_tex.debug(f"[/api/texture] tile_id={tile_id}")
# 
#   parsed = _parse_tile_id(tile_id)
#   if parsed is None:
#     log_tex.warning(f"[/api/texture] {tile_id}: invalid tile_id format")
#     return Response(b"", status=400)
#   d, c, r = parsed
#   texture_bbox = _tile_bbox(d, c, r)
#   road_debug = request.args.get("roadDebug") == "1"
#   water_debug = request.args.get("waterDebug") == "1"
#   hydro_debug = request.args.get("hydroDebug") == "1"
# 
#   db = _get_db()
#   row = db.execute(
#     "SELECT texture, source FROM textures WHERE tile_id = ?",
#     (tile_id,),
#   ).fetchone()
# 
#   cached_crop = None
#   if row is not None:
#     cached, source = row[0], row[1]
#     log_tex.debug(f"[/api/texture] {tile_id}: cache HIT source={source} size={len(cached)} bytes")
#     if source not in _TEX_TEMPORARY:
#       is_crop = source == "ancestor_crop_nodata"
#       return _painted_texture_response(
#         cached,
#         texture_bbox,
#         tile_id=tile_id,
#         road_debug=road_debug,
#         water_debug=water_debug,
#         hydro_debug=hydro_debug,
#         headers={
#           "Cache-Control": "public, max-age=86400" if not is_crop else "public, max-age=3600",
#           "X-Tex-Source": source,
#           "X-Tex-Status": "ready",
#           "X-Tex-Quality": "ancestor_crop" if is_crop else "full",
#           "X-Tex-Temporary": "0",
#         },
#       )
#     cached_crop = cached
# 
#   if row is None:
#     log_tex.debug(f"[/api/texture] {tile_id}: cache MISS, queuing fetch (depth={d} col={c} row={r})")
# 
#   # Queue a fetch for cache misses and re-fetchable sources. A persisted
#   # rate-limit row re-registers with the retry queue after a server restart;
#   # the queued/active set prevents duplicate work during normal polling.
#   source = row[1] if row else None
#   if source == "ancestor_crop_ratelimit":
#     _tex_retry_enqueue(tile_id, _tile_bbox(d, c, r), attempt=0)
#   else:
#     _queue_texture_fetch(tile_id, _tile_bbox(d, c, r))
# 
#   if cached_crop is not None:
#     return _painted_texture_response(
#       _repair_white_ocean_jpeg(db, tile_id, cached_crop),
#       texture_bbox,
#       tile_id=tile_id,
#       road_debug=road_debug,
#       water_debug=water_debug,
#       hydro_debug=hydro_debug,
#       headers={
#         "Cache-Control": "no-store",
#         "X-Tex-Ancestor": "precomputed_crop",
#         "X-Tex-Status": "ancestor_fallback",
#         "X-Tex-Quality": "ancestor_crop",
#         "X-Tex-Temporary": "1",
#         "X-Tex-Source": source or "",
#       },
#     )
# 
#   from texture import is_white_fill_jpeg
#   child_d, child_c, child_r = d, c, r
#   ancestor_found = None
#   while d > 0:
#     d -= 1
#     c //= 2
#     r //= 2
#     ancestor_id = f"{d}-{c}-{r}"
#     ancestor_tex = _read_texture(db, ancestor_id)
#     if ancestor_tex is not None:
#       # Skip poisoned white-fill ancestors (pre-filter WMS no-data frames) —
#       # a 202 and the blue vertex fallback beat serving a white square.
#       if is_white_fill_jpeg(ancestor_tex):
#         continue
#       ancestor_found = (ancestor_id, ancestor_tex)
#       break
# 
#   if ancestor_found is None:
#     log_tex.debug(f"[/api/texture] {tile_id}: no ancestor found, returning 202 (fetching)")
#     return Response(
#       b"",
#       status=202,
#       headers={"Cache-Control": "no-store", "X-Tex-Status": "fetching"},
#     )
# 
#   ancestor_id, ancestor_tex = ancestor_found
#   log_tex.debug(f"[/api/texture] {tile_id}: using ancestor {ancestor_id} ({len(ancestor_tex)} bytes)")
#   img = _Image.open(io.BytesIO(ancestor_tex))
#   w, h = img.size
# 
#   ancestor_depth = cast(
#     tuple[int, int, int], _parse_tile_id(ancestor_id)
#   )[0]
#   depth_diff = child_d - ancestor_depth
#   sub_c = child_c % (1 << depth_diff)
#   sub_r = child_r % (1 << depth_diff)
#   n = 1 << depth_diff
# 
#   x0 = sub_c * w // n
#   x1 = (sub_c + 1) * w // n
#   y0 = (n - 1 - sub_r) * h // n
#   y1 = (n - sub_r) * h // n
# 
#   cropped = img.crop((x0, y0, x1, y1)).resize((256, 256), _Image.Resampling.BILINEAR)
#   buf = io.BytesIO()
#   cropped.save(buf, format="JPEG", quality=85)
# 
#   return _painted_texture_response(
#     _repair_white_ocean_jpeg(db, tile_id, buf.getvalue()),
#     texture_bbox,
#     tile_id=tile_id,
#     road_debug=road_debug,
#     water_debug=water_debug,
#     hydro_debug=hydro_debug,
#     headers={
#       "Cache-Control": "no-store",
#       "X-Tex-Ancestor": ancestor_id,
#       "X-Tex-Status": "ancestor_fallback",
#       "X-Tex-Quality": "ancestor_crop",
#       "X-Tex-Temporary": "1",
#     },
#   )
# 
# 
# @app.get("/api/cliff-graft/<tile_id>.png")
# def api_cliff_graft(tile_id: str):
#   """Serve one shared, persistently prepared cliff-graft donor.
# 
#   The shader projects this asset across every eligible D13+ tile. Water
#   inpainting is deterministic and stored in SQLite, so browsers only decode
#   the finished PNG and never repeat the classifier/image preparation.
#   """
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
#   parsed = _parse_tile_id(tile_id)
#   if parsed is None:
#     return Response(b"", status=400)
# 
#   depth, column, row = parsed
#   db = _get_db()
#   if depth >= WMS_CONTRACT_DEPTH:
#     shift = depth - WMS_CONTRACT_DEPTH
#     _ensure_d12_class_map(
#       db,
#       f"{WMS_CONTRACT_DEPTH}-{column >> shift}-{row >> shift}",
#     )
# 
#   from cliff_graft_cache import (
#     CLIFF_GRAFT_ASSET_VERSION,
#     get_or_create_cliff_graft_asset,
#   )
# 
#   try:
#     asset = get_or_create_cliff_graft_asset(db, tile_id)
#   except (TypeError, ValueError, zlib.error) as exc:
#     log_tex.warning(
#       f"[cliff-graft] {tile_id}: preparation failed "
#       f"({type(exc).__name__}: {exc})"
#     )
#     return Response(
#       b"", status=500,
#       headers={"Cache-Control": "no-store", "X-Cliff-Graft-Status": "invalid"},
#     )
#   if asset is None:
#     texture_row = db.execute(
#       "SELECT 1 FROM textures WHERE tile_id = ?", (tile_id,),
#     ).fetchone()
#     if texture_row is None:
#       _queue_texture_fetch(tile_id, tuple(_tile_bbox(*parsed)))
#     return Response(
#       b"", status=202,
#       headers={"Cache-Control": "no-store", "X-Cliff-Graft-Status": "pending"},
#     )
# 
#   cache_status = "miss" if asset["generated"] else "hit"
#   if asset["generated"]:
#     log_tex.info(
#       f"[cliff-graft] {tile_id}: persisted recipe "
#       f"v{CLIFF_GRAFT_ASSET_VERSION} ({asset['width']}x{asset['height']}, "
#       f"{asset['water_pixels']} water pixels replaced)"
#     )
#   response = Response(
#     asset["texture"],
#     mimetype="image/png",
#     headers={
#       "Cache-Control": "public, max-age=300",
#       "X-Cliff-Graft-Status": "ready",
#       "X-Cliff-Graft-Cache": cache_status,
#       "X-Cliff-Graft-Recipe": str(CLIFF_GRAFT_ASSET_VERSION),
#       "X-Cliff-Graft-Water-Pixels": str(asset["water_pixels"]),
#     },
#   )
#   response.set_etag(asset["fingerprint"])
#   return response.make_conditional(request)
# 
# 
# def _tile_package_png(values, *, boolean: bool = False) -> bytes:
#   """Encode a south-first terrain raster as a north-first PNG."""
#   array = _np.asarray(values)
#   if boolean:
#     array = array.astype(_np.uint8) * 255
#   else:
#     array = array.astype(_np.uint8)
#   output = io.BytesIO()
#   _Image.fromarray(_np.ascontiguousarray(array[::-1]), mode="L").save(
#     output, format="PNG"
#   )
#   return output.getvalue()
# 
# 
# @app.post("/api/tile-package/<tile_id>.zip")
# def api_tile_package(tile_id: str):
#   """Download all cached inputs plus the exact seam-repaired render DEM."""
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
#   parsed = _parse_tile_id(tile_id)
#   if parsed is None:
#     return jsonify({"error": "invalid tile id"}), 400
# 
#   payload = request.get_json(silent=True) or {}
#   try:
#     resolution = int(payload.get("resolution", 0))
#     encoded_heightmap = str(payload.get("heightmap", ""))
#     heightmap_bytes = base64.b64decode(encoded_heightmap, validate=True)
#     final_heightmap = _np.frombuffer(heightmap_bytes, dtype=_np.float32).copy()
#   except (TypeError, ValueError, binascii.Error):
#     return jsonify({"error": "invalid rendered heightmap"}), 400
#   if resolution < 2 or resolution > 2048 or final_heightmap.size != resolution ** 2:
#     return jsonify({"error": "rendered heightmap shape mismatch"}), 400
#   final_heightmap = final_heightmap.reshape((resolution, resolution))
# 
#   db = _get_db()
#   row = db.execute(
#     "SELECT depth, col, row, x_min, y_min, x_max, y_max, source, "
#     "updated_at, confidence_map FROM tiles WHERE tile_id = ?",
#     (tile_id,),
#   ).fetchone()
#   if row is None:
#     return jsonify({"error": "tile not found"}), 404
#   depth, column, tile_row = int(row[0]), int(row[1]), int(row[2])
#   bbox = (float(row[3]), float(row[4]), float(row[5]), float(row[6]))
# 
#   texture_row = db.execute(
#     "SELECT texture, source, updated_at FROM textures WHERE tile_id = ?",
#     (tile_id,),
#   ).fetchone()
#   canonical_texture = texture_row[0] if texture_row else None
#   rendered_texture = canonical_texture
#   roads = []
#   if ASSETS_DB_PATH.exists():
#     from asset_catalog import connect as connect_assets, query_roads
#     assets_db = connect_assets(ASSETS_DB_PATH)
#     try:
#       roads = query_roads(assets_db, bbox)
#     finally:
#       assets_db.close()
#     if canonical_texture:
#       from road_texture_cache import get_or_create_road_texture
#       road_bake = get_or_create_road_texture(
#         db, tile_id, bbox, ASSETS_DB_PATH,
#       )
#       if road_bake is not None:
#         rendered_texture = road_bake["texture"]
# 
#   from classifier.storage import decode_class_map
#   from coastline import read_hydrography_mask, read_water_mask
# 
#   def exact_mask(table):
#     mask_row = db.execute(
#       f"SELECT width, height, mask, source, version, updated_at "
#       f"FROM {table} WHERE tile_id = ?", (tile_id,),
#     ).fetchone()
#     if mask_row is None:
#       return None, None
#     width, height = int(mask_row[0]), int(mask_row[1])
#     values = _np.frombuffer(zlib.decompress(mask_row[2]), dtype=_np.uint8)
#     if values.size != width * height:
#       return None, None
#     return values.reshape((height, width)).astype(bool), {
#       "source": mask_row[3], "version": mask_row[4], "updated_at": mask_row[5],
#     }
# 
#   coastline_mask, coastline_meta = exact_mask("coastline_masks")
#   hydrography_mask = read_hydrography_mask(db, tile_id)
#   effective_water_mask = read_water_mask(db, tile_id)
#   classifier_row = db.execute(
#     "SELECT class_schema, width, height, class_map, source, updated_at "
#     "FROM classifier_tiles WHERE tile_id = ?", (tile_id,),
#   ).fetchone()
# 
#   seam_rows = db.execute(
#     "SELECT tile_a, direction, tile_b, edge, updated_at "
#     "FROM terrain_seam_cache WHERE tile_a = ? OR tile_b = ? "
#     "ORDER BY updated_at", (tile_id, tile_id),
#   ).fetchall()
#   seam_metadata = [{
#     "tileA": seam[0], "direction": seam[1], "tileB": seam[2],
#     "edge": _np.frombuffer(seam[3], dtype=_np.float32).tolist(),
#     "updatedAt": seam[4],
#   } for seam in seam_rows]
# 
#   manifest = {
#     "format": "atlantis-terrain-tile-package-v1",
#     "tileId": tile_id,
#     "address": {"depth": depth, "column": column, "row": tile_row},
#     "bbox": {"crs": "EPSG:3413", "values": list(bbox)},
#     "heightmap": {
#       "file": "heightmap-final.npy", "dtype": "float32",
#       "shape": [resolution, resolution], "rowOrder": "south-to-north",
#       "seamCacheApplied": True, "source": row[7], "updatedAt": row[8],
#     },
#     "texture": ({
#       "sourceFile": "texture-source.jpg", "renderedFile": "texture-final.jpg",
#       "source": texture_row[1], "updatedAt": texture_row[2],
#     } if texture_row else None),
#     "masks": {
#       "coastline": coastline_meta,
#       "hydrography": hydrography_mask is not None,
#       "effectiveWater": effective_water_mask is not None,
#       "classifier": classifier_row[0] if classifier_row else None,
#     },
#     "roads": {"file": "roads.json", "count": len(roads), "crs": "EPSG:3413"},
#     "seamCache": {"file": "seam-cache.json", "entries": len(seam_metadata)},
#   }
# 
#   archive_buffer = io.BytesIO()
#   with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
#     archive.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
#     heightmap_buffer = io.BytesIO()
#     _np.save(heightmap_buffer, final_heightmap, allow_pickle=False)
#     archive.writestr("heightmap-final.npy", heightmap_buffer.getvalue())
#     archive.writestr("heightmap-final.f32", final_heightmap.tobytes())
#     if row[9] is not None:
#       confidence = _np.frombuffer(zlib.decompress(row[9]), dtype=_np.uint8)
#       if confidence.size == resolution ** 2:
#         archive.writestr(
#           "confidence-map.png",
#           _tile_package_png(confidence.reshape((resolution, resolution))),
#         )
#     if canonical_texture:
#       archive.writestr("texture-source.jpg", canonical_texture)
#       archive.writestr("texture-final.jpg", rendered_texture or canonical_texture)
#     if coastline_mask is not None:
#       archive.writestr("coastline-mask.png", _tile_package_png(coastline_mask, boolean=True))
#     if hydrography_mask is not None:
#       archive.writestr("hydrography-mask.png", _tile_package_png(hydrography_mask, boolean=True))
#     if effective_water_mask is not None:
#       archive.writestr("effective-water-mask.png", _tile_package_png(effective_water_mask, boolean=True))
#     if classifier_row is not None:
#       labels = decode_class_map(classifier_row[3], classifier_row[1], classifier_row[2])
#       archive.writestr("classifier-map.png", _tile_package_png(_np.flipud(labels)))
#     archive.writestr("roads.json", json.dumps(roads, indent=2, default=str))
#     archive.writestr("seam-cache.json", json.dumps(seam_metadata, indent=2))
# 
#   return Response(
#     archive_buffer.getvalue(), mimetype="application/zip",
#     headers={
#       "Content-Disposition": f'attachment; filename="{tile_id}-package.zip"',
#       "Cache-Control": "no-store",
#     },
#   )
# 
# 
# def _heightmap_ancestor_crop(db, d: int, c: int, r: int, max_up: int = 4):
#   """Tile heightmap, cropped out of the nearest ancestor's when the tile
#   itself was never seeded/flown (the training workspace can ask for a tile whose
#   rows only exist where browser terrain demand requested them). Same philosophy
#   as ancestor-crop textures. Returns (hm, source, depth_found) or
#   (None, None, None); hm is (GRID_N, GRID_N), row 0 = south.
#   """
#   import numpy as np
#   from PIL import Image as _Image
# 
#   from database import GRID_N, _decompress_float32
# 
#   bits = []
#   dd, cc, rr = d, c, r
#   for _ in range(max_up + 1):
#     row = db.execute(
#       "SELECT heightmap, source FROM tiles WHERE tile_id = ?",
#       (f"{dd}-{cc}-{rr}",)).fetchone()
#     if row is not None and row[0] is not None:
#       hm = _decompress_float32(row[0], (GRID_N, GRID_N))
#       half = (GRID_N - 1) // 2
#       for cb, rb in reversed(bits):
#         # row 0 = south: north child (rb=1) is the upper index half
#         sub = hm[rb * half:rb * half + half + 1, cb * half:cb * half + half + 1]
#         hm = np.array(_Image.fromarray(sub, mode="F")
#                       .resize((GRID_N, GRID_N), _Image.Resampling.BILINEAR))
#       return hm, row[1], dd
#     if dd == 0:
#       break
#     bits.append((cc & 1, rr & 1))
#     dd, cc, rr = dd - 1, cc >> 1, rr >> 1
#   return None, None, None
# 
# 
# @app.get("/api/channel/<tile_id>/<chan>.png")
# def api_terrain_channel(tile_id: str, chan: str):
#   """DEM-derived conditioning channel (elev/slope/southness/sun) as a debug
#   PNG, computed on the fly from the tile heightmap and pixel-aligned with
#   /api/texture. Southness renders diverging blue (north-
#   facing) / red (south-facing). Query params: res (default 512).
#   """
#   if chan not in ("elev", "slope", "southness", "sun"):
#     return Response(b"", status=400)
#   parsed = _parse_tile_id(tile_id)
#   if parsed is None:
#     return Response(b"", status=400)
#   d, c, r = parsed
# 
#   try:
#     res = max(64, min(2048, int(request.args.get("res", "512"))))
#   except ValueError:
#     return Response(b"", status=400)
# 
#   import io as _io
# 
#   import numpy as np
#   from PIL import Image as _Image
# 
#   from classifier.terrain_channels import render_channel, terrain_channels
# 
#   db = _get_db()
#   hm, hm_source, hm_depth = _heightmap_ancestor_crop(db, d, c, r)
#   if hm is None:
#     return Response(b"", status=404, headers={"X-Tex-Status": "no_heightmap"})
#   bbox = _tile_bbox(d, c, r)
#   chans = terrain_channels(hm, bbox[2] - bbox[0])
#   img = render_channel(chans, chan)
#   # DB heightmaps are row 0 = south; flip to image orientation
#   img = np.ascontiguousarray(img[::-1])
#   img = np.array(_Image.fromarray(img).resize((res, res), _Image.Resampling.BILINEAR))
#   buf = _io.BytesIO()
#   _Image.fromarray(img).save(buf, format="PNG")
#   return Response(
#     buf.getvalue(),
#     mimetype="image/png",
#     headers={"Cache-Control": "no-store", "X-Tex-Source": f"channel_{chan}",
#              "X-DEM-Depth": str(hm_depth), "X-DEM-Source": str(hm_source)},
#   )
# 
# 
# def _ensure_d12_class_map(db, tile_id: str) -> None:
#   """Compatibility hook for callers and tests that patch server inference."""
#   _classifier_ensure_d12_class_map(db, tile_id)
# 
# 
# configure_classifier_routes(ClassifierRouteDependencies(
#   get_db=lambda: _get_db(),
#   terrain_unavailable_response=lambda **kwargs: (
#     _terrain_unavailable_response(**kwargs)
#   ),
#   numpy=lambda: _np,
#   classifier_job_control=lambda: _classifier_job_control,
#   classifier_training_control=lambda: _classifier_training_control,
#   regression_case_summaries=lambda: regression_case_summaries(),
#   assets_db_path=lambda: ASSETS_DB_PATH,
#   texture_temporary_sources=set(_TEX_TEMPORARY),
#   ensure_d12_class_map=lambda db, tile_id: _ensure_d12_class_map(db, tile_id),
#   logger=log,
# ))
# app.register_blueprint(classifier_bp)
# 
# 
# @app.post("/api/regression/cases")
# def api_regression_flag():
#   """Flag a tile as a classifier regression case (⚑ in the 3D client).
# 
#   The case list is user-curated regression_cases.json (committed); baking
#   runs the full verification pipeline for the flagged tile immediately so
#   the gallery at /api/regression/ shows it without a manual rebake.
#   """
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
#   payload = request.get_json(silent=True) or {}
#   tile_id = str(payload.get("tile", ""))
#   if _parse_tile_id(tile_id) is None:
#     return Response(b"bad tile id", status=400)
#   note = str(payload.get("note", ""))[:400]
# 
#   import regression_cases
# 
#   regression_cases.add_case(tile_id, note)
#   try:
#     regression_cases.bake_and_rebuild(_get_db(), tile_id)
#     baked = True
#   except Exception as exc:
#     log.warning(
#       f"[regression] {tile_id}: flagged but bake failed "
#       f"({type(exc).__name__}: {exc})"
#     )
#     baked = False
#   training_pair = False
#   try:
#     from classifier.training_data import export_pairs
# 
#     export_pairs(
#       _get_db(), [tile_id], regression_tiles={
#         str(case.get("tile", "")) for case in regression_cases.load_cases()
#       }, allow_network=False,
#     )
#     training_pair = True
#   except Exception as exc:
#     log.info(
#       f"[regression] {tile_id}: frozen neural evaluation pair unavailable "
#       f"({type(exc).__name__}: {exc})"
#     )
#   return {
#     "ok": True,
#     "baked": baked,
#     "trainingPair": training_pair,
#     "split": "regression",
#     "cases": [case["tile"] for case in regression_cases.load_cases()],
#   }
# 
# 
# @app.get("/api/regression/")
# @app.get("/api/regression/<path:subpath>")
# def api_regression_gallery(subpath="index.html"):
#   """Serve the baked regression gallery (sample/regression, gitignored)."""
#   from flask import send_from_directory
# 
#   import regression_cases
# 
#   if subpath == "index.html" and not os.path.exists(
#     os.path.join(regression_cases.OUT_DIR, "index.html")
#   ):
#     regression_cases.build_gallery([])
#   return send_from_directory(regression_cases.OUT_DIR, subpath)
# 
# 
# @app.get("/api/pipeline/at.json")
# def api_pipeline_at():
#   """Resolve lat/lon (e.g. the 3D camera position) to the deepest tile that
#   has real progress (a texture), falling back to the deepest seeded tile —
#   so the HUD's pipeline link lands on the tile under the camera."""
#   try:
#     lat = float(request.args["lat"])
#     lon = float(request.args["lon"])
#   except (KeyError, ValueError):
#     return jsonify({"error": "lat and lon required"}), 400
#   from coords import to_stereo
#   x, y = to_stereo(lat, lon)
#   db = _get_db()
#   base = ("FROM tiles t {join} WHERE t.x_min <= ? AND t.x_max > ? "
#           "AND t.y_min <= ? AND t.y_max > ? ORDER BY t.depth DESC LIMIT 1")
#   args = (x, x, y, y)
#   row = db.execute(
#     "SELECT t.tile_id " + base.format(
#       join="JOIN textures xt ON xt.tile_id = t.tile_id"), args).fetchone()
#   if row is None:
#     row = db.execute("SELECT t.tile_id " + base.format(join=""), args).fetchone()
#   if row is None:
#     return jsonify({"error": "no tile at this location"}), 404
#   return jsonify({"tile": row[0]})
# 
# 
# @app.get("/api/pipeline/<tile_id>.json")
# def api_pipeline(tile_id: str):
#   """Per-stage pipeline status for one tile. Says what
#   each stage has (and where it came from) without rendering anything."""
#   if _parse_tile_id(tile_id) is None:
#     return jsonify({"error": f"bad tile id: {tile_id!r}"}), 400
#   db = _get_db()
#   t = db.execute(
#     "SELECT depth, x_min, y_min, x_max, y_max, heightmap IS NOT NULL, source, "
#     "updated_at, dem_demanded_at, dem_requested_at, cog_requested_at "
#     "FROM tiles WHERE tile_id = ?",
#     (tile_id,)).fetchone()
#   if t is None:
#     return jsonify({"error": "tile not in grid — never seeded"}), 404
#   x = db.execute(
#     "SELECT source, LENGTH(texture), updated_at FROM textures WHERE tile_id = ?",
#     (tile_id,)).fetchone()
#   return jsonify({
#     "tile": tile_id,
#     "depth": t[0],
#     "bbox": list(t[1:5]),
#     "size_m": round(t[3] - t[1], 1),
#     "heightmap": {
#       "ok": bool(t[5]), "source": t[6], "updated": t[7],
#       "dem_demanded_at": t[8], "dem_requested_at": t[9],
#       "cog_requested_at": t[10],
#     },
#     "texture": ({"ok": True, "source": x[0], "bytes": x[1], "updated": x[2]}
#                 if x else {"ok": False}),
#   })
# 
# 
# @app.get("/api/coverage/index.json")
# def api_coverage_index():
#   """Every tile with a REAL cached texture (placeholder crops excluded) —
#   ?maxdepth caps the depth (default 12, the contract level); deeper tiles are
#   detail, not coverage."""
#   try:
#     maxdepth = int(request.args.get("maxdepth", "12"))
#   except ValueError:
#     return jsonify({"error": "bad maxdepth"}), 400
#   db = _get_db()
#   tiles = [
#     {"tile": tid, "bbox": [x0, y0, x1, y1], "depth": d}
#     for tid, d, x0, y0, x1, y1 in db.execute(
#       "SELECT t.tile_id, t.depth, t.x_min, t.y_min, t.x_max, t.y_max "
#       "FROM textures x JOIN tiles t ON t.tile_id = x.tile_id "
#       "WHERE t.depth <= ? AND x.source NOT IN "
#       "('sentinel2_crop', 'ancestor_crop', 'ancestor_crop_ratelimit', "
#       "'ancestor_crop_nodata') ORDER BY t.depth", (maxdepth,))
#   ]
#   return jsonify({"tiles": tiles})
# 
# 
# 
# @app.post("/api/tile_inspect")
# def api_tile_inspect():
#   """Client clicked a tile in map mode — dump DB info."""
#   unavailable = _terrain_unavailable_response()
#   if unavailable is not None:
#     return unavailable
# 
#   data = request.get_json(silent=True) or {}
#   tid = data.get("tileId", "?")
#   db = _get_db()
#   row = db.execute(
#     "SELECT source, updated_at, length(texture) FROM textures WHERE tile_id = ?",
#     (tid,),
#   ).fetchone()
#   parsed = _parse_tile_id(tid)
#   log_db.info(f"[INSPECT] {tid}")
#   log_db.info(
#     f"  client: tex={data.get('tex')} dim={data.get('texDim')} "
#     f"color={data.get('color')} vc={data.get('vc')} po={data.get('po')} poFactor={data.get('poFactor')}"
#   )
# 
#   # --- terrain tile info from tiles table ---
#   if parsed:
#     d, c, r = parsed
#     tile_row = db.execute(
#       "SELECT depth, x_min, y_min, x_max, y_max, geometric_error, source, "
#       "updated_at, heightmap, confidence_map, dem_demanded_at, "
#       "dem_requested_at, cog_requested_at "
#       "FROM tiles WHERE tile_id = ?",
#       (tid,),
#     ).fetchone()
#     if tile_row:
#       geo_err = tile_row[5]
#       tile_src = tile_row[6]
#       tile_updated = tile_row[7]
#       hm_blob, cm_blob = tile_row[8], tile_row[9]
#       dem_demanded_at = tile_row[10]
#       dem_requested_at, cog_requested_at = tile_row[11], tile_row[12]
#       bbox_db = [tile_row[1], tile_row[2], tile_row[3], tile_row[4]]
#       tile_w = bbox_db[2] - bbox_db[0]
#       tile_h = bbox_db[3] - bbox_db[1]
#       log_db.info(f"  tile: source={tile_src} geo_err={geo_err:.2f}m depth={d} "
#             f"size={tile_w:.0f}x{tile_h:.0f}m")
#       log_db.info(f"  bbox: [{bbox_db[0]:.0f}, {bbox_db[1]:.0f}, {bbox_db[2]:.0f}, {bbox_db[3]:.0f}]")
#       log_db.info(
#         f"  requests: dem_demanded_at={dem_demanded_at} "
#         f"dem_requested_at={dem_requested_at} "
#         f"cog_requested_at={cog_requested_at}"
#       )
#       if hm_blob:
#         try:
#           import zlib
#           hm = _np.frombuffer(zlib.decompress(hm_blob), dtype=_np.float32).reshape((_GRID_N, _GRID_N))
#           log_db.info(f"  heightmap: min={hm.min():.1f}m max={hm.max():.1f}m "
#                 f"mean={hm.mean():.1f}m range={hm.max() - hm.min():.1f}m")
#         except Exception as e:
#           log_db.warning(f"  heightmap: error — {e}")
#       else:
#         log_db.info(f"  heightmap: NONE")
# 
#       # --- neighbor info ---
#       neighbors = {}
#       for label, nc, nr in [("N", c, r+1), ("S", c, r-1), ("E", c+1, r), ("W", c-1, r)]:
#         if nc < 0 or nr < 0:
#           continue
#         nid = f"{d}-{nc}-{nr}"
#         nr_row = db.execute(
#           "SELECT source, heightmap IS NOT NULL FROM tiles WHERE tile_id = ?",
#           (nid,),
#         ).fetchone()
#         if nr_row:
#           neighbors[label] = f"{nr_row[0]}{'(hm)' if nr_row[1] else ''}"
#         else:
#           neighbors[label] = "missing"
#       log_db.info(f"  neighbors: {neighbors}")
#     else:
#       log_db.warning(f"  tile: NOT IN tiles TABLE")
# 
#   # --- texture info from textures table ---
#   if row:
#     source, updated, size = row
#     log_tex.info(f"  texture: source={source} updated={updated} size={size} bytes")
#     try:
#       tex_blob = db.execute(
#         "SELECT texture FROM textures WHERE tile_id = ?", (tid,)
#       ).fetchone()[0]
#       img = _Image.open(io.BytesIO(tex_blob))
#       arr = _np.array(img)
#       zero_pct = _np.mean(arr.max(axis=2) == 0) * 100
#       log_tex.info(
#         f"  pixels: {img.size[0]}x{img.size[1]} min={arr.min()} max={arr.max()} "
#         f"mean={arr.mean():.1f} zero={zero_pct:.1f}%"
#       )
#     except Exception as e:
#       log_tex.warning(f"  pixels: error — {e}")
#   else:
#     log_tex.info(f"  texture: NOT IN DB")
# 
#   # --- auto-fix: reset terminal ancestor_crop states and re-queue ---
#   if row and row[0] in ("ancestor_crop_nodata", "ancestor_crop_ratelimit", "ancestor_crop"):
#     state = row[0]
#     db.execute("DELETE FROM textures WHERE tile_id = ?", (tid,))
#     db.commit()
#     log_tex.warning(f"  AUTO-FIX: deleted {state}, re-queuing fetch")
#     if parsed:
#       d, c, r = parsed
#       _queue_texture_fetch(tid, _tile_bbox(d, c, r))
# 
#   # --- texture ancestry ---
#   if parsed and not row:
#     ancestor = _nearest_ancestor_texture(tid, _texture_ids_in(db, [tid]))
#     if ancestor:
#       log_tex.info(f"  tex fallback: using ancestor {ancestor}")
# 
#   hist = data.get("history", [])
#   if hist:
#     log_db.info(f"  history ({len(hist)} events):")
#     for ev in hist:
#       log_db.info(f"    {ev}")
#   return jsonify({"ok": True})
# 
# 
# def _client_log_level(raw_level: Any) -> int:
#   if isinstance(raw_level, str):
#     level = raw_level.strip().lower()
#   else:
#     level = ""
#   return {
#     "debug": logging.DEBUG,
#     "info": logging.INFO,
#     "warn": logging.WARNING,
#     "warning": logging.WARNING,
#     "error": logging.ERROR,
#     "critical": logging.CRITICAL,
#   }.get(level, logging.INFO)
# 
# 
# @app.post("/api/client_log")
# def api_client_log():
#   data = request.get_json(silent=True) or {}
#   raw_entries = data.get("entries")
#   if isinstance(raw_entries, list):
#     entries = raw_entries
#   elif isinstance(data, dict):
#     entries = [data]
#   else:
#     entries = []
# 
#   if not entries:
#     return jsonify({"ok": True, "written": 0, "dropped": 0})
# 
#   # Keep each request bounded even if a client floods us.
#   max_entries = 200
#   incoming_count = len(entries)
#   dropped = max(0, incoming_count - max_entries)
#   written = 0
# 
#   scene_mode = data.get("sceneMode")
# 
#   for item in entries[:max_entries]:
#     if not isinstance(item, dict):
#       dropped += 1
#       continue
#     payload = {
#       "ts": item.get("ts"),
#       "sceneMode": item.get("sceneMode") or scene_mode,
#       "phase": item.get("phase") or item.get("event"),
#       "elapsedMs": item.get("elapsedMs"),
#       "memory": item.get("memory"),
#       "details": item.get("details"),
#     }
#     payload = {k: v for k, v in payload.items() if v is not None}
#     if "phase" not in payload:
#       payload["phase"] = "client.log"
#     line = json.dumps(payload, ensure_ascii=False, default=str)
#     if len(line) > 20000:
#       line = line[:20000] + "...<truncated>"
#     _client_log.log(_client_log_level(item.get("level")), line)
#     _client_log_ring.append(payload)
#     if len(_client_log_ring) > _CLIENT_LOG_RING_MAX:
#       del _client_log_ring[:len(_client_log_ring) - _CLIENT_LOG_RING_MAX]
#     _broadcast_client_log(payload)
#     written += 1
# 
#   return jsonify(
#     {
#       "ok": True,
#       "written": written,
#       "dropped": dropped,
#       "logPath": str(CLIENT_LOG_PATH),
#     }
#   )
# 
# 
# @app.get("/api/client_log/ring")
# def api_client_log_ring():
#   """Raw JSON dump of the ring buffer for debugging."""
#   return jsonify({"count": len(_client_log_ring), "entries": _client_log_ring[-50:]})
# 
# 
# @app.get("/api/gpu-profile")
# def api_gpu_profile():
#   response = jsonify(_gpu_profile_control.snapshot())
#   response.headers["Cache-Control"] = "no-store"
#   return response
# 
# 
# @app.post("/api/gpu-profile/start")
# def api_gpu_profile_start():
#   data = request.get_json(silent=True) or {}
#   # A timer query can force a command-buffer boundary on tile-based GPUs.
#   # Keep the default sparse enough that an explicit capture does not turn its
#   # own instrumentation overhead into an apparent steady-state regression.
#   raw_interval = data.get("sampleInterval", 60)
#   if isinstance(raw_interval, bool):
#     raw_interval = None
#   if not isinstance(raw_interval, (str, int, float)):
#     sample_interval = 0
#   else:
#     try:
#       sample_interval = int(raw_interval)
#     except ValueError:
#       sample_interval = 0
#   if sample_interval < 1 or sample_interval > 600:
#     return jsonify({
#       "ok": False,
#       "error": "sampleInterval must be an integer from 1 to 600",
#     }), 400
#   try:
#     state = _gpu_profile_control.start(sample_interval)
#   except RuntimeError as exc:
#     return jsonify({"ok": False, "error": str(exc)}), 409
#   log.info(
#     f"[gpu-profile] start id={state['profileId']} "
#     f"sample_interval={sample_interval}"
#   )
#   return jsonify(state), 202
# 
# 
# @app.post("/api/gpu-profile/stop")
# def api_gpu_profile_stop():
#   try:
#     state = _gpu_profile_control.stop()
#   except RuntimeError as exc:
#     return jsonify({"ok": False, "error": str(exc)}), 409
#   log.info(f"[gpu-profile] stop id={state['profileId']}")
#   return jsonify(state), 202
# 
# 
# @app.post("/api/gpu-profile/report")
# def api_gpu_profile_report():
#   data = request.get_json(silent=True) or {}
#   try:
#     state = _gpu_profile_control.report(
#       profile_id=str(data.get("profileId") or ""),
#       phase=str(data.get("phase") or ""),
#       client=data.get("client") if isinstance(data.get("client"), dict) else None,
#       result=data.get("result") if isinstance(data.get("result"), dict) else None,
#       error=str(data.get("error") or "") or None,
#     )
#   except LookupError as exc:
#     return jsonify({"ok": False, "error": str(exc)}), 409
#   except (RuntimeError, ValueError) as exc:
#     return jsonify({"ok": False, "error": str(exc)}), 400
#   log.info(
#     f"[gpu-profile] browser phase={data.get('phase')} "
#     f"id={state['profileId']} backend={state.get('client', {}).get('backend')}"
#   )
#   return jsonify(state)
# 
# 
# def _broadcast_client_log(payload: dict) -> None:
#   """Push a log entry to all connected websocket viewers (thread-safe)."""
#   if _ws_loop is None or _ws_queue is None:
#     return
#   msg = json.dumps(payload, ensure_ascii=False, default=str)
#   _ws_loop.call_soon_threadsafe(_ws_queue.put_nowait, msg)
# 
# 
# async def _ws_broadcaster() -> None:
#   """Async task that pulls from the queue and sends to all ws clients."""
#   import websockets
#   queue = _ws_queue
#   if queue is None:
#     return
#   while True:
#     msg = await queue.get()
#     dead = set()
#     for ws in list(_ws_clients):
#       try:
#         await ws.send(msg)
#       except Exception as exc:
#         log.debug(
#           f"[WS] dropping client after send failed: "
#           f"{type(exc).__name__}: {exc}"
#         )
#         dead.add(ws)
#     for ws in dead:
#       _ws_clients.discard(ws)
# 
# 
# async def _ws_handler(websocket) -> None:
#   log.info(f"[WS] client connected, total={len(_ws_clients) + 1}")
#   _ws_clients.add(websocket)
#   try:
#     await websocket.wait_closed()
#   except Exception as exc:
#     log.debug(
#       f"[WS] client close raised {type(exc).__name__}: {exc}"
#     )
#   finally:
#     _ws_clients.discard(websocket)
#     log.info(f"[WS] client disconnected, total={len(_ws_clients)}")
# 
# 
# def _start_ws_server(host: str, port: int) -> None:
#   """Run the async websocket server in a background thread."""
#   import websockets
# 
#   global _ws_loop, _ws_queue
# 
#   async def _serve():
#     global _ws_loop, _ws_queue
#     _ws_loop = asyncio.get_running_loop()
#     _ws_queue = asyncio.Queue()
#     asyncio.create_task(_ws_broadcaster())
#     async with websockets.serve(_ws_handler, host, port, compression=None):
#       log.info(f"WebSocket server listening on ws://{host}:{port}")
#       await asyncio.Future()  # run forever
# 
#   asyncio.run(_serve())
# 
# 
# @app.get("/api/health/terrain")
# def terrain_health():
#   unavailable = _terrain_unavailable_response(status=500)
#   if unavailable is not None:
#     return unavailable
# 
#   db_exists = DB_PATH.exists()
#   try:
#     row = _get_db().execute("SELECT COUNT(*) FROM tiles").fetchone()
#     tile_rows = int(row[0]) if row else 0
#   except Exception as exc:
#     log.error(
#       f"[health] terrain database check failed: {type(exc).__name__}: {exc}"
#     )
#     return jsonify({
#       "ok": False,
#       "dbPath": str(DB_PATH),
#       "dbExists": db_exists,
#       "error": f"{type(exc).__name__}: {exc}",
#     }), 500
# 
#   return jsonify(
#     {
#       "ok": True,
#       "dbPath": str(DB_PATH),
#       "dbExists": db_exists,
#       "tileRows": tile_rows,
#     }
#   )
# 
# 
# @app.get("/client_log.html")
# def client_log_page():
#   if CLIENT_LOG_HTML_PATH.is_file():
#     response = send_from_directory(str(CLIENT_LOG_HTML_PATH.parent), CLIENT_LOG_HTML_PATH.name)
#     response.headers["Cache-Control"] = "no-store"
#     return response
#   dist_candidate = DIST_DIR / "client_log.html"
#   if dist_candidate.is_file():
#     response = send_from_directory(STATIC_DIR, "client_log.html")
#     response.headers["Cache-Control"] = "no-store"
#     return response
#   return Response("client_log.html not found", status=404, mimetype="text/plain")
# 
# 
# @app.get("/")
# def index():
#   return send_from_directory(STATIC_DIR, "index.html")
# 
# 
# @app.get("/coverage.html")
# def retired_coverage_page():
#   return Response(
#     "Not found",
#     status=404,
#     mimetype="text/plain",
#     headers={"Cache-Control": "no-store"},
#   )
# 
# 
# @app.get("/<path:path>")
# def static_files(path: str):
#   if path.startswith("api/"):
#     return Response(b"", status=404)
#   candidate = DIST_DIR / path
#   if candidate.is_file():
#     return send_from_directory(STATIC_DIR, path)
#   return send_from_directory(STATIC_DIR, "index.html")
# 
# 
# if __name__ == "__main__":
#   logging.getLogger("werkzeug").setLevel(logging.WARNING)
#   host = os.environ.get("FLASK_HOST", "127.0.0.1")
#   port = int(os.environ.get("FLASK_PORT", "5180"))
#   _require_available_port(host, port)
#   if DIST_DIR.exists():
#     log.info(f"Serving static dist from: {DIST_DIR}")
#   else:
#     log.info("No dist/ found — running API-only (use Vite dev server for frontend)")
#   log.info(f"Terrain DB path: {DB_PATH}")
#   log.info(f"Client debug log path: {CLIENT_LOG_PATH}")
#   _bootstrap_backend()
#   if not _backend_ready:
#     raise SystemExit(f"Backend init failed: {_backend_error}")
#   ws_port = int(os.environ.get("WS_PORT", "5181"))
#   ws_thread = threading.Thread(target=_start_ws_server, args=(host, ws_port), daemon=True)
#   ws_thread.start()
#   app.run(host=host, port=port, debug=False)
