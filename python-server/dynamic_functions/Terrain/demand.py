"""Independent nonblocking demand lanes for camera-selected terrain work."""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import subprocess
import threading
import time
import urllib.error
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import atlantis

from dynamic_functions.Terrain.arctic_dem import _fetch_heightmap
from dynamic_functions.Terrain.binary_batch import encode_composed_tiles_binary
from dynamic_functions.Terrain.bathymetry_demand import (
    eligible_fjord_jobs,
    run_bathymetry_job,
)
from dynamic_functions.Terrain.camera_lod import (
    compose_camera_from_ready_data,
    resolve_lod_coverage,
    select_lod_tiles,
)
from dynamic_functions.Terrain.coastline import (
    SOURCE as COASTLINE_SOURCE,
    VERSION as COASTLINE_VERSION,
    _acquire_mask as _acquire_coastline,
    write_coastline_mask,
)
from dynamic_functions.Terrain.Database.database import connection_lock, db
from dynamic_functions.Terrain.Database.textures import write_texture_metatile
from dynamic_functions.Terrain.Database.tiles import write_dem
from dynamic_functions.Terrain.dataforsyningen import (
    _fetch_metatile,
    _split_metatile,
)
from dynamic_functions.Terrain.hydrography import (
    SOURCE as HYDROGRAPHY_SOURCE,
    VERSION as HYDROGRAPHY_VERSION,
    _acquire_mask as _acquire_hydrography,
    write_hydrography_mask,
)
from dynamic_functions.Terrain.terrain_config import (
    WMS_CONTRACT_DEPTH,
)
from dynamic_functions.Terrain.tidal_connectivity import (
    _build_connected_hydrography,
    write_connectivity_snapshot,
)
from dynamic_functions.Terrain.tile_address import (
    format_tile_id,
    require_tile_id,
)


Worker = Callable[[str], dict]
FailureClassifier = Callable[[Exception], bool]
MAX_DEMAND_ITEMS = 2500
_SQL_CHUNK = 500
_REGISTRY_KEY = "Terrain.demand.registry.v2"
DEFAULT_RETRY_DELAYS = (2.0, 10.0)


def retryable_failure(exc: Exception) -> bool:
    """Classify only provider/transport failures as retryable."""

    if type(exc).__name__.endswith("ClobberError"):
        return False
    if isinstance(exc, (TypeError, ValueError)):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 425, 429, 500, 502, 503, 504}
    if isinstance(
        exc,
        (urllib.error.URLError, TimeoutError, ConnectionError),
    ):
        return True
    if isinstance(exc, subprocess.CalledProcessError):
        # curl transport/DNS/timeout/TLS failures; HTTP credential and missing
        # data errors use fetch code 22 and stay terminal.
        return exc.returncode in {
            5,
            6,
            7,
            18,
            23,
            26,
            28,
            35,
            47,
            52,
            55,
            56,
            92,
        }
    return isinstance(exc, OSError)


def _publish_lock() -> threading.RLock:
    """Backward-compatible name for the shared SQLite connection lock."""

    return connection_lock()


class DemandLane:
    """A bounded executor with a replaceable, deduplicated userspace queue."""

    def __init__(
        self,
        name: str,
        worker: Worker,
        capacity: int,
        *,
        retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
        classifier: FailureClassifier = retryable_failure,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not name or capacity <= 0:
            raise ValueError("demand lane requires a name and positive capacity")
        self.name = name
        self.capacity = int(capacity)
        self._worker = worker
        if not retry_delays or any(delay < 0.0 for delay in retry_delays):
            raise ValueError("retry delays must be non-negative")
        self._retry_delays = tuple(float(delay) for delay in retry_delays)
        self._max_attempts = len(self._retry_delays) + 1
        self._classifier = classifier
        self._clock = clock
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._pending: OrderedDict[str, None] = OrderedDict()
        self._active: set[str] = set()
        self._claimed: set[str] = set()
        self._completed: set[str] = set()
        self._failures: dict[str, dict] = {}
        self._attempts: dict[str, int] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=self.capacity,
            thread_name_prefix=f"terrain-{name}",
        )
        self._closed = False

    def submit(
        self,
        item_ids: list[str],
        *,
        reopen_completed: bool = False,
    ) -> dict:
        validated = self._validated_item_ids(item_ids)
        accepted = []
        with self._lock:
            if self._closed:
                raise RuntimeError(f"demand lane {self.name} is closed")
            self._claimed.update(validated)
            if reopen_completed:
                # Callers use this only for authoritative database misses.
                # A prior worker completion without its promised output must
                # not suppress the item forever.
                self._completed.difference_update(validated)
            for item_id in validated:
                if self._retry_eligible_locked(item_id):
                    self._failures.pop(item_id, None)
                if (
                    item_id in self._active
                    or item_id in self._pending
                    or item_id in self._completed
                    or item_id in self._failures
                ):
                    continue
                self._pending[item_id] = None
                accepted.append(item_id)
            self._dispatch_locked()
            status = self._status_locked()
        return {"accepted": accepted, "acceptedCount": len(accepted), **status}

    def _validated_item_ids(self, item_ids: list[str]) -> list[str]:
        if not isinstance(item_ids, list):
            raise TypeError("demand item_ids must be a list")
        if len(item_ids) > MAX_DEMAND_ITEMS:
            raise ValueError(
                f"demand contains {len(item_ids)} items; maximum is "
                f"{MAX_DEMAND_ITEMS}"
            )
        result = []
        seen = set()
        for item_id in item_ids:
            if not isinstance(item_id, str) or not item_id:
                raise TypeError("demand item IDs must be non-empty strings")
            if item_id not in seen:
                seen.add(item_id)
                result.append(item_id)
        return result

    def replace_pending(self, item_ids: list[str]) -> dict:
        """Replace only unstarted work with the newest priority ordering."""

        validated = self._validated_item_ids(item_ids)
        with self._lock:
            if self._closed:
                raise RuntimeError(f"demand lane {self.name} is closed")
            self._claimed = set(validated)
            # ``validated`` is derived from authoritative database misses.
            # A completed item that appears here did not publish its promised
            # output (for example, a provider returned no metatile). Reopen it
            # instead of allowing an in-memory success marker to suppress the
            # missing work forever.
            self._completed.difference_update(validated)
            previous = set(self._pending)
            replacement: OrderedDict[str, None] = OrderedDict()
            for item_id in validated:
                if self._retry_eligible_locked(item_id):
                    self._failures.pop(item_id, None)
                if (
                    item_id in self._active
                    or item_id in self._completed
                    or item_id in self._failures
                ):
                    continue
                replacement[item_id] = None
            self._pending = replacement
            dropped = sorted(previous - set(replacement))
            accepted = [
                item_id for item_id in replacement if item_id not in previous
            ]
            retained = [
                item_id for item_id in replacement if item_id in previous
            ]
            self._dispatch_locked()
            status = self._status_locked()
        return {
            "accepted": accepted,
            "acceptedCount": len(accepted),
            "retained": retained,
            "dropped": dropped,
            **status,
        }

    def _dispatch_locked(self) -> None:
        while self._pending and len(self._active) < self.capacity:
            item_id, _ = self._pending.popitem(last=False)
            self._active.add(item_id)
            self._attempts[item_id] = self._attempts.get(item_id, 0) + 1
            future = self._executor.submit(self._worker, item_id)
            future.add_done_callback(
                lambda completed, demand_id=item_id: self._finished(
                    demand_id, completed
                )
            )

    def _finished(self, item_id: str, future: Future) -> None:
        with self._lock:
            self._active.discard(item_id)
            try:
                result = future.result()
                if not isinstance(result, dict):
                    raise TypeError("demand workers must return an object")
                if not result.get("deferred", False):
                    self._completed.add(item_id)
            except Exception as exc:
                attempts = self._attempts[item_id]
                transient = self._classifier(exc)
                retryable = transient and attempts < self._max_attempts
                retry_at = (
                    self._clock() + self._retry_delays[attempts - 1]
                    if retryable
                    else None
                )
                self._failures[item_id] = {
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                    "attempts": attempts,
                    "retryable": retryable,
                    "retryAt": retry_at,
                    "exhausted": transient and not retryable,
                }
            self._dispatch_locked()
            if not self._active and not self._pending:
                self._idle.notify_all()

    def _retry_eligible_locked(self, item_id: str) -> bool:
        failure = self._failures.get(item_id)
        return bool(
            failure
            and failure.get("retryable")
            and isinstance(failure.get("retryAt"), (int, float))
            and self._clock() >= failure["retryAt"]
        )

    def _status_locked(self) -> dict:
        failures = {
            item_id: {**failure, "claimed": item_id in self._claimed}
            for item_id, failure in self._failures.items()
        }
        return {
            "name": self.name,
            "capacity": self.capacity,
            "active": sorted(self._active),
            "claimedActive": sorted(self._active & self._claimed),
            "pending": list(self._pending),
            "claimedCount": len(self._claimed),
            "completedCount": len(self._completed),
            "failures": failures,
        }

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def wait_for_idle(self, timeout: float) -> bool:
        """Test/administration helper; camera paths never call this."""

        with self._idle:
            return self._idle.wait_for(
                lambda: not self._active and not self._pending,
                timeout=timeout,
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._claimed.clear()
            self._pending.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)


class DemandCoordinator:
    """Submit and inspect named lanes without sharing their worker capacity."""

    def __init__(self, lanes: dict[str, DemandLane]) -> None:
        self.lanes = dict(lanes)

    def submit(
        self,
        demands: dict[str, list[str]],
        *,
        reopen_completed: bool = False,
    ) -> dict:
        unknown = set(demands) - set(self.lanes)
        if unknown:
            raise ValueError(f"unknown terrain demand lanes: {sorted(unknown)}")
        return {
            name: self.lanes[name].submit(
                item_ids,
                reopen_completed=reopen_completed,
            )
            for name, item_ids in demands.items()
        }

    def status(self) -> dict:
        return {name: lane.status() for name, lane in self.lanes.items()}

    def refresh(self, demands: dict[str, list[str]]) -> dict:
        """Replace every unstarted lane queue from one latest camera claim."""

        unknown = set(demands) - set(self.lanes)
        if unknown:
            raise ValueError(f"unknown terrain demand lanes: {sorted(unknown)}")
        return {
            name: lane.replace_pending(demands.get(name, []))
            for name, lane in self.lanes.items()
        }


def polling_state(lanes: dict[str, dict], now: float | None = None) -> dict:
    """Summarize useful polling without waking or waiting for workers."""

    current_time = time.time() if now is None else float(now)
    active = 0
    pending = 0
    retryable = 0
    terminal = 0
    retry_times = []
    by_lane = {}
    for name, lane in lanes.items():
        claimed_active = lane.get("claimedActive", lane.get("active", []))
        lane_pending = lane.get("pending", [])
        lane_retry_times = []
        lane_terminal = 0
        for failure in lane.get("failures", {}).values():
            if not failure.get("claimed", True):
                continue
            retry_at = failure.get("retryAt")
            if failure.get("retryable") and isinstance(
                retry_at, (int, float)
            ):
                lane_retry_times.append(float(retry_at))
            else:
                lane_terminal += 1
        active += len(claimed_active)
        pending += len(lane_pending)
        retryable += len(lane_retry_times)
        terminal += lane_terminal
        retry_times.extend(lane_retry_times)
        by_lane[name] = {
            "active": len(claimed_active),
            "pending": len(lane_pending),
            "retryable": len(lane_retry_times),
            "terminal": lane_terminal,
        }

    next_retry_at = min(retry_times) if retry_times else None
    if active or pending:
        next_action = "poll"
        retry_after_ms = 1000
    elif next_retry_at is not None:
        next_action = "retry"
        retry_after_ms = max(
            1,
            int((next_retry_at - current_time) * 1000),
        )
    else:
        next_action = "idle"
        retry_after_ms = None
    return {
        "nextAction": next_action,
        "shouldPoll": next_action != "idle",
        "retryAfterMs": retry_after_ms,
        "nextRetryAt": next_retry_at,
        "activeWork": active,
        "pendingWork": pending,
        "retryableFailures": retryable,
        "terminalFailures": terminal,
        "lanes": by_lane,
    }


def _dem_worker(tile_id: str) -> dict:
    heightmap, sources, geoid_undulation = _fetch_heightmap(tile_id)
    with _publish_lock():
        written = write_dem(
            db(), tile_id, heightmap, "arcticdem_10m", "EGM2008"
        )
    return {
        "tileId": tile_id,
        "written": written,
        "sources": sources,
        "geoidUndulation": geoid_undulation,
    }


def _texture_worker(tile_id: str) -> dict:
    token = os.environ.get("DATAFORSYNINGEN_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DATAFORSYNINGEN_TOKEN is required")
    metatile, provider = _fetch_metatile(tile_id, token)
    if metatile is None:
        status = str(provider.get("status") or "provider_error")
        message = str(provider.get("message") or status)
        detail = f"Dataforsyningen texture {tile_id}: {status}: {message}"
        if status in {"network_error", "transient_error", "rate_limited"}:
            raise ConnectionError(detail)
        raise RuntimeError(detail)
    children = _split_metatile(metatile, tile_id)
    with _publish_lock():
        written = write_texture_metatile(db(), children, "dataforsyningen")
    return {"tileId": tile_id, "written": written, **provider}


def _coastline_worker(tile_id: str) -> dict:
    mask, acquisition = _acquire_coastline(tile_id)
    with _publish_lock():
        written = write_coastline_mask(
            db(),
            tile_id,
            mask,
            COASTLINE_SOURCE,
            COASTLINE_VERSION,
        )
    return {"tileId": tile_id, "written": written, **acquisition}


def _hydrography_worker(tile_id: str) -> dict:
    mask, acquisition = _acquire_hydrography(tile_id)
    with _publish_lock():
        written = write_hydrography_mask(
            db(),
            tile_id,
            mask,
            HYDROGRAPHY_SOURCE,
            HYDROGRAPHY_VERSION,
        )
    return {"tileId": tile_id, "written": written, **acquisition}


def _connectivity_worker(depth_text: str) -> dict:
    depth = int(depth_text.partition(":")[0])
    if depth < 0 or depth > WMS_CONTRACT_DEPTH:
        raise ValueError(f"invalid connectivity depth: {depth}")
    with _publish_lock():
        connection = db()
        masks = _build_connected_hydrography(connection, depth)
        if not masks:
            return {"depth": depth, "deferred": True, "published": 0}
        published = 0
        for tile_id, mask in masks.items():
            published += int(
                write_connectivity_snapshot(
                    connection, tile_id, mask, commit=False
                )
            )
        connection.commit()
    return {"depth": depth, "published": published}


def _new_coordinator() -> DemandCoordinator:
    return DemandCoordinator(
        {
            "dem": DemandLane("dem", _dem_worker, 4),
            "texture": DemandLane("texture", _texture_worker, 2),
            "coastline": DemandLane("coastline", _coastline_worker, 1),
            "hydrography": DemandLane(
                "hydrography", _hydrography_worker, 1
            ),
            "connectivity": DemandLane(
                "connectivity", _connectivity_worker, 1
            ),
            "bathymetry": DemandLane(
                "bathymetry", run_bathymetry_job, 1,
                retry_delays=(30.0, 120.0, 600.0),
            ),
        }
    )


def _coordinator() -> DemandCoordinator:
    coordinator = atlantis.server_shared.get(_REGISTRY_KEY)
    if coordinator is None:
        coordinator = _new_coordinator()
        atlantis.server_shared.set(_REGISTRY_KEY, coordinator)
    return coordinator


def _contract_tile_id(tile_id: str) -> str:
    depth, column, row = require_tile_id(tile_id)
    if depth <= WMS_CONTRACT_DEPTH:
        return tile_id
    shift = depth - WMS_CONTRACT_DEPTH
    return format_tile_id(
        WMS_CONTRACT_DEPTH, column >> shift, row >> shift
    )


def _metatile_id(tile_id: str) -> str:
    depth, column, row = require_tile_id(tile_id)
    if depth < 2:
        return tile_id
    return format_tile_id(depth, column - column % 4, row - row % 4)


def submit_texture_demand(tile_id: str) -> dict:
    """Queue one browser texture miss without replacing camera lane claims."""

    require_tile_id(tile_id)
    return _coordinator().lanes["texture"].submit([_metatile_id(tile_id)])


def _present_ids(
    connection: sqlite3.Connection,
    table: str,
    item_ids: list[str],
) -> set[str]:
    if table not in {
        "dem",
        "textures",
        "coastline_masks",
        "hydrography_masks",
        "tidal_connectivity_masks",
        "bathymetry",
    }:
        raise ValueError(f"unsupported demand table: {table}")
    ready: set[str] = set()
    for start in range(0, len(item_ids), _SQL_CHUNK):
        chunk = item_ids[start : start + _SQL_CHUNK]
        if not chunk:
            continue
        marks = ",".join("?" for _ in chunk)
        if table == "dem":
            sql = (
                "SELECT tile_id FROM tiles "
                f"WHERE tile_id IN ({marks}) "
                "AND heightmap IS NOT NULL AND confidence_map IS NOT NULL"
            )
        else:
            sql = f"SELECT tile_id FROM {table} WHERE tile_id IN ({marks})"
        ready.update(row[0] for row in connection.execute(sql, chunk).fetchall())
    return ready


def demand_candidates(
    connection: sqlite3.Connection,
    target_ids: list[str],
    coverage_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    """Return dependency-staged missing work without altering state."""

    targets = list(dict.fromkeys(target_ids))
    if len(targets) > MAX_DEMAND_ITEMS:
        raise ValueError("camera target set exceeds demand budget")
    for tile_id in targets:
        require_tile_id(tile_id)
    coverage = list(dict.fromkeys(coverage_ids or []))
    for tile_id in coverage:
        require_tile_id(tile_id)
    domain_ids = list(dict.fromkeys([*targets, *coverage]))
    ready_dem = _present_ids(connection, "dem", targets)
    ready_domain_dem = _present_ids(connection, "dem", domain_ids)
    ready_texture = _present_ids(connection, "textures", domain_ids)
    water_targets = list(
        dict.fromkeys(
            _contract_tile_id(tile_id)
            for tile_id in domain_ids
            if tile_id in ready_domain_dem
        )
    )
    ready_coast = _present_ids(
        connection, "coastline_masks", water_targets
    )
    ready_hydro = _present_ids(
        connection, "hydrography_masks", water_targets
    )
    ready_connectivity = _present_ids(
        connection, "tidal_connectivity_masks", water_targets
    )
    missing_connectivity = sorted(ready_hydro - ready_connectivity)
    connectivity_by_depth: dict[int, list[str]] = {}
    for tile_id in missing_connectivity:
        depth = require_tile_id(tile_id)[0]
        connectivity_by_depth.setdefault(depth, []).append(tile_id)
    connectivity_generations = []
    for depth, tile_ids in sorted(connectivity_by_depth.items()):
        generation = hashlib.sha256(
            "\n".join(tile_ids).encode("ascii")
        ).hexdigest()[:12]
        connectivity_generations.append(f"{depth}:{generation}")

    missing_texture_targets = [
        tile_id for tile_id in domain_ids if tile_id not in ready_texture
    ]
    bathymetry_jobs = sorted(eligible_fjord_jobs(connection, domain_ids))
    return {
        "dem": [tile_id for tile_id in targets if tile_id not in ready_dem],
        "texture": list(
            dict.fromkeys(
                _metatile_id(tile_id) for tile_id in missing_texture_targets
            )
        ),
        "coastline": [
            tile_id for tile_id in water_targets if tile_id not in ready_coast
        ],
        "hydrography": [
            tile_id for tile_id in water_targets if tile_id not in ready_hydro
        ],
        "connectivity": connectivity_generations,
        "bathymetry": bathymetry_jobs,
    }


def submit_camera_demand_from_selection(
    connection: sqlite3.Connection,
    selection: dict,
    coordinator: DemandCoordinator | None = None,
    *,
    demand_origin: str = "viewer",
) -> dict:
    if demand_origin not in {"viewer", "bathymetry"}:
        raise ValueError(f"unsupported terrain demand origin: {demand_origin}")
    target_ids, coverage_ids = prioritized_selection_ids(selection)
    lanes = coordinator or _coordinator()
    candidates = demand_candidates(connection, target_ids, coverage_ids)
    if demand_origin == "viewer":
        # The latest camera claim authoritatively replaces stale unstarted
        # viewer work in every lane.
        submitted = lanes.refresh(candidates)
    else:
        # The external bathymetry collector probes terrain availability. It
        # may request DEM/coastline prerequisites, but must never replace the
        # interactive camera queues or recursively schedule textures and
        # bathymetry jobs for its moving sweep.
        candidates = {
            name: candidates[name]
            for name in ("dem", "coastline")
        }
        submitted = lanes.submit(candidates, reopen_completed=True)
    return {
        "nonblocking": True,
        "waitedForWorkers": False,
        "candidates": candidates,
        "submitted": submitted,
        "lanes": lanes.status(),
    }


def prioritized_selection_ids(selection: dict) -> tuple[list[str], list[str]]:
    """Order desired and fallback IDs from nearest to farthest camera demand."""

    target_ids = selection.get("tileIds")
    if not isinstance(target_ids, list):
        raise TypeError("camera selection must contain tileIds")
    tiles = selection.get("tiles")
    if (
        isinstance(tiles, list)
        and len(tiles) == len(target_ids)
        and {tile.get("tileId") for tile in tiles if isinstance(tile, dict)}
        == set(target_ids)
        and all(
        isinstance(tile, dict)
        and isinstance(tile.get("tileId"), str)
        and isinstance(tile.get("distance"), (int, float))
        for tile in tiles
        )
    ):
        ordered_targets = [
            tile["tileId"]
            for tile in sorted(
                tiles,
                key=lambda tile: (float(tile["distance"]), tile["tileId"]),
            )
        ]
    else:
        ordered_targets = list(target_ids)

    rank = {tile_id: index for index, tile_id in enumerate(ordered_targets)}
    coverage = selection.get("coverage")
    if isinstance(coverage, list) and all(
        isinstance(tile, dict)
        and isinstance(tile.get("tileId"), str)
        and isinstance(tile.get("targetIds"), list)
        for tile in coverage
    ):
        ordered_coverage = [
            tile["tileId"]
            for tile in sorted(
                coverage,
                key=lambda tile: (
                    min(
                        (
                            rank.get(target_id, len(rank))
                            for target_id in tile["targetIds"]
                        ),
                        default=len(rank),
                    ),
                    tile["tileId"],
                ),
            )
        ]
    else:
        coverage_ids = selection.get("coverageTileIds", [])
        if not isinstance(coverage_ids, list):
            raise TypeError("camera selection coverageTileIds must be a list")
        ordered_coverage = list(coverage_ids)
    return ordered_targets, ordered_coverage


@visible
def submit_camera_demand(
    camera_x: float,
    camera_y: float,
    max_range: float = 16000.0,
    max_depth: int = WMS_CONTRACT_DEPTH,
    altitude: float = 0.0,
    previous_depth: int | None = None,
) -> dict:
    """Queue current camera misses in independent lanes and return immediately."""

    selection = select_lod_tiles(
        camera_x,
        camera_y,
        max_range,
        max_depth,
        altitude,
        previous_depth,
    )
    connection = db()
    selection.update(resolve_lod_coverage(connection, selection))
    demand = submit_camera_demand_from_selection(connection, selection)
    return {
        "targetTileCount": selection["tileCount"],
        "depthCap": selection["depthCap"],
        **demand,
    }


@visible
def demand_status() -> dict:
    """Return current lane activity without waiting for any worker."""

    lanes = _coordinator().status()
    return {
        "lanes": lanes,
        "polling": polling_state(lanes),
        "nonblocking": True,
    }


@visible
def compose_camera_demand_binary(
    camera_x: float,
    camera_y: float,
    max_range: float = 16000.0,
    max_depth: int = WMS_CONTRACT_DEPTH,
    altitude: float = 0.0,
    previous_depth: int | None = None,
    origin_x: float | None = None,
    origin_y: float | None = None,
    known_digests: dict[str, str] | None = None,
) -> dict:
    """Compose ready coverage, submit misses, and return without worker waits."""

    body, header = compose_camera_demand_binary_from_ready_data(
        db(),
        camera_x,
        camera_y,
        max_range,
        max_depth,
        altitude,
        previous_depth,
        origin_x,
        origin_y,
        known_digests,
    )
    return {
        "format": "binary-v1",
        "mediaType": "application/octet-stream",
        "contentLength": len(body),
        "digest": hashlib.sha256(body).hexdigest(),
        "tileCount": header["tileCount"],
        "targetTileCount": header["targetTileCount"],
        "missingTileCount": len(header["missing"]),
        "tilesReused": header["tilesReused"],
        "scheduledWork": header["scheduledWork"],
        "contentBase64": base64.b64encode(body).decode("ascii"),
    }


def _browser_pipeline_fields(demand: dict) -> dict:
    """Return the compact legacy fields consumed by the existing viewer."""

    lanes = demand["lanes"]
    polling = polling_state(lanes)

    def counts(name: str) -> tuple[int, int, int]:
        lane = lanes.get(
            name,
            {"claimedActive": [], "pending": [], "failures": {}},
        )
        active = len(lane["claimedActive"])
        pending = len(lane["pending"])
        retryable = sum(
            1
            for failure in lane["failures"].values()
            if failure.get("claimed") and failure.get("retryable")
        )
        return active, pending, retryable

    dem_active, dem_pending, dem_retryable = counts("dem")
    texture_active, texture_pending, texture_retryable = counts("texture")
    coastline_active, coastline_pending, _ = counts("coastline")
    hydro_active, hydro_pending, _ = counts("hydrography")
    connectivity_active, connectivity_pending, _ = counts("connectivity")
    bathymetry_active, bathymetry_pending, _ = counts("bathymetry")
    dem_actionable = bool(dem_active or dem_pending or dem_retryable)
    dem_retry_at = [
        failure["retryAt"]
        for failure in lanes["dem"]["failures"].values()
        if failure.get("claimed")
        and failure.get("retryable")
        and isinstance(failure.get("retryAt"), (int, float))
    ]
    dem_retry_after_ms = (
        max(1, int((min(dem_retry_at) - time.time()) * 1000))
        if dem_retry_at and not (dem_active or dem_pending)
        else None
    )
    return {
        "downloading": lanes["dem"]["claimedActive"],
        "polling": polling,
        "demActionable": dem_actionable,
        "demRetryAfterMs": dem_retry_after_ms,
        "texFetching": texture_active + texture_pending,
        "texRetryQueue": texture_retryable,
        "texStatusCounts": {},
        "coastlineQueued": (
            coastline_active
            + coastline_pending
            + hydro_active
            + hydro_pending
            + connectivity_active
            + connectivity_pending
        ),
        "bathymetryQueued": bathymetry_active + bathymetry_pending,
    }


def compose_camera_demand_binary_from_ready_data(
    connection: sqlite3.Connection,
    camera_x: float,
    camera_y: float,
    max_range: float = 16000.0,
    max_depth: int = WMS_CONTRACT_DEPTH,
    altitude: float = 0.0,
    previous_depth: int | None = None,
    origin_x: float | None = None,
    origin_y: float | None = None,
    known_digests: dict[str, str] | None = None,
    coordinator: DemandCoordinator | None = None,
    demand_origin: str = "viewer",
) -> tuple[bytes, dict]:
    """Return raw browser bytes while acquisition continues independently."""

    composition = compose_camera_from_ready_data(
        connection,
        camera_x,
        camera_y,
        max_range,
        max_depth,
        altitude,
        previous_depth,
        origin_x,
        origin_y,
    )
    selection = select_lod_tiles(
        camera_x,
        camera_y,
        max_range,
        max_depth,
        altitude,
        previous_depth,
    )
    selection["coverage"] = [
        {
            "tileId": tile["tileId"],
            "targetIds": tile.get("targetIds", []),
        }
        for tile in composition["tiles"]
    ]
    demand = submit_camera_demand_from_selection(
        connection,
        selection,
        coordinator,
        demand_origin=demand_origin,
    )
    composition.update(
        {
            **_browser_pipeline_fields(demand),
            "scheduledWork": any(
                lane["acceptedCount"]
                for lane in demand["submitted"].values()
            ),
        }
    )
    return encode_composed_tiles_binary(composition, known_digests)
