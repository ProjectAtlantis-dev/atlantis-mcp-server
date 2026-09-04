"""Connection lifecycle and initial inspection tools for Terrain assets."""

from __future__ import annotations

import builtins
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

import atlantis

from dynamic_functions.Terrain.Asset import schema


DATABASE_PATH = Path(__file__).with_name("assets.db")
_CONNECTION_KEY = "Terrain.Asset.connection.v1"
_CONNECTION_LOCK_KEY = "Terrain.Asset.connection_lock.v1"
_LOCK_INIT_GUARD = threading.Lock()


def connection_lock() -> threading.RLock:
    """Return the reload-safe lock protecting the shared connection."""
    lock = atlantis.server_shared.get(_CONNECTION_LOCK_KEY)
    if lock is not None:
        return lock
    with _LOCK_INIT_GUARD:
        lock = atlantis.server_shared.get(_CONNECTION_LOCK_KEY)
        if lock is None:
            lock = threading.RLock()
            atlantis.server_shared.set(_CONNECTION_LOCK_KEY, lock)
    return lock


def _connect() -> sqlite3.Connection:
    """Open the process-wide asset catalog connection."""
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30.0,
        check_same_thread=False,
    )
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _get_connection() -> sqlite3.Connection | None:
    """Return the asset connection without implicitly starting it."""
    return atlantis.server_shared.get(_CONNECTION_KEY)


def _close_connection() -> bool:
    """Close the shared connection and return whether it had been open."""
    connection = _get_connection()
    if connection is None:
        return False
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.close()
    atlantis.server_shared.remove(_CONNECTION_KEY)
    return True


def db() -> sqlite3.Connection:
    """Return the asset connection, starting it when necessary."""
    with connection_lock():
        connection = _get_connection()
        if connection is None:
            DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
            connection = _connect()
            try:
                schema.create(connection)
            except Exception:
                connection.close()
                raise
            atlantis.server_shared.set(_CONNECTION_KEY, connection)
        return connection


def _stopped_status() -> dict[str, Any]:
    return {
        "running": False,
        "path": str(DATABASE_PATH),
        "exists": DATABASE_PATH.exists(),
    }


async def _update_dashboard() -> None:
    """Refresh the composite Terrain dashboard after an Asset state change."""
    from dynamic_functions.Terrain.Database.database import _update_dashboard as update

    await update()


@visible
async def start() -> dict[str, Any]:
    """Open the local asset catalog and establish its schema."""
    db()
    result = status()
    await atlantis.client_log("Terrain asset catalog started")
    await _update_dashboard()
    return result


@visible
async def stop() -> dict[str, Any]:
    """Commit pending work and close the local asset catalog."""
    with connection_lock():
        _close_connection()
    result = _stopped_status()
    await atlantis.client_log("Terrain asset catalog stopped")
    await _update_dashboard()
    return result


@visible
def status() -> dict[str, Any]:
    """Report local asset catalog connection and row-count status."""
    connection = _get_connection()
    if connection is None:
        return _stopped_status()

    try:
        with connection_lock():
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            asset_count = connection.execute(
                "SELECT COUNT(*) FROM assets"
            ).fetchone()[0]
            enabled_count = connection.execute(
                "SELECT COUNT(*) FROM assets WHERE enabled = 1"
            ).fetchone()[0]
            type_rows = connection.execute(
                "SELECT type, COUNT(*) FROM assets GROUP BY type ORDER BY type"
            ).fetchall()
            metadata_count = connection.execute(
                "SELECT COUNT(*) FROM asset_metadata"
            ).fetchone()[0]
    except sqlite3.Error:
        return _stopped_status()

    return {
        "running": True,
        "path": str(DATABASE_PATH),
        "exists": True,
        "journal_mode": journal_mode,
        "asset_count": asset_count,
        "enabled_count": enabled_count,
        "metadata_count": metadata_count,
        "type_counts": [
            {"type": asset_type, "count": count}
            for asset_type, count in type_rows
        ],
    }


def _decoded_properties(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


@visible
def list(
    asset_type: str = "",
    enabled: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """List a bounded page of assets, optionally filtered by type and state."""
    if isinstance(limit, bool) or not 1 <= int(limit) <= 1000:
        raise ValueError("limit must be an integer from 1 to 1000")
    if isinstance(offset, bool) or int(offset) < 0:
        raise ValueError("offset must be a non-negative integer")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("enabled must be true, false, or null")

    normalized_type = str(asset_type).strip()
    clauses: builtins.list[str] = []
    parameters: builtins.list[Any] = []
    if normalized_type:
        clauses.append("type = ?")
        parameters.append(normalized_type)
    if enabled is not None:
        clauses.append("enabled = ?")
        parameters.append(1 if enabled else 0)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    with connection_lock():
        connection = db()
        total = connection.execute(
            f"SELECT COUNT(*) FROM assets{where}", parameters
        ).fetchone()[0]
        cursor = connection.execute(
            "SELECT id, type, enabled, lat, lon, heading_deg, z, properties, "
            "saved_at, updated_at, cx, cy, min_x, min_y, max_x, max_y "
            f"FROM assets{where} ORDER BY updated_at DESC, id LIMIT ? OFFSET ?",
            [*parameters, int(limit), int(offset)],
        )
        names = [column[0] for column in cursor.description]
        rows = []
        for values in cursor.fetchall():
            item = dict(zip(names, values))
            item["enabled"] = bool(item["enabled"])
            item["properties"] = _decoded_properties(item["properties"])
            rows.append(item)

    return {
        "assets": rows,
        "count": len(rows),
        "total": total,
        "limit": int(limit),
        "offset": int(offset),
    }


@visible
async def rebuild(
    source_directory: str = "",
    ground_samples_path: str = "",
) -> dict[str, Any]:
    """Atomically rebuild the catalog from local metadata, vectors, and terrain."""
    from dynamic_functions.Terrain.Asset.rebuild import (
        DEFAULT_SOURCE_DIRECTORY,
        DEFAULT_GROUND_SAMPLES_PATH,
        build_catalog,
        temporary_output_path,
    )

    sources = (
        Path(source_directory).expanduser().resolve()
        if str(source_directory).strip()
        else DEFAULT_SOURCE_DIRECTORY
    )
    ground_samples = (
        Path(ground_samples_path).expanduser().resolve()
        if str(ground_samples_path).strip()
        else DEFAULT_GROUND_SAMPLES_PATH
    )
    temporary_path = temporary_output_path(DATABASE_PATH)
    was_running = False
    try:
        with connection_lock():
            was_running = _close_connection()
            result = build_catalog(
                temporary_path,
                source_directory=sources,
                ground_samples_path=ground_samples,
            )
            temporary_path.replace(DATABASE_PATH)
            if was_running:
                db()
    except Exception:
        for path in (
            temporary_path,
            Path(f"{temporary_path}-wal"),
            Path(f"{temporary_path}-shm"),
        ):
            path.unlink(missing_ok=True)
        if was_running and _get_connection() is None and DATABASE_PATH.exists():
            db()
        raise
    result.update({
        "ok": True,
        "path": str(DATABASE_PATH),
        "source_directory": str(sources),
        "ground_samples_path": str(ground_samples),
        "running": _get_connection() is not None,
    })
    await atlantis.client_log(
        f"Terrain asset catalog rebuilt with {result['asset_count']} assets"
    )
    await _update_dashboard()
    return result


def ux_status() -> str:
    """Build the dashboard component for the local asset catalog status."""
    uid = uuid.uuid4().hex[:8]
    running = bool(status()["running"])
    if running:
        light_color = "#22c55e"
        light_glow = "34, 197, 94"
    else:
        light_color = "#ef4444"
        light_glow = "239, 68, 68"
    state_label = "on" if running else "off"

    return f"""
<style>
  #terrain-asset-status-{uid} {{
    box-sizing: border-box;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 14px;
    align-items: center;
    width: 100%;
    padding: 4.8px;
    color: #fffaf0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #terrain-asset-status-{uid} .terrain-asset-label {{
    justify-self: start;
    margin: 0;
    color: rgba(42, 42, 42, 0.92);
    font-family: "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
    font-size: 17px;
    font-stretch: condensed;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-shadow:
      0 -1px 0 rgba(0, 0, 0, 0.72),
      0 1px 0 rgba(255, 255, 255, 0.52);
  }}
  #terrain-asset-status-{uid} .terrain-asset-light {{
    justify-self: end;
    width: 34px;
    height: 5px;
    background: {light_color};
    border-radius: 1px;
    box-shadow:
      0 0 5px rgba({light_glow}, 0.72),
      0 0 11px rgba({light_glow}, 0.34);
  }}
</style>
<div id="terrain-asset-status-{uid}" aria-label="Terrain asset catalog status">
  <span class="terrain-asset-label">ASSET DB</span>
  <span class="terrain-asset-light" role="status" aria-label="{state_label}"></span>
</div>
"""
