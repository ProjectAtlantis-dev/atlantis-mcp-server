"""Strict reads and writes for the MCP-owned Terrain asset catalog."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any

from dynamic_functions.Terrain.Asset import database


class AssetCatalogUnavailable(RuntimeError):
    """The local catalog is missing or does not satisfy its contract."""


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    try:
        rows = connection.execute("SELECT key, value FROM asset_metadata").fetchall()
    except sqlite3.Error as exc:
        raise AssetCatalogUnavailable(f"asset metadata is unreadable: {exc}") from exc
    raw = dict(rows)
    required = {
        "schema_version",
        "vehicle_asset_type",
        "structure_asset_type",
        "vehicle_definition",
        "structure_definition",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise AssetCatalogUnavailable(
            "asset metadata is incomplete; missing: " + ", ".join(missing)
        )
    try:
        return {key: json.loads(raw[key]) for key in required}
    except (TypeError, json.JSONDecodeError) as exc:
        raise AssetCatalogUnavailable(f"asset metadata is invalid JSON: {exc}") from exc


def _connection() -> sqlite3.Connection:
    if not database.DATABASE_PATH.is_file():
        raise AssetCatalogUnavailable(
            f"local asset catalog is missing: {database.DATABASE_PATH}"
        )
    connection = database.db()
    _metadata(connection)
    return connection


def save_vehicle_state(payload: Any) -> dict[str, Any]:
    """Persist the primary vehicle state using the viewer's existing contract."""
    source = _object(payload, "vehicle state payload")
    lat = _finite_number(source.get("lat"), "lat")
    lon = _finite_number(source.get("lon"), "lon")
    heading = _finite_number(source.get("headingDeg"), "headingDeg") % 360.0
    if not -90.0 <= lat <= 90.0:
        raise ValueError("lat must be between -90 and 90")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("lon must be between -180 and 180")
    if "z" not in source or source["z"] is None:
        raise ValueError("z is required")
    z = _finite_number(source["z"], "z")

    terrain_depth = None
    if source.get("terrainDepth") is not None:
        terrain_depth_value = _finite_number(source["terrainDepth"], "terrainDepth")
        if terrain_depth_value < 0:
            raise ValueError("terrainDepth must be non-negative")
        terrain_depth = math.floor(terrain_depth_value)
    terrain_tile_id = None
    if source.get("terrainTileId") is not None:
        terrain_tile_id = str(source["terrainTileId"]).strip()
        if not terrain_tile_id:
            raise ValueError("terrainTileId must be non-empty when supplied")

    with database.connection_lock():
        connection = _connection()
        metadata = _metadata(connection)
        vehicle_type = metadata["vehicle_asset_type"]
        row = connection.execute(
            "SELECT id, properties FROM assets WHERE type=? AND enabled=1 "
            "ORDER BY updated_at DESC, id LIMIT 1",
            (vehicle_type,),
        ).fetchone()
        if row is None:
            row = connection.execute(
                "SELECT id, properties FROM assets WHERE type=? "
                "ORDER BY updated_at DESC, id LIMIT 1",
                (vehicle_type,),
            ).fetchone()
        if row is None:
            raise AssetCatalogUnavailable(
                "local asset catalog contains no vehicle asset to update"
            )
        vehicle_id, raw_properties = row
        try:
            existing_properties = json.loads(raw_properties)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AssetCatalogUnavailable(
                f"vehicle asset {vehicle_id!r} has invalid properties"
            ) from exc
        if not isinstance(existing_properties, dict):
            raise AssetCatalogUnavailable(
                f"vehicle asset {vehicle_id!r} properties are not an object"
            )
        if "headlightsOn" not in existing_properties:
            raise AssetCatalogUnavailable(
                f"vehicle asset {vehicle_id!r} is missing headlightsOn"
            )
        properties: dict[str, Any] = {
            "headlightsOn": bool(existing_properties["headlightsOn"])
        }
        if terrain_depth is not None:
            properties["terrainDepth"] = terrain_depth
        if terrain_tile_id is not None:
            properties["terrainTileId"] = terrain_tile_id
        saved_at = time.time()
        connection.execute(
            "UPDATE assets SET enabled=1,lat=?,lon=?,heading_deg=?,z=?,"
            "properties=?,saved_at=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (
                lat,
                lon,
                heading,
                z,
                json.dumps(properties, ensure_ascii=False, separators=(",", ":")),
                saved_at,
                vehicle_id,
            ),
        )
        connection.commit()
    state: dict[str, Any] = {
        "lat": lat,
        "lon": lon,
        "headingDeg": heading,
        "z": z,
        "savedAt": saved_at,
    }
    if terrain_depth is not None:
        state["terrainDepth"] = terrain_depth
    if terrain_tile_id is not None:
        state["terrainTileId"] = terrain_tile_id
    return {"ok": True, "vehicleId": vehicle_id, "state": state}


def patch_asset(asset_id: str, payload: Any) -> dict[str, Any] | None:
    """Patch the mutable fields of one existing asset."""
    normalized_id = str(asset_id).strip()
    if not normalized_id:
        raise ValueError("missing asset id")
    source = _object(payload, "request body")
    allowed = {"enabled", "lat", "lon", "headingDeg", "z", "properties"}
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ValueError("unknown asset patch fields: " + ", ".join(unknown))

    with database.connection_lock():
        connection = _connection()
        row = connection.execute(
            "SELECT id,type,enabled,lat,lon,heading_deg,z,properties "
            "FROM assets WHERE id=?",
            (normalized_id,),
        ).fetchone()
        if row is None:
            return None
        row_id, asset_type, enabled, lat, lon, heading, z, raw_properties = row
        if "enabled" in source:
            enabled = 1 if _boolean(source["enabled"], "enabled") else 0
        if "lat" in source:
            lat = _finite_number(source["lat"], "lat")
            if not -90.0 <= lat <= 90.0:
                raise ValueError("lat must be between -90 and 90")
        if "lon" in source:
            lon = _finite_number(source["lon"], "lon")
            if not -180.0 <= lon <= 180.0:
                raise ValueError("lon must be between -180 and 180")
        if "headingDeg" in source:
            heading = _finite_number(source["headingDeg"], "headingDeg") % 360.0
        if "z" in source:
            z = None if source["z"] is None else _finite_number(source["z"], "z")
        try:
            properties = json.loads(raw_properties)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AssetCatalogUnavailable(
                f"asset {normalized_id!r} has invalid properties"
            ) from exc
        if not isinstance(properties, dict):
            raise AssetCatalogUnavailable(
                f"asset {normalized_id!r} properties are not an object"
            )
        if "properties" in source:
            properties.update(_object(source["properties"], "properties"))
        connection.execute(
            "UPDATE assets SET enabled=?,lat=?,lon=?,heading_deg=?,z=?,"
            "properties=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (
                enabled,
                lat,
                lon,
                heading,
                z,
                json.dumps(properties, ensure_ascii=False, separators=(",", ":")),
                normalized_id,
            ),
        )
        connection.commit()
    return {
        "ok": True,
        "id": row_id,
        "type": asset_type,
        "enabled": bool(enabled),
        "lat": lat,
        "lon": lon,
        "headingDeg": heading,
        "z": z,
        "properties": properties,
    }
