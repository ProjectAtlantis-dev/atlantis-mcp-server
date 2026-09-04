"""Read-only startup-asset and building compatibility for the viewer."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import zlib
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
_LOCAL_ASSETS_DB = _HERE / "Asset" / "assets.db"
_BUILDING_TYPE = "BYGNING"
_BUILDING_FULL_DETAIL_RANGE_M = 2500.0
_BUILDING_FAR_MIN_AREA_M2 = 300.0


class AssetCatalogUnavailable(RuntimeError):
    """The MCP-owned asset catalog is absent or incomplete."""


def resolve_assets_db_path() -> Path | None:
    """Resolve only the MCP-owned asset catalog path."""
    return _LOCAL_ASSETS_DB if _LOCAL_ASSETS_DB.is_file() else None


def _required_assets_db_path() -> Path:
    path = resolve_assets_db_path()
    if path is None:
        raise AssetCatalogUnavailable(
            f"local asset catalog is missing: {_LOCAL_ASSETS_DB}"
        )
    return path


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro", uri=True, timeout=5.0
    )
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _decoded_properties(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("asset properties must decode to an object")
    return value


def _catalog_metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    required = {
        "schema_version",
        "vehicle_asset_type",
        "structure_asset_type",
        "vehicle_definition",
        "structure_definition",
    }
    try:
        rows = connection.execute(
            "SELECT key, value FROM asset_metadata"
        ).fetchall()
    except sqlite3.Error as exc:
        raise AssetCatalogUnavailable(
            f"local asset catalog metadata is unreadable: {exc}"
        ) from exc
    raw = {key: value for key, value in rows}
    missing = sorted(required - raw.keys())
    if missing:
        raise AssetCatalogUnavailable(
            "local asset catalog metadata is incomplete; missing: "
            + ", ".join(missing)
        )
    try:
        metadata = {key: json.loads(raw[key]) for key in required}
    except (TypeError, json.JSONDecodeError) as exc:
        raise AssetCatalogUnavailable(
            f"local asset catalog metadata is invalid JSON: {exc}"
        ) from exc
    if not isinstance(metadata["schema_version"], int):
        raise AssetCatalogUnavailable("schema_version metadata must be an integer")
    for key in ("vehicle_asset_type", "structure_asset_type"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise AssetCatalogUnavailable(f"{key} metadata must be a non-empty string")
    for key in ("vehicle_definition", "structure_definition"):
        if not isinstance(metadata[key], dict):
            raise AssetCatalogUnavailable(f"{key} metadata must be an object")
    vehicle_definition = metadata["vehicle_definition"]
    if (
        not isinstance(vehicle_definition.get("url"), str)
        or not vehicle_definition["url"].strip()
    ):
        raise AssetCatalogUnavailable(
            "vehicle_definition.url metadata must be a non-empty string"
        )
    for key in ("realLengthM", "tireDiameterM", "altOffsetM"):
        value = vehicle_definition.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssetCatalogUnavailable(
                f"vehicle_definition.{key} metadata must be numeric"
            )
    return metadata


def startup_assets() -> dict[str, Any]:
    path = _required_assets_db_path()
    vehicles: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    connection = _connect_read_only(path)
    try:
        metadata = _catalog_metadata(connection)
        for row in connection.execute(
            "SELECT id,lat,lon,heading_deg,z,properties,saved_at "
            "FROM assets WHERE enabled=1 AND type=? "
            "ORDER BY updated_at DESC,id",
            (metadata["vehicle_asset_type"],),
        ):
            props = _decoded_properties(row[5])
            if "headlightsOn" not in props:
                raise AssetCatalogUnavailable(
                    f"vehicle asset {row[0]!r} is missing headlightsOn"
                )
            if row[4] is None:
                raise AssetCatalogUnavailable(
                    f"vehicle asset {row[0]!r} is missing z"
                )
            item = {
                "id": row[0], "lat": row[1], "lon": row[2],
                "headingDeg": row[3], "headlightsOn": props["headlightsOn"],
                "savedAt": row[6],
                "z": row[4],
            }
            for key in ("terrainDepth", "terrainTileId"):
                if props.get(key) is not None:
                    item[key] = props[key]
            vehicles.append(item)
        for row in connection.execute(
            "SELECT id,lat,lon,heading_deg,properties FROM assets "
            "WHERE enabled=1 AND type=? ORDER BY updated_at DESC,id",
            (metadata["structure_asset_type"],),
        ):
            props = _decoded_properties(row[4])
            if "scale" not in props:
                raise AssetCatalogUnavailable(
                    f"structure asset {row[0]!r} is missing scale"
                )
            item = {
                "id": row[0], "lat": row[1], "lon": row[2],
                "headingDeg": row[3], "scale": props["scale"],
            }
            if props.get("tileId"):
                item["tileId"] = props["tileId"]
            structures.append(item)
        if not vehicles:
            raise AssetCatalogUnavailable(
                "local asset catalog contains no enabled vehicle assets"
            )
    except sqlite3.Error as exc:
        raise AssetCatalogUnavailable(
            f"local asset catalog is unreadable: {exc}"
        ) from exc
    finally:
        connection.close()
    return {
        "ok": True,
        "source": "asset_catalog",
        "catalogStatus": "ready",
        "catalogPath": str(path),
        "schemaVersion": metadata["schema_version"],
        "vehicle_definition": metadata["vehicle_definition"],
        "structure_definition": metadata["structure_definition"],
        "vehicle_instances": vehicles,
        "structure_instances": structures,
    }


def _ring_area_and_center(ring: list[list[float]]) -> tuple[float, float, float]:
    if len(ring) < 3:
        return 0.0, 0.0, 0.0
    twice_area = 0.0
    previous = ring[-1]
    for point in ring:
        twice_area += previous[0] * point[1] - point[0] * previous[1]
        previous = point
    return (
        abs(twice_area) / 2.0,
        sum(point[0] for point in ring) / len(ring),
        sum(point[1] for point in ring) / len(ring),
    )


def query_buildings(qx: float, qy: float, max_range: float, ox: float, oy: float) -> tuple[list[dict], str]:
    path = _required_assets_db_path()
    connection = _connect_read_only(path)
    try:
        rows = connection.execute(
            "SELECT id,properties FROM assets WHERE type=? AND enabled=1 "
            "AND cx BETWEEN ? AND ? AND cy BETWEEN ? AND ? LIMIT 20000",
            (_BUILDING_TYPE, qx - max_range, qx + max_range, qy - max_range, qy + max_range),
        ).fetchall()
    except sqlite3.Error as exc:
        raise AssetCatalogUnavailable(
            f"local building catalog is unreadable: {exc}"
        ) from exc
    finally:
        connection.close()
    buildings = []
    for asset_id, raw in rows:
        props = _decoded_properties(raw)
        ring = props.get("ring")
        if not isinstance(ring, list) or len(ring) < 3:
            continue
        area, center_x, center_y = _ring_area_and_center(ring)
        if (
            math.hypot(center_x - qx, center_y - qy) > _BUILDING_FULL_DETAIL_RANGE_M
            and area < _BUILDING_FAR_MIN_AREA_M2
        ):
            continue
        relative = [
            [float(point[0]) - ox, float(point[1]) - oy, float(point[2])]
            for point in ring
            if isinstance(point, list) and len(point) >= 3
        ]
        if len(relative) >= 3:
            if "groundZ" not in props:
                raise AssetCatalogUnavailable(
                    f"building asset {asset_id!r} is missing groundZ"
                )
            buildings.append({
                "id": str(asset_id),
                "groundZ": float(props["groundZ"]),
                "ring": relative,
            })
    return buildings, "asset_catalog"


def encode_buildings_response(
    buildings: list[dict], *, qx: float, qy: float, ox: float, oy: float,
    source: str,
) -> bytes:
    entries = []
    blobs = []
    digest = 0
    for building in buildings:
        blob = b"".join(
            struct.pack("<fff", float(point[0]), float(point[1]), float(point[2]))
            for point in building["ring"]
        )
        entry = {
            "id": building["id"],
            "groundZ": building["groundZ"],
            "ringBytes": len(blob),
        }
        entries.append(entry)
        blobs.append(blob)
    entries_json = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = zlib.crc32(entries_json, digest)
    for blob in blobs:
        digest = zlib.crc32(blob, digest)
    payload = {
        "tiles": [], "buildings": entries, "count": len(entries),
        "buildingsHash": f"{digest & 0xFFFFFFFF:08x}",
        "buildingsStatus": "ready",
        "buildingsSource": source,
        "qx": qx, "qy": qy, "ox": ox, "oy": oy,
    }
    header = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header += b" " * (-(len(header) + 4) % 4)
    return b"".join((struct.pack("<I", len(header)), header, *blobs))
