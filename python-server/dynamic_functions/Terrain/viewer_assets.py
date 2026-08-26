"""Read-only startup-asset and building compatibility for the viewer."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import zlib
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
_METADATA_PATH = _HERE / "assets_metadata.json"
_LOCAL_ASSETS_DB = _HERE / "Database" / "assets.db"
_LEGACY_ASSETS_DB = _HERE.parents[3] / "atlantis-terrain" / "assetserver" / "assets.db"
_BUILDING_TYPE = "BYGNING"
_BUILDING_FULL_DETAIL_RANGE_M = 2500.0
_BUILDING_FAR_MIN_AREA_M2 = 300.0


def _metadata() -> dict[str, Any]:
    value = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("asset metadata root must be an object")
    definition = value.get("vehicle_definition")
    if isinstance(definition, dict):
        definition = dict(definition)
        headlights = definition.get("headlights")
        if isinstance(headlights, dict):
            headlights = dict(headlights)
            color = headlights.get("color")
            if isinstance(color, str) and color.startswith("#"):
                headlights["color"] = int(color[1:], 16)
            definition["headlights"] = headlights
        value["vehicle_definition"] = definition
    return value


def resolve_assets_db_path() -> Path | None:
    configured = os.environ.get("ATLANTIS_ASSETS_DB")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((_LOCAL_ASSETS_DB, _LEGACY_ASSETS_DB))
    return next((path for path in candidates if path.is_file()), None)


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


def startup_assets() -> dict[str, Any]:
    metadata = _metadata()
    path = resolve_assets_db_path()
    vehicle_type = str(metadata.get("vehicle_asset_type") or "")
    structure_type = str(metadata.get("structure_asset_type") or "")
    vehicles: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    if path is not None:
        connection = _connect_read_only(path)
        try:
            for row in connection.execute(
                "SELECT id,lat,lon,heading_deg,z,properties,saved_at "
                "FROM assets WHERE enabled=1 AND type=? "
                "ORDER BY updated_at DESC,id",
                (vehicle_type,),
            ):
                props = _decoded_properties(row[5])
                item = {
                    "id": row[0], "lat": row[1], "lon": row[2],
                    "headingDeg": row[3], "headlightsOn": props.get("headlightsOn", True),
                    "savedAt": row[6] or 0,
                }
                if row[4] is not None:
                    item["z"] = row[4]
                for key in ("terrainDepth", "terrainTileId"):
                    if props.get(key) is not None:
                        item[key] = props[key]
                vehicles.append(item)
            for row in connection.execute(
                "SELECT id,lat,lon,heading_deg,properties FROM assets "
                "WHERE enabled=1 AND type=? ORDER BY updated_at DESC,id",
                (structure_type,),
            ):
                props = _decoded_properties(row[4])
                item = {
                    "id": row[0], "lat": row[1], "lon": row[2],
                    "headingDeg": row[3], "scale": props.get("scale", 1),
                }
                if props.get("tileId"):
                    item["tileId"] = props["tileId"]
                structures.append(item)
        finally:
            connection.close()
    if not vehicles:
        vehicles = [dict(item) for item in metadata.get("seed_vehicle_instances", [])]
    return {
        "ok": True,
        "source": "asset_catalog" if path is not None else "metadata_fallback",
        "catalogStatus": "ready" if path is not None else "unavailable",
        "catalogPath": str(path) if path is not None else None,
        "schemaVersion": 4,
        "seeded": {"structureInstances": False, "vehicleInstances": False},
        "vehicle_definition": metadata.get("vehicle_definition", {}),
        "structure_definition": metadata.get("structure_definition", {}),
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
    path = resolve_assets_db_path()
    if path is None:
        return [], "unavailable"
    connection = _connect_read_only(path)
    try:
        rows = connection.execute(
            "SELECT id,properties FROM assets WHERE type=? AND enabled=1 "
            "AND cx BETWEEN ? AND ? AND cy BETWEEN ? AND ? LIMIT 20000",
            (_BUILDING_TYPE, qx - max_range, qx + max_range, qy - max_range, qy + max_range),
        ).fetchall()
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
            buildings.append({
                "id": str(asset_id),
                "groundZ": float(props.get("groundZ", 0.0)),
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
            "groundZ": building.get("groundZ", 0),
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
        "buildingsStatus": "ready" if source != "unavailable" else "unavailable",
        "buildingsSource": source,
        "qx": qx, "qy": qy, "ox": ox, "oy": oy,
    }
    header = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header += b" " * (-(len(header) + 4) % 4)
    return b"".join((struct.pack("<I", len(header)), header, *blobs))
