"""Foreground-only rebuild of the MCP-owned asset catalog."""

from __future__ import annotations

import datetime
import json
import math
import re
import sqlite3
import struct
import uuid
import zipfile
from pathlib import Path
from typing import Any, Iterator

from pyproj import Transformer

from dynamic_functions.Terrain.Asset import schema
from dynamic_functions.Terrain.coords import to_wgs84


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIRECTORY = HERE / "grundkort"
DEFAULT_METADATA_PATH = HERE / "metadata.json"
DEFAULT_GROUND_SAMPLES_PATH = HERE / "building_ground_samples.json"
BUILDING_LAYER = "BYGNING"
ROAD_LAYERS = ("VEJMIDTE", "STIMIDTE")


class AssetRebuildError(RuntimeError):
    """The source data cannot produce a complete local catalog."""


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssetRebuildError(f"asset metadata file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AssetRebuildError(f"asset metadata is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssetRebuildError("asset metadata root must be an object")
    required = {
        "schemaVersion",
        "vehicleAssetType",
        "structureAssetType",
        "grundkortSettlements",
        "vehicleDefinition",
        "structureDefinition",
        "seedVehicleInstances",
        "seedStructureInstances",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise AssetRebuildError(
            "asset metadata is incomplete; missing: " + ", ".join(missing)
        )
    if not isinstance(payload["schemaVersion"], int):
        raise AssetRebuildError("schemaVersion must be an integer")
    for key in ("vehicleAssetType", "structureAssetType"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise AssetRebuildError(f"{key} must be a non-empty string")
    for key in ("vehicleDefinition", "structureDefinition"):
        if not isinstance(payload[key], dict):
            raise AssetRebuildError(f"{key} must be an object")
    for key in ("seedVehicleInstances", "seedStructureInstances"):
        if not isinstance(payload[key], list):
            raise AssetRebuildError(f"{key} must be an array")
    settlements = payload["grundkortSettlements"]
    if (
        not isinstance(settlements, list)
        or not settlements
        or any(not isinstance(value, str) or not value for value in settlements)
        or len(set(settlements)) != len(settlements)
    ):
        raise AssetRebuildError(
            "grundkortSettlements must be a non-empty array of unique strings"
        )
    return payload


def _read_ground_samples(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssetRebuildError(
            f"building ground sample registry is missing: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AssetRebuildError(
            f"building ground sample registry is invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise AssetRebuildError("building ground sample registry schemaVersion must be 1")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, dict):
        raise AssetRebuildError("building ground sample registry samples must be an object")
    samples: dict[str, float] = {}
    for asset_id, raw_value in raw_samples.items():
        if not isinstance(asset_id, str) or not asset_id:
            raise AssetRebuildError("building ground sample IDs must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise AssetRebuildError(
                f"building ground sample {asset_id!r} must be numeric"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise AssetRebuildError(
                f"building ground sample {asset_id!r} must be finite"
            )
        samples[asset_id] = value
    if not samples:
        raise AssetRebuildError("building ground sample registry is empty")
    return samples


def _decode_dbf_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin1")


def _read_dbf_records(data: bytes) -> list[dict[str, str]]:
    if len(data) < 32:
        raise AssetRebuildError("DBF payload is truncated")
    record_count = struct.unpack("<I", data[4:8])[0]
    header_length = struct.unpack("<H", data[8:10])[0]
    field_count = (header_length - 33) // 32
    fields = []
    for index in range(field_count):
        descriptor = data[32 + index * 32:64 + index * 32]
        name = descriptor[:11].split(b"\x00")[0].decode("latin1")
        fields.append((name, descriptor[16]))
    record_length = sum(length for _, length in fields) + 1
    records = []
    for index in range(record_count):
        record = data[
            header_length + index * record_length:
            header_length + (index + 1) * record_length
        ]
        if len(record) != record_length:
            raise AssetRebuildError("DBF record payload is truncated")
        position = 1
        row = {}
        for name, length in fields:
            row[name] = _decode_dbf_text(record[position:position + length]).strip()
            position += length
        records.append(row)
    return records


def _shape_records(data: bytes, expected_type: int) -> Iterator[bytes | None]:
    if len(data) < 100:
        raise AssetRebuildError("shapefile payload is truncated")
    file_length = struct.unpack(">i", data[24:28])[0] * 2
    position = 100
    while position < file_length:
        if position + 8 > len(data):
            raise AssetRebuildError("shapefile record header is truncated")
        content_length = struct.unpack(">i", data[position + 4:position + 8])[0] * 2
        content = data[position + 8:position + 8 + content_length]
        position += 8 + content_length
        if len(content) != content_length or len(content) < 4:
            raise AssetRebuildError("shapefile record is truncated")
        record_type = struct.unpack("<i", content[:4])[0]
        yield content if record_type == expected_type else None


def _polygonz_outer_rings(data: bytes) -> Iterator[list[tuple[float, float, float]] | None]:
    for content in _shape_records(data, 15):
        if content is None:
            yield None
            continue
        part_count, point_count = struct.unpack("<2i", content[36:44])
        offset = 44
        parts = struct.unpack(
            f"<{part_count}i", content[offset:offset + 4 * part_count]
        )
        offset += 4 * part_count
        xy = struct.unpack(
            f"<{2 * point_count}d", content[offset:offset + 16 * point_count]
        )
        offset += 16 * point_count + 16
        elevations = struct.unpack(
            f"<{point_count}d", content[offset:offset + 8 * point_count]
        )
        end = parts[1] if part_count > 1 else point_count
        yield [
            (xy[2 * index], xy[2 * index + 1], elevations[index])
            for index in range(parts[0], end)
        ]


def _polylinez_parts(data: bytes) -> Iterator[list[list[tuple[float, float, float]]] | None]:
    for content in _shape_records(data, 13):
        if content is None:
            yield None
            continue
        part_count, point_count = struct.unpack("<2i", content[36:44])
        offset = 44
        parts = struct.unpack(
            f"<{part_count}i", content[offset:offset + 4 * part_count]
        )
        offset += 4 * part_count
        xy = struct.unpack(
            f"<{2 * point_count}d", content[offset:offset + 16 * point_count]
        )
        offset += 16 * point_count + 16
        elevations = struct.unpack(
            f"<{point_count}d", content[offset:offset + 8 * point_count]
        )
        bounds = [*parts, point_count]
        yield [
            [
                (xy[2 * index], xy[2 * index + 1], elevations[index])
                for index in range(bounds[part], bounds[part + 1])
            ]
            for part in range(part_count)
        ]


def _source_epsg(prj_text: str) -> int:
    match = re.search(r"UTM[_ ]Zone[_ ](\d+)N", prj_text, re.IGNORECASE)
    if match is None:
        raise AssetRebuildError(
            f"cannot find a GR96 UTM zone in PRJ: {prj_text[:120]}"
        )
    return 3160 + int(match.group(1))


def _settlement_code(path: Path) -> str:
    match = re.match(r"(\d{4}[A-Z]{3})", path.name)
    if match is None:
        raise AssetRebuildError(
            f"cannot infer settlement code from archive name: {path.name}"
        )
    return match.group(1)


def _archive_members(archive: zipfile.ZipFile) -> dict[str, str]:
    return {Path(name).name.upper(): name for name in archive.namelist()}


def _required_member(
    archive: zipfile.ZipFile, members: dict[str, str], name: str
) -> bytes:
    try:
        return archive.read(members[name.upper()])
    except KeyError as exc:
        raise AssetRebuildError(f"archive is missing required member {name}") from exc


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _store_metadata(connection: sqlite3.Connection, metadata: dict[str, Any]) -> None:
    values = {
        "schema_version": metadata["schemaVersion"],
        "vehicle_asset_type": metadata["vehicleAssetType"],
        "structure_asset_type": metadata["structureAssetType"],
        "vehicle_definition": metadata["vehicleDefinition"],
        "structure_definition": metadata["structureDefinition"],
    }
    connection.executemany(
        "INSERT INTO asset_metadata(key,value) VALUES (?,?)",
        [(key, _json(value)) for key, value in values.items()],
    )


def _seed_assets(
    connection: sqlite3.Connection, metadata: dict[str, Any], now: str
) -> dict[str, int]:
    counts = {metadata["vehicleAssetType"]: 0, metadata["structureAssetType"]: 0}
    for seed in metadata["seedVehicleInstances"]:
        if not isinstance(seed, dict):
            raise AssetRebuildError("every seedVehicleInstances entry must be an object")
        required = {"id", "lat", "lon", "headingDeg", "z", "headlightsOn"}
        missing = sorted(required - seed.keys())
        if missing:
            raise AssetRebuildError(
                "vehicle seed is incomplete; missing: " + ", ".join(missing)
            )
        connection.execute(
            "INSERT INTO assets(id,type,enabled,lat,lon,heading_deg,z,properties,"
            "saved_at,updated_at) VALUES (?,?,1,?,?,?,?,?,NULL,?)",
            (
                str(seed["id"]),
                metadata["vehicleAssetType"],
                float(seed["lat"]),
                float(seed["lon"]),
                float(seed["headingDeg"]) % 360.0,
                float(seed["z"]),
                _json({"headlightsOn": bool(seed["headlightsOn"])}),
                now,
            ),
        )
        counts[metadata["vehicleAssetType"]] += 1
    if not counts[metadata["vehicleAssetType"]]:
        raise AssetRebuildError("metadata contains no seed vehicle instances")
    for seed in metadata["seedStructureInstances"]:
        if not isinstance(seed, dict):
            raise AssetRebuildError("every seedStructureInstances entry must be an object")
        required = {"id", "lat", "lon", "headingDeg", "scale"}
        missing = sorted(required - seed.keys())
        if missing:
            raise AssetRebuildError(
                "structure seed is incomplete; missing: " + ", ".join(missing)
            )
        properties: dict[str, Any] = {"scale": float(seed["scale"])}
        if seed.get("tileId") is not None:
            properties["tileId"] = str(seed["tileId"])
        connection.execute(
            "INSERT INTO assets(id,type,enabled,lat,lon,heading_deg,z,properties,"
            "updated_at) VALUES (?,?,1,?,?,?,?,?,?)",
            (
                str(seed["id"]),
                metadata["structureAssetType"],
                float(seed["lat"]),
                float(seed["lon"]),
                float(seed["headingDeg"]) % 360.0,
                None,
                _json(properties),
                now,
            ),
        )
        counts[metadata["structureAssetType"]] += 1
    return counts


def _ingest_buildings(
    connection: sqlite3.Connection,
    archive: zipfile.ZipFile,
    members: dict[str, str],
    settlement: str,
    transformer: Transformer,
    ground_samples: dict[str, float],
    now: str,
) -> int:
    attributes = _read_dbf_records(
        _required_member(archive, members, f"{BUILDING_LAYER}.DBF")
    )
    rings = list(
        _polygonz_outer_rings(
            _required_member(archive, members, f"{BUILDING_LAYER}.SHP")
        )
    )
    if len(attributes) != len(rings):
        raise AssetRebuildError(
            f"{settlement} building DBF/shape count mismatch: "
            f"{len(attributes)} != {len(rings)}"
        )
    written = 0
    for index, ring in enumerate(rings):
        if not ring or len(ring) < 3:
            continue
        if ring[0][:2] == ring[-1][:2]:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        source_x, source_y, elevations = zip(*ring)
        projected_x, projected_y = transformer.transform(source_x, source_y)
        center_x = sum(projected_x) / len(projected_x)
        center_y = sum(projected_y) / len(projected_y)
        row = attributes[index]
        building_id = f"{settlement}_{row.get('lokal_id', '') or index}"
        ground = ground_samples.get(building_id)
        if ground is None:
            raise AssetRebuildError(
                f"building {building_id!r} has no authoritative ground sample"
            )
        roof_minimum = min(elevations)
        ground = min(ground, roof_minimum - 0.5)
        ring_3413 = [
            [round(x, 2), round(y, 2), round(z, 2)]
            for x, y, z in zip(projected_x, projected_y, elevations)
        ]
        latitude, longitude = to_wgs84(center_x, center_y)
        properties = {
            "sourceLayer": BUILDING_LAYER,
            "sourceProperties": row,
            "groundZ": round(ground, 2),
            "groundSampled": True,
            "ring": ring_3413,
        }
        connection.execute(
            "INSERT INTO assets(id,type,enabled,lat,lon,heading_deg,z,properties,"
            "cx,cy,min_x,min_y,max_x,max_y,updated_at) "
            "VALUES (?,?,1,?,?,0,?,?,?,?,?,?,?,?,?)",
            (
                building_id,
                BUILDING_LAYER,
                float(latitude),
                float(longitude),
                round(ground, 2),
                _json(properties),
                center_x,
                center_y,
                min(projected_x),
                min(projected_y),
                max(projected_x),
                max(projected_y),
                now,
            ),
        )
        written += 1
    return written


def _ingest_roads(
    connection: sqlite3.Connection,
    archive: zipfile.ZipFile,
    members: dict[str, str],
    settlement: str,
    transformer: Transformer,
    now: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer in ROAD_LAYERS:
        required = {f"{layer}.SHP", f"{layer}.DBF"}
        if not all(name in members for name in required):
            counts[layer] = 0
            continue
        attributes = _read_dbf_records(archive.read(members[f"{layer}.DBF"]))
        records = list(_polylinez_parts(archive.read(members[f"{layer}.SHP"])))
        if len(attributes) != len(records):
            raise AssetRebuildError(
                f"{settlement} {layer} DBF/shape count mismatch: "
                f"{len(attributes)} != {len(records)}"
            )
        written = 0
        for index, parts in enumerate(records):
            if not parts:
                continue
            row = attributes[index]
            for part_index, part in enumerate(parts):
                if len(part) < 2:
                    continue
                source_x, source_y, elevations = zip(*part)
                projected_x, projected_y = transformer.transform(source_x, source_y)
                path = [
                    [round(x, 2), round(y, 2), round(z, 2)]
                    for x, y, z in zip(projected_x, projected_y, elevations)
                ]
                road_id = f"{settlement}_{layer}_{row.get('lokal_id', '') or index}"
                if part_index:
                    road_id += f"_p{part_index}"
                center_x = sum(projected_x) / len(projected_x)
                center_y = sum(projected_y) / len(projected_y)
                latitude, longitude = to_wgs84(center_x, center_y)
                connection.execute(
                    "INSERT INTO assets(id,type,enabled,lat,lon,heading_deg,z,"
                    "properties,cx,cy,min_x,min_y,max_x,max_y,updated_at) "
                    "VALUES (?,?,1,?,?,0,NULL,?,?,?,?,?,?,?,?)",
                    (
                        road_id,
                        layer,
                        float(latitude),
                        float(longitude),
                        _json({
                            "sourceLayer": layer,
                            "sourceProperties": row,
                            "path": path,
                        }),
                        center_x,
                        center_y,
                        min(projected_x),
                        min(projected_y),
                        max(projected_x),
                        max(projected_y),
                        now,
                    ),
                )
                written += 1
        counts[layer] = written
    return counts


def build_catalog(
    output_path: Path,
    source_directory: Path = DEFAULT_SOURCE_DIRECTORY,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    ground_samples_path: Path = DEFAULT_GROUND_SAMPLES_PATH,
) -> dict[str, Any]:
    """Build and validate a new catalog at an explicit output path."""
    metadata = _read_metadata(metadata_path)
    available_archives = sorted(source_directory.glob("*_TekniskGrundkort_SHP.zip"))
    if not available_archives:
        raise AssetRebuildError(f"no Grundkort archives found in {source_directory}")
    archives_by_settlement: dict[str, Path] = {}
    for archive in available_archives:
        settlement = _settlement_code(archive)
        if settlement in archives_by_settlement:
            raise AssetRebuildError(
                f"multiple Grundkort archives found for {settlement}"
            )
        archives_by_settlement[settlement] = archive
    selected_settlements = metadata["grundkortSettlements"]
    missing_archives = [
        settlement
        for settlement in selected_settlements
        if settlement not in archives_by_settlement
    ]
    if missing_archives:
        raise AssetRebuildError(
            "required Grundkort archives are missing: " + ", ".join(missing_archives)
        )
    archives = [archives_by_settlement[value] for value in selected_settlements]
    ground_samples = _read_ground_samples(ground_samples_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise AssetRebuildError(f"rebuild output already exists: {output_path}")

    connection = sqlite3.connect(output_path, timeout=30.0)
    try:
        schema.create(connection)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _store_metadata(connection, metadata)
        counts = _seed_assets(connection, metadata, now)
        counts.setdefault(BUILDING_LAYER, 0)
        for layer in ROAD_LAYERS:
            counts.setdefault(layer, 0)
        for archive_path in archives:
            settlement = _settlement_code(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                members = _archive_members(archive)
                prj = _required_member(
                    archive, members, f"{BUILDING_LAYER}.PRJ"
                ).decode("latin1")
                transformer = Transformer.from_crs(
                    _source_epsg(prj), 3413, always_xy=True
                )
                counts[BUILDING_LAYER] += _ingest_buildings(
                    connection,
                    archive,
                    members,
                    settlement,
                    transformer,
                    ground_samples,
                    now,
                )
                for layer, count in _ingest_roads(
                    connection, archive, members, settlement, transformer, now
                ).items():
                    counts[layer] += count
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise AssetRebuildError(f"rebuilt catalog failed integrity check: {integrity}")
        metadata_count = connection.execute(
            "SELECT COUNT(*) FROM asset_metadata"
        ).fetchone()[0]
        if metadata_count != 5:
            raise AssetRebuildError(
                f"rebuilt catalog has {metadata_count} metadata rows instead of 5"
            )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
        return {
            "archives": len(archives),
            "asset_count": sum(counts.values()),
            "metadata_count": metadata_count,
            "type_counts": [
                {"type": asset_type, "count": count}
                for asset_type, count in sorted(counts.items())
            ],
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def temporary_output_path(destination: Path) -> Path:
    """Return a collision-resistant sibling path for an atomic rebuild."""
    return destination.with_name(f"{destination.name}.rebuild-{uuid.uuid4().hex}")
