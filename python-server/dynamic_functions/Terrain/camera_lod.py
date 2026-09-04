"""Pure supplied-camera LOD selection and read-only ready-data coverage."""

from __future__ import annotations

import base64
import hashlib
import math
import sqlite3
from collections import Counter

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.binary_batch import encode_composed_tiles_binary
from dynamic_functions.Terrain.composition import (
    MAX_COMPOSE_TILES,
    compose_tiles_from_ready_data,
)
from dynamic_functions.Terrain.terrain_config import (
    GREENLAND_BBOX,
    MAX_TILE_DEPTH,
    WMS_CONTRACT_DEPTH,
)
from dynamic_functions.Terrain.tile_address import (
    ancestor_tile_ids,
    format_tile_id,
    require_tile_id,
    tile_bounds,
)


LOD_COARSE_FLOOR_DEPTH = 8
LOD_FINE_PLATEAU_RATIO = 0.55
LOD_FINE_PLATEAU_MAX_M = 3000.0
LOD_TRANSITION_MAX_M = 12000.0
LOD_PAST_CONTRACT_CORE_TILE_WIDTHS = 3.0
LOD_ALTITUDE_WIDTH_FACTOR = 2.0
LOD_ALTITUDE_HYSTERESIS = 0.25
MAX_LOD_TILES = 2500
_SQL_CHUNK = 500


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_depth(value: object, name: str = "max_depth") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= MAX_TILE_DEPTH:
        raise ValueError(f"{name} must be between 0 and {MAX_TILE_DEPTH}")
    return value


def _tile_width_m(depth: int) -> float:
    return (GREENLAND_BBOX[2] - GREENLAND_BBOX[0]) / (1 << depth)


def altitude_depth_cap(
    altitude: float,
    max_depth: int,
    previous_depth: int | None = None,
) -> int:
    """Return Flask's altitude ceiling with optional client hysteresis."""

    depth = max_depth
    while (
        depth > LOD_COARSE_FLOOR_DEPTH
        and altitude > LOD_ALTITUDE_WIDTH_FACTOR * _tile_width_m(depth)
    ):
        depth -= 1
    if previous_depth is None:
        return depth

    previous = max(
        LOD_COARSE_FLOOR_DEPTH,
        min(int(max_depth), int(previous_depth)),
    )
    if depth < previous:
        boundary = LOD_ALTITUDE_WIDTH_FACTOR * _tile_width_m(previous)
        if altitude <= boundary * (1.0 + LOD_ALTITUDE_HYSTERESIS):
            return previous
    elif depth > previous:
        boundary = LOD_ALTITUDE_WIDTH_FACTOR * _tile_width_m(
            min(int(max_depth), previous + 1)
        )
        if altitude >= boundary * (1.0 - LOD_ALTITUDE_HYSTERESIS):
            return previous
    return depth


def lod_target_depth(
    distance: float,
    max_range: float,
    max_depth: int,
    altitude: float = 0.0,
) -> int:
    """Return the Flask-compatible radial LOD ceiling at one distance."""

    max_depth = max(0, int(max_depth))
    if altitude > 0.0:
        max_depth = min(max_depth, altitude_depth_cap(altitude, max_depth))
    if max_range <= 0:
        return max_depth
    contract_ceiling = min(max_depth, WMS_CONTRACT_DEPTH)
    coarse_depth = min(contract_ceiling, LOD_COARSE_FLOOR_DEPTH)
    distance = max(0.0, distance)
    fine_plateau_end = min(
        max_range * LOD_FINE_PLATEAU_RATIO,
        LOD_FINE_PLATEAU_MAX_M,
    )

    if contract_ceiling == coarse_depth:
        depth = contract_ceiling
    elif distance <= fine_plateau_end:
        depth = contract_ceiling
    else:
        coarse_rim_start = min(
            max_range,
            fine_plateau_end + LOD_TRANSITION_MAX_M,
        )
        if distance >= coarse_rim_start:
            depth = coarse_depth
        else:
            transition = (
                (distance - fine_plateau_end)
                / (coarse_rim_start - fine_plateau_end)
            )
            continuous = contract_ceiling - (
                contract_ceiling - coarse_depth
            ) * transition
            depth = max(
                coarse_depth,
                min(contract_ceiling, math.floor(continuous)),
            )

    for deeper in range(WMS_CONTRACT_DEPTH + 1, max_depth + 1):
        core = min(
            fine_plateau_end,
            LOD_PAST_CONTRACT_CORE_TILE_WIDTHS * _tile_width_m(deeper),
        )
        if distance <= core:
            depth = deeper
    return depth


def _distance_to_bbox(
    x: float,
    y: float,
    bbox: tuple[float, float, float, float],
) -> float:
    dx = max(bbox[0] - x, 0.0, x - bbox[2])
    dy = max(bbox[1] - y, 0.0, y - bbox[3])
    return math.hypot(dx, dy)


def _coarse_lod_neighbors(tile_ids: list[str]) -> set[tuple[int, int, int]]:
    """Return selected leaves bordering another leaf over one level finer."""

    addresses = {require_tile_id(tile_id) for tile_id in tile_ids}
    coarse: set[tuple[int, int, int]] = set()
    for fine_depth, fine_column, fine_row in addresses:
        if fine_depth < 2:
            continue
        limit = 1 << fine_depth
        for neighbor_column, neighbor_row in (
            (fine_column - 1, fine_row),
            (fine_column + 1, fine_row),
            (fine_column, fine_row - 1),
            (fine_column, fine_row + 1),
        ):
            if not (
                0 <= neighbor_column < limit and 0 <= neighbor_row < limit
            ):
                continue
            for coarse_depth in range(fine_depth - 2, -1, -1):
                scale = 1 << (fine_depth - coarse_depth)
                candidate = (
                    coarse_depth,
                    neighbor_column // scale,
                    neighbor_row // scale,
                )
                if candidate in addresses:
                    coarse.add(candidate)
                    break
    return coarse


def _balance_lod_tiles(
    tiles: list[dict],
    camera_x: float,
    camera_y: float,
    max_range: float,
    max_depth: int,
) -> tuple[list[dict], int]:
    """Purely refine the coarse side of every depth gap beyond 2:1."""

    by_id = {tile["tileId"]: tile for tile in tiles}
    refined = 0
    for _ in range(max_depth + 1):
        coarse = _coarse_lod_neighbors(list(by_id))
        if not coarse:
            break
        for depth, column, row in sorted(coarse):
            tile_id = format_tile_id(depth, column, row)
            if tile_id not in by_id or depth >= max_depth:
                continue
            del by_id[tile_id]
            child_depth = depth + 1
            column2, row2 = column * 2, row * 2
            for child_column, child_row in (
                (column2, row2),
                (column2 + 1, row2),
                (column2, row2 + 1),
                (column2 + 1, row2 + 1),
            ):
                child_id = format_tile_id(
                    child_depth, child_column, child_row
                )
                bbox = tile_bounds(child_id, GREENLAND_BBOX)
                distance = _distance_to_bbox(camera_x, camera_y, bbox)
                if distance <= max_range:
                    by_id[child_id] = {
                        "tileId": child_id,
                        "depth": child_depth,
                        "bbox": [float(value) for value in bbox],
                        "distance": distance,
                    }
            refined += 1
            if len(by_id) > MAX_LOD_TILES:
                raise RuntimeError(
                    f"camera LOD exceeded the {MAX_LOD_TILES}-tile budget"
                )
    result = list(by_id.values())
    result.sort(
        key=lambda tile: (
            -tile["depth"],
            require_tile_id(tile["tileId"])[1],
            require_tile_id(tile["tileId"])[2],
        )
    )
    return result, refined


def _walk_lod(
    camera_x: float,
    camera_y: float,
    max_range: float,
    settled_depth: int,
    altitude: float,
) -> list[dict]:
    leaves: list[dict] = []

    def visit(depth: int, column: int, row: int) -> None:
        tile_id = format_tile_id(depth, column, row)
        bbox = tile_bounds(tile_id, GREENLAND_BBOX)
        distance = _distance_to_bbox(camera_x, camera_y, bbox)
        if distance > max_range:
            return
        target_depth = lod_target_depth(
            distance,
            max_range,
            settled_depth,
            altitude,
        )
        if depth >= target_depth:
            leaves.append(
                {
                    "tileId": tile_id,
                    "depth": depth,
                    "bbox": [float(value) for value in bbox],
                    "distance": distance,
                }
            )
            if len(leaves) > MAX_LOD_TILES:
                raise RuntimeError(
                    f"camera LOD exceeded the {MAX_LOD_TILES}-tile budget"
                )
            return
        child_depth = depth + 1
        column2, row2 = column * 2, row * 2
        for child_column, child_row in (
            (column2, row2),
            (column2 + 1, row2),
            (column2, row2 + 1),
            (column2 + 1, row2 + 1),
        ):
            visit(child_depth, child_column, child_row)

    visit(0, 0, 0)
    leaves.sort(
        key=lambda tile: (
            -tile["depth"],
            require_tile_id(tile["tileId"])[1],
            require_tile_id(tile["tileId"])[2],
        )
    )
    return leaves


def select_lod_tiles(
    camera_x: float,
    camera_y: float,
    max_range: float = 16000.0,
    max_depth: int = WMS_CONTRACT_DEPTH,
    altitude: float = 0.0,
    previous_depth: int | None = None,
) -> dict:
    """Select a bounded quadtree leaf set using only supplied camera values."""

    x = _finite_number(camera_x, "camera_x")
    y = _finite_number(camera_y, "camera_y")
    radius = _finite_number(max_range, "max_range")
    if radius <= 0.0:
        raise ValueError("max_range must be greater than zero")
    requested_depth = _validated_depth(max_depth)
    camera_altitude = _finite_number(altitude, "altitude")
    if camera_altitude < 0.0:
        raise ValueError("altitude must be non-negative")
    previous = None
    if previous_depth is not None:
        previous = _validated_depth(previous_depth, "previous_depth")
    settled_depth = altitude_depth_cap(
        camera_altitude,
        requested_depth,
        previous,
    )
    tiles = _walk_lod(x, y, radius, settled_depth, camera_altitude)
    tiles, balanced = _balance_lod_tiles(
        tiles, x, y, radius, settled_depth
    )
    depth_counts = Counter(tile["depth"] for tile in tiles)
    return {
        "camera": {
            "x": x,
            "y": y,
            "altitude": camera_altitude,
            "maxRange": radius,
        },
        "requestedDepth": requested_depth,
        "depthCap": settled_depth,
        "tiles": tiles,
        "tileIds": [tile["tileId"] for tile in tiles],
        "tileCount": len(tiles),
        "depthCounts": {
            str(depth): count for depth, count in sorted(depth_counts.items())
        },
        "balancedCoarseTiles": balanced,
        "twoToOneBalanced": not _coarse_lod_neighbors(
            [tile["tileId"] for tile in tiles]
        ),
        "pure": True,
        "databaseAccess": False,
        "networkAccess": False,
        "scheduledWork": False,
    }


def _dem_readiness(
    connection: sqlite3.Connection,
    target_ids: list[str],
) -> dict:
    """Return render-ready DEM IDs and explicit water-dependency blocks.

    A newly stored DEM is source-ready but not render-ready until its water
    classification exists. Keeping that distinction here prevents an exact
    child from displacing a coherent ancestor during the interval between the
    independent DEM and coastline workers.
    """

    candidates: set[str] = set()
    for tile_id in target_ids:
        candidates.update(ancestor_tile_ids(tile_id, include_self=True))
    ordered = sorted(candidates)
    dem_ready: set[str] = set()
    for start in range(0, len(ordered), _SQL_CHUNK):
        chunk = ordered[start : start + _SQL_CHUNK]
        marks = ",".join("?" for _ in chunk)
        rows = connection.execute(
            "SELECT tile_id FROM tiles "
            f"WHERE tile_id IN ({marks}) "
            "AND heightmap IS NOT NULL AND confidence_map IS NOT NULL",
            chunk,
        ).fetchall()
        dem_ready.update(row[0] for row in rows)

    def coastline_id(tile_id: str) -> str:
        depth, column, row = require_tile_id(tile_id)
        if depth <= WMS_CONTRACT_DEPTH:
            return tile_id
        shift = depth - WMS_CONTRACT_DEPTH
        return format_tile_id(
            WMS_CONTRACT_DEPTH,
            column >> shift,
            row >> shift,
        )

    coastline_ids = sorted({coastline_id(tile_id) for tile_id in dem_ready})
    coastline_ready: set[str] = set()
    for start in range(0, len(coastline_ids), _SQL_CHUNK):
        chunk = coastline_ids[start : start + _SQL_CHUNK]
        if not chunk:
            continue
        marks = ",".join("?" for _ in chunk)
        rows = connection.execute(
            "SELECT tile_id FROM coastline_masks "
            f"WHERE tile_id IN ({marks})",
            chunk,
        ).fetchall()
        coastline_ready.update(row[0] for row in rows)
    ready = {
        tile_id
        for tile_id in dem_ready
        if coastline_id(tile_id) in coastline_ready
    }
    blocked = [
        {
            "tileId": tile_id,
            "coastlineTileId": coastline_id(tile_id),
            "requested": tile_id in target_ids,
        }
        for tile_id in sorted(dem_ready - ready, key=require_tile_id)
    ]
    return {"ready": ready, "waterDependencyBlocked": blocked}


def _ready_dem_ids(
    connection: sqlite3.Connection,
    target_ids: list[str],
) -> set[str]:
    """Return DEM tiles whose contract-depth coastline is also publishable."""

    return _dem_readiness(connection, target_ids)["ready"]


def _is_descendant(tile_id: str, ancestor_id: str) -> bool:
    depth, column, row = require_tile_id(tile_id)
    ancestor_depth, ancestor_column, ancestor_row = require_tile_id(ancestor_id)
    if depth < ancestor_depth:
        return False
    shift = depth - ancestor_depth
    return column >> shift == ancestor_column and row >> shift == ancestor_row


def resolve_lod_coverage(
    connection: sqlite3.Connection,
    selection: dict,
) -> dict:
    """Resolve current camera leaves to exact tiles or ready ancestors.

    Each desired tile independently selects its nearest render-ready ancestor.
    Distinct selections deliberately remain hierarchical: a coarse fallback
    may coexist with exact descendants needed by neighboring targets.  The
    viewer clips the coarse mesh around those descendants, so one unresolved
    target cannot downgrade an otherwise ready part of the camera footprint.

    Stored descendants outside the current desired leaf set are deliberately
    ignored.  They remain durable database cache entries, but camera movement
    to a coarser LOD must also reduce the rendered geometry depth.  Monotonic
    quality is scoped to one current camera selection: as its requested leaves
    become ready, coverage may advance from ancestor fallback to exact tiles
    but must not fall back again.
    """

    target_tiles = selection.get("tiles")
    if not isinstance(target_tiles, list):
        raise TypeError("selection must contain a tiles list")
    target_ids = [tile["tileId"] for tile in target_tiles]
    readiness = _dem_readiness(connection, target_ids)
    ready = readiness["ready"]
    resolved_by_target: dict[str, str | None] = {}
    exact_ids: set[str] = set()
    for target_id in target_ids:
        resolved = next(
            (
                candidate
                for candidate in ancestor_tile_ids(target_id, include_self=True)
                if candidate in ready
            ),
            None,
        )
        resolved_by_target[target_id] = resolved
        if resolved == target_id:
            exact_ids.add(target_id)

    coverage_targets: dict[str, list[str]] = {}
    for target_id, resolved in resolved_by_target.items():
        if resolved is not None:
            coverage_targets.setdefault(resolved, []).append(target_id)
    coverage_ids = sorted(coverage_targets, key=require_tile_id)

    target_by_id = {tile["tileId"]: tile for tile in target_tiles}
    coverage = []
    for tile_id in coverage_ids:
        depth, _column, _row = require_tile_id(tile_id)
        bbox = tile_bounds(tile_id, GREENLAND_BBOX)
        covered_targets = coverage_targets[tile_id]
        coverage.append(
            {
                "tileId": tile_id,
                "depth": depth,
                "bbox": [float(value) for value in bbox],
                "targetIds": covered_targets,
                "fallback": any(
                    target_id != tile_id and _is_descendant(target_id, tile_id)
                    for target_id in covered_targets
                ),
            }
        )

    missing = []
    for target_id in target_ids:
        if target_id in exact_ids:
            continue
        target = target_by_id[target_id]
        covered_by = resolved_by_target[target_id]
        if covered_by not in coverage_targets:
            covered_by = None
        missing.append(
            {
                "tileId": target_id,
                "bbox": list(target["bbox"]),
                "state": "fallback" if covered_by is not None else "missing",
                "fallbackTileId": covered_by,
            }
        )

    return {
        "coverage": coverage,
        "coverageTileIds": coverage_ids,
        "coverageTileCount": len(coverage),
        "missing": missing,
        "missingTileIds": [tile["tileId"] for tile in missing],
        "missingTileCount": len(missing),
        "exactTargetCount": len(exact_ids),
        "waterDependencyBlocked": readiness["waterDependencyBlocked"],
        "waterDependencyBlockedCount": len(
            readiness["waterDependencyBlocked"]
        ),
        "readOnly": True,
        "networkAccess": False,
        "scheduledWork": False,
    }


def compose_camera_from_ready_data(
    connection: sqlite3.Connection,
    camera_x: float,
    camera_y: float,
    max_range: float = 16000.0,
    max_depth: int = WMS_CONTRACT_DEPTH,
    altitude: float = 0.0,
    previous_depth: int | None = None,
    origin_x: float | None = None,
    origin_y: float | None = None,
) -> dict:
    """Select current camera LOD and compose its ready render coverage."""

    selection = select_lod_tiles(
        camera_x,
        camera_y,
        max_range,
        max_depth,
        altitude,
        previous_depth,
    )
    coverage = resolve_lod_coverage(connection, selection)
    composed = {
        "tiles": [],
        "tileCount": 0,
        "readOnly": True,
        "networkAccess": False,
        "scheduledWork": False,
    }
    for start in range(0, len(coverage["coverageTileIds"]), MAX_COMPOSE_TILES):
        batch = compose_tiles_from_ready_data(
            connection,
            coverage["coverageTileIds"][start : start + MAX_COMPOSE_TILES],
        )
        composed["tiles"].extend(batch["tiles"])
    composed["tileCount"] = len(composed["tiles"])
    ox = selection["camera"]["x"] if origin_x is None else _finite_number(
        origin_x, "origin_x"
    )
    oy = selection["camera"]["y"] if origin_y is None else _finite_number(
        origin_y, "origin_y"
    )
    coverage_by_id = {
        tile["tileId"]: tile for tile in coverage["coverage"]
    }
    for tile in composed["tiles"]:
        geometry = coverage_by_id[tile["tileId"]]
        stereo_bbox = geometry["bbox"]
        relative_bbox = [
            stereo_bbox[0] - ox,
            stereo_bbox[1] - oy,
            stereo_bbox[2] - ox,
            stereo_bbox[3] - oy,
        ]
        dem = tile.get("dem", {})
        texture = tile.get("texture", {})
        tile.update(
            {
                "bbox": relative_bbox,
                "stereoBbox": stereo_bbox,
                "depth": geometry["depth"],
                "center": [
                    (relative_bbox[0] + relative_bbox[2]) / 2.0,
                    (relative_bbox[1] + relative_bbox[3]) / 2.0,
                ],
                "size": relative_bbox[2] - relative_bbox[0],
                "targetIds": geometry["targetIds"],
                "source": dem.get("source"),
                "hasTexture": texture.get("state") == "ready",
                "texAvailable": texture.get("state") == "ready",
                "texStatus": (
                    "ready"
                    if texture.get("state") == "ready" and texture.get("exact")
                    else "ancestor_fallback"
                    if texture.get("state") == "ready"
                    else texture.get("state", "missing")
                ),
                "texSource": texture.get("source"),
                "texAncestorId": texture.get("resolvedTileId"),
                "texIsFetching": False,
            }
        )

    missing = []
    for item in coverage["missing"]:
        stereo_bbox = item["bbox"]
        missing.append(
            {
                **item,
                "id": item["tileId"],
                "bbox": [
                    stereo_bbox[0] - ox,
                    stereo_bbox[1] - oy,
                    stereo_bbox[2] - ox,
                    stereo_bbox[3] - oy,
                ],
                "stereoBbox": stereo_bbox,
            }
        )

    composed.update(
        {
            "qx": selection["camera"]["x"],
            "qy": selection["camera"]["y"],
            "ox": ox,
            "oy": oy,
            "depthCap": max(
                (tile["depth"] for tile in composed["tiles"]),
                default=None,
            ),
            "requestedDepthCap": selection["depthCap"],
            "targetTileCount": selection["tileCount"],
            "missing": missing,
            "downloading": [],
            "tilesReused": 0,
            "waterDependencyBlocked": coverage["waterDependencyBlocked"],
            "waterDependencyBlockedCount": coverage[
                "waterDependencyBlockedCount"
            ],
        }
    )
    return composed


@visible
def camera_lod(
    camera_x: float,
    camera_y: float,
    max_range: float = 16000.0,
    max_depth: int = WMS_CONTRACT_DEPTH,
    altitude: float = 0.0,
    previous_depth: int | None = None,
) -> dict:
    """Return pure desired LOD plus ready DEM ancestor coverage metadata."""

    selection = select_lod_tiles(
        camera_x,
        camera_y,
        max_range,
        max_depth,
        altitude,
        previous_depth,
    )
    return {**selection, **resolve_lod_coverage(db(), selection)}


@visible
def compose_camera_binary(
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
    """Return ready camera coverage as a base64-wrapped binary-v1 envelope."""

    composition = compose_camera_from_ready_data(
        db(),
        camera_x,
        camera_y,
        max_range,
        max_depth,
        altitude,
        previous_depth,
        origin_x,
        origin_y,
    )
    body, header = encode_composed_tiles_binary(composition, known_digests)
    return {
        "format": "binary-v1",
        "mediaType": "application/octet-stream",
        "contentLength": len(body),
        "digest": hashlib.sha256(body).hexdigest(),
        "tileCount": header["tileCount"],
        "targetTileCount": header["targetTileCount"],
        "missingTileCount": len(header["missing"]),
        "tilesReused": header["tilesReused"],
        "contentBase64": base64.b64encode(body).decode("ascii"),
    }
