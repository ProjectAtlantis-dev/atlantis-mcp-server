"""Deterministic gate for pure supplied-camera terrain LOD."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import write_dem
from dynamic_functions.Terrain.binary_batch import encode_composed_tiles_binary
from dynamic_functions.Terrain.camera_lod import (
    altitude_depth_cap,
    compose_camera_from_ready_data,
    lod_target_depth,
    resolve_lod_coverage,
    select_lod_tiles,
)
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import ancestor_tile_ids, tile_bounds


_PARENT = "8-220-120"
_PARENT_EXACT_CHILD = "10-881-480"
_CAMERA_TILE = "10-900-500"
_MISSING = "10-920-520"
_FIXTURES = (_PARENT, _PARENT_EXACT_CHILD, _CAMERA_TILE, _MISSING)
_EXPECTED_SELECTION_DIGEST = (
    "1cceaa661e08d05779d399329c7e5f3afac36973b36bbd678ef4311b9a49e934"
)
_EXPECTED_BINARY_DIGEST = (
    "67785f528b6fcc6fd038999c66e9b33c27bc6837552c4c86c0171558ad7d7194"
)


def _cleanup(connection) -> None:
    tile_ids: set[str] = set(_FIXTURES)
    for tile_id in _FIXTURES:
        tile_ids.update(ancestor_tile_ids(tile_id, include_self=True))
    marks = ",".join("?" for _ in tile_ids)
    args = tuple(tile_ids)
    for table in (
        "tidal_connectivity_masks",
        "coastline_masks",
        "hydrography_masks",
        "textures",
        "tiles",
    ):
        connection.execute(
            f"DELETE FROM {table} WHERE tile_id IN ({marks})", args
        )


def _selection_tile(tile_id: str) -> dict:
    depth = int(tile_id.split("-", 1)[0])
    return {
        "tileId": tile_id,
        "depth": depth,
        "bbox": [float(value) for value in tile_bounds(tile_id, GREENLAND_BBOX)],
        "distance": 0.0,
    }


@visible
def camera_lod_offline() -> dict:
    """Prove Flask parity, coherent fallback, wire shape, and zero demand."""

    camera_bbox = tile_bounds(_CAMERA_TILE, GREENLAND_BBOX)
    camera_x = (camera_bbox[0] + camera_bbox[2]) / 2.0
    camera_y = (camera_bbox[1] + camera_bbox[3]) / 2.0
    selection = select_lod_tiles(camera_x, camera_y, 16000.0, 12)
    selection_digest = hashlib.sha256(
        "\n".join(selection["tileIds"]).encode("ascii")
    ).hexdigest()

    radial_cases = {
        0: 12,
        3000: 12,
        3001: 11,
        6000: 11,
        6001: 10,
        9000: 10,
        9001: 9,
        12000: 9,
        12001: 8,
        16000: 8,
    }
    deep_cases = {
        (13, 988): 13,
        (13, 989): 12,
        (16, 123): 16,
        (16, 124): 15,
        (16, 247): 15,
        (16, 248): 14,
        (16, 494): 14,
        (16, 495): 13,
    }
    altitude_cases = {0: 13, 600: 13, 700: 12, 1400: 11, 1e6: 8}

    connection = db()
    connection.execute("SAVEPOINT camera_lod_test")
    try:
        _cleanup(connection)
        base = np.full((65, 65), 123.0, dtype=np.float32)
        for index, tile_id in enumerate(
            (_PARENT, _PARENT_EXACT_CHILD, _CAMERA_TILE)
        ):
            write_dem(
                connection,
                tile_id,
                base + np.float32(index),
                "arcticdem_10m",
                "EGM2008",
                commit=False,
            )
        land = np.zeros((65, 65), dtype=bool)
        for tile_id in (_PARENT, _CAMERA_TILE):
            write_coastline_mask(
                connection,
                tile_id,
                land,
                "fixture_coastline",
                1,
                commit=False,
            )

        manual_selection = {
            "tiles": [
                _selection_tile("10-880-480"),
                _selection_tile(_PARENT_EXACT_CHILD),
                _selection_tile(_CAMERA_TILE),
                _selection_tile(_MISSING),
            ]
        }
        before = connection.total_changes
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("camera LOD attempted network access"),
        ):
            coverage = resolve_lod_coverage(connection, manual_selection)
            composed = compose_camera_from_ready_data(
                connection,
                camera_x,
                camera_y,
                100.0,
                10,
            )
            wide_composed = compose_camera_from_ready_data(
                connection,
                camera_x,
                camera_y,
                16000.0,
                10,
            )
            body, header = encode_composed_tiles_binary(composed)
        read_only = connection.total_changes == before
        wire_tile = header["tiles"][0]
        missing_by_id = {
            item["tileId"]: item for item in coverage["missing"]
        }

        invalid_rejected = False
        try:
            select_lod_tiles(camera_x, camera_y, 0.0, 10)
        except ValueError:
            invalid_rejected = True

        return {
            "radialBoundaryParity": all(
                lod_target_depth(distance, 16000.0, 12) == expected
                for distance, expected in radial_cases.items()
            ),
            "pastContractParity": all(
                lod_target_depth(distance, 16000.0, max_depth) == expected
                for (max_depth, distance), expected in deep_cases.items()
            ),
            "altitudeParity": all(
                altitude_depth_cap(altitude, 13) == expected
                for altitude, expected in altitude_cases.items()
            ),
            "hysteresisHeld": altitude_depth_cap(700.0, 13, 13) == 13,
            "stableSelection": bool(
                selection["tileCount"] == 259
                and selection_digest == _EXPECTED_SELECTION_DIGEST
            ),
            "twoToOneBalanced": selection["twoToOneBalanced"],
            "pureSelection": bool(
                selection["pure"]
                and selection["databaseAccess"] is False
                and selection["networkAccess"] is False
                and selection["scheduledWork"] is False
            ),
            "coherentFallbackAntichain": coverage["coverageTileIds"]
            == [_PARENT, _CAMERA_TILE],
            "fallbackReported": bool(
                missing_by_id["10-880-480"]["state"] == "fallback"
                and missing_by_id["10-880-480"]["fallbackTileId"] == _PARENT
            ),
            "waterDependencyPreservesParent": bool(
                missing_by_id[_PARENT_EXACT_CHILD]["state"] == "fallback"
                and missing_by_id[_PARENT_EXACT_CHILD]["fallbackTileId"]
                == _PARENT
            ),
            "trueMissReported": bool(
                missing_by_id[_MISSING]["state"] == "missing"
                and missing_by_id[_MISSING]["fallbackTileId"] is None
            ),
            "cameraGeometry": bool(
                composed["targetTileCount"] == 1
                and composed["tileCount"] == 1
                and composed["missing"] == []
                and composed["tiles"][0]["tileId"] == _CAMERA_TILE
                and composed["tiles"][0]["depth"] == 10
                and composed["tiles"][0]["center"] == [0.0, 0.0]
            ),
            "missingViewerFields": bool(
                wide_composed["missing"]
                and all(
                    item["id"] == item["tileId"]
                    and len(item["bbox"]) == 4
                    and len(item["stereoBbox"]) == 4
                    and item["bbox"][0]
                    == item["stereoBbox"][0] - wide_composed["ox"]
                    and item["bbox"][1]
                    == item["stereoBbox"][1] - wide_composed["oy"]
                    for item in wide_composed["missing"]
                )
            ),
            "browserWireFields": bool(
                wire_tile["id"] == _CAMERA_TILE
                and wire_tile["resolution"] == 65
                and wire_tile["heightmapBytes"] == 65 * 65 * 4
                and isinstance(wire_tile["bbox"], list)
            ),
            "readOnly": read_only,
            "noNetworkOrScheduling": bool(
                composed["networkAccess"] is False
                and composed["scheduledWork"] is False
                and coverage["networkAccess"] is False
                and coverage["scheduledWork"] is False
            ),
            "invalidInputRejected": invalid_rejected,
            "binaryDigest": hashlib.sha256(body).hexdigest(),
            "stableBinary": hashlib.sha256(body).hexdigest()
            == _EXPECTED_BINARY_DIGEST,
            "contentLength": len(body),
        }
    finally:
        connection.execute("ROLLBACK TO camera_lod_test")
        connection.execute("RELEASE camera_lod_test")
