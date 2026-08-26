"""Offline HTTP compatibility gate for the existing terrain viewer."""

from __future__ import annotations

import io
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from PIL import Image

from dynamic_functions.Terrain.Database import schema
from dynamic_functions.Terrain.demand import _browser_pipeline_fields
from dynamic_functions.Terrain.http_adapter import (
    binary_response,
    parse_tiles_request,
    serve_texture,
    texture_response,
)
from dynamic_functions.Terrain.terrain_config import (
    MAX_TILE_DEPTH,
    WMS_CONTRACT_DEPTH,
)


def _jpeg_quadrants() -> bytes:
    pixels = np.empty((256, 256, 3), dtype=np.uint8)
    pixels[:128, :128] = (255, 0, 0)
    pixels[:128, 128:] = (0, 255, 0)
    pixels[128:, :128] = (0, 0, 255)
    pixels[128:, 128:] = (255, 255, 0)
    output = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(output, "JPEG", quality=95)
    return output.getvalue()


def _jpeg_grid(grid_size: int = 4) -> tuple[bytes, dict[tuple[int, int], tuple[int, int, int]]]:
    child_size = 64
    pixels = np.empty((child_size * grid_size, child_size * grid_size, 3), dtype=np.uint8)
    colors = {}
    for column in range(grid_size):
        for row in range(grid_size):
            color = (20 + column * 45, 25 + row * 40, 35 + (column * grid_size + row) * 9)
            colors[(column, row)] = color
            left = column * child_size
            top = (grid_size - 1 - row) * child_size
            pixels[top : top + child_size, left : left + child_size] = color
    output = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(output, "JPEG", quality=95)
    return output.getvalue(), colors


class _ConcurrentUseProbe:
    """Fail deterministically if two threads enter one SQLite connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.guard = threading.Lock()
        self.active = 0

    def execute(self, *args, **kwargs):
        with self.guard:
            self.active += 1
            concurrent = self.active > 1
        try:
            if concurrent:
                raise sqlite3.InterfaceError("concurrent shared connection use")
            time.sleep(0.001)
            return self.connection.execute(*args, **kwargs)
        finally:
            with self.guard:
                self.active -= 1


def _lane(*, active=(), pending=(), failures=None) -> dict:
    return {
        "claimedActive": list(active),
        "pending": list(pending),
        "failures": failures or {},
    }


@visible
def http_adapter_offline() -> dict:
    """Prove request mapping, wire headers, texture fallback, and queueing."""

    parsed = parse_tiles_request(
        {
            "sx": "-333722.4",
            "sy": "-2824336.2",
            "agl": "8.5",
            "range": "1200",
            "format": "binary",
            "ox": "-333700",
            "oy": "-2824300",
        },
        {"known": {"10-334-192": "1234abcd"}, "depthCap": 12},
    )
    lat_lon = parse_tiles_request(
        {"lat": "64.175", "lon": "-51.7388"}, {}
    )
    bathymetry_demand = parse_tiles_request(
        {
            "sx": "-333722.4",
            "sy": "-2824336.2",
            "demand": "bathymetry",
        },
        {},
    )
    invalid_rejected = 0
    for query, body in (
        ({"sx": "1"}, {}),
        ({"sx": "nan", "sy": "2"}, {}),
        ({"lat": "91", "lon": "0"}, {}),
        ({"sx": "1", "sy": "2", "ox": "3"}, {}),
        ({"sx": "1", "sy": "2", "format": "json"}, {}),
        ({"sx": "1", "sy": "2"}, {"depthCap": "12"}),
        ({"sx": "1", "sy": "2"}, {"known": []}),
    ):
        try:
            parse_tiles_request(query, body)
        except ValueError:
            invalid_rejected += 1

    binary = binary_response(b"terrain-wire")

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    schema.create(connection)
    parent_jpeg = _jpeg_quadrants()
    exact_jpeg = _jpeg_quadrants()
    connection.executemany(
        "INSERT INTO textures (tile_id, source, texture, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (
            ("3-5-5", "fixture-parent", parent_jpeg, "now"),
            ("4-11-11", "fixture-exact", exact_jpeg, "now"),
        ),
    )
    scheduled = []
    exact = texture_response(
        connection, "4-11-11", schedule=scheduled.append
    )
    not_modified = texture_response(
        connection,
        "4-11-11",
        schedule=scheduled.append,
        if_none_match=exact.headers["etag"],
    )
    ancestor = texture_response(
        connection, "4-10-10", schedule=scheduled.append
    )
    missing = texture_response(
        connection, "4-1-1", schedule=scheduled.append
    )
    initial_scheduled = list(scheduled)
    with Image.open(io.BytesIO(ancestor.body)) as cropped:
        mean = np.asarray(cropped.convert("RGB"), dtype=np.float32).mean(
            axis=(0, 1)
        )

    grid_jpeg, grid_colors = _jpeg_grid()
    connection.execute(
        "INSERT INTO textures (tile_id, source, texture, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("10-345-187", "fixture-grid", grid_jpeg, "now"),
    )
    grid_orientation = True
    for column_offset in range(4):
        for row_offset in range(4):
            child_id = f"12-{1380 + column_offset}-{748 + row_offset}"
            response = texture_response(
                connection, child_id, schedule=scheduled.append
            )
            with Image.open(io.BytesIO(response.body)) as child:
                center = tuple(
                    int(value)
                    for value in np.asarray(child.convert("RGB"))[128, 128]
                )
            expected = grid_colors[(column_offset, row_offset)]
            grid_orientation &= bool(
                response.headers["x-tex-ancestor"] == "10-345-187"
                and all(
                    abs(actual - wanted) <= 5
                    for actual, wanted in zip(center, expected)
                )
            )

    probe = _ConcurrentUseProbe(connection)
    expected_by_id = {
        "4-11-11": exact_jpeg,
        "3-5-5": parent_jpeg,
    }
    concurrent_isolation = True
    try:
        with patch(
            "dynamic_functions.Terrain.http_adapter.db",
            return_value=probe,
        ):
            request_ids = list(expected_by_id) * 16
            with ThreadPoolExecutor(max_workers=8) as executor:
                responses = list(executor.map(serve_texture, request_ids))
        concurrent_isolation = all(
            response.body == expected_by_id[tile_id]
            for tile_id, response in zip(request_ids, responses)
        )
    except sqlite3.InterfaceError:
        concurrent_isolation = False
    connection.close()

    lanes = {
        "dem": _lane(active=("dem-a",)),
        "texture": _lane(pending=("tex-a",)),
        "coastline": _lane(),
        "hydrography": _lane(),
        "connectivity": _lane(active=("12:generation",)),
    }
    compact = _browser_pipeline_fields({"lanes": lanes})

    return {
        "stereoMapped": bool(
            parsed["camera_x"] == -333722.4
            and parsed["camera_y"] == -2824336.2
            and parsed["altitude"] == 8.5
            and parsed["max_range"] == 1200.0
            and parsed["max_depth"] == MAX_TILE_DEPTH
            and parsed["previous_depth"] == 12
        ),
        "latLonMapped": bool(
            abs(lat_lon["camera_x"] + 333722.4) < 1.0
            and abs(lat_lon["camera_y"] + 2824336.2) < 1.0
        ),
        "bathymetryDemandCompatibility": bool(
            bathymetry_demand["max_depth"] == WMS_CONTRACT_DEPTH
            and bathymetry_demand["legacy_json"] is True
            and parsed["legacy_json"] is False
        ),
        "invalidRejected": invalid_rejected == 7,
        "binaryHeaders": bool(
            binary.status_code == 200
            and binary.body == b"terrain-wire"
            and binary.media_type == "application/octet-stream"
            and binary.headers["x-terrain-format"] == "binary-v1"
            and binary.headers["cache-control"] == "no-store"
        ),
        "exactTexture": bool(
            exact.status_code == 200
            and exact.body == exact_jpeg
            and exact.headers["x-tex-status"] == "ready"
            and exact.headers["x-tex-temporary"] == "0"
        ),
        "etag304": not_modified.status_code == 304,
        "ancestorTexture": bool(
            ancestor.status_code == 200
            and ancestor.headers["x-tex-ancestor"] == "3-5-5"
            and ancestor.headers["x-tex-status"] == "ancestor_fallback"
            and ancestor.headers["x-tex-temporary"] == "1"
            and mean[2] > 240.0
            and mean[0] < 15.0
            and mean[1] < 15.0
        ),
        "ancestorGridOrientation": grid_orientation,
        "concurrentTextureIsolation": concurrent_isolation,
        "missingQueues": bool(
            missing.status_code == 202
            and missing.headers["x-tex-status"] == "fetching"
            and initial_scheduled == ["4-10-10", "4-1-1"]
        ),
        "compactViewerStatus": bool(
            compact["downloading"] == ["dem-a"]
            and compact["demActionable"]
            and compact["texFetching"] == 1
            and compact["coastlineQueued"] == 1
            and compact["polling"]["nextAction"] == "poll"
            and "demand" not in compact
        ),
    }
