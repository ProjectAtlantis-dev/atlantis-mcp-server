"""Offline HTTP compatibility gate for the existing terrain viewer."""

from __future__ import annotations

import io
import sqlite3

import numpy as np
from PIL import Image

from dynamic_functions.Terrain.Database import schema
from dynamic_functions.Terrain.demand import _browser_pipeline_fields
from dynamic_functions.Terrain.http_adapter import (
    binary_response,
    parse_tiles_request,
    texture_response,
)
from dynamic_functions.Terrain.terrain_config import MAX_TILE_DEPTH


def _jpeg_quadrants() -> bytes:
    pixels = np.empty((256, 256, 3), dtype=np.uint8)
    pixels[:128, :128] = (255, 0, 0)
    pixels[:128, 128:] = (0, 255, 0)
    pixels[128:, :128] = (0, 0, 255)
    pixels[128:, 128:] = (255, 255, 0)
    output = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(output, "JPEG", quality=95)
    return output.getvalue()


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

    connection = sqlite3.connect(":memory:")
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
    with Image.open(io.BytesIO(ancestor.body)) as cropped:
        mean = np.asarray(cropped.convert("RGB"), dtype=np.float32).mean(
            axis=(0, 1)
        )
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
        "missingQueues": bool(
            missing.status_code == 202
            and missing.headers["x-tex-status"] == "fetching"
            and scheduled == ["4-10-10", "4-1-1"]
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
