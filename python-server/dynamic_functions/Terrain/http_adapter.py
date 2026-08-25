"""HTTP compatibility adapter for the existing Atlantis terrain viewer."""

from __future__ import annotations

import hashlib
import io
import math
import sqlite3
from collections.abc import Callable, Mapping
from typing import Any

from PIL import Image
from starlette.responses import Response

from dynamic_functions.Terrain.coords import to_stereo
from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.textures import read_texture_with_ancestor
from dynamic_functions.Terrain.demand import (
    compose_camera_demand_binary_from_ready_data,
    submit_texture_demand,
)
from dynamic_functions.Terrain.terrain_config import MAX_TILE_DEPTH
from dynamic_functions.Terrain.tile_address import (
    ancestor_tile_ids,
    require_tile_id,
)


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_number(value: Any, name: str) -> float | None:
    if value is None or value == "":
        return None
    return _number(value, name)


def parse_tiles_request(query: Mapping[str, str], body: object) -> dict:
    """Validate the legacy viewer request and return camera composition args."""

    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ValueError("request JSON must be an object")
    if query.get("format", "binary") != "binary":
        raise ValueError("only format=binary is supported")

    has_stereo = query.get("sx") is not None or query.get("sy") is not None
    if has_stereo:
        if query.get("sx") is None or query.get("sy") is None:
            raise ValueError("sx and sy must be supplied together")
        camera_x = _number(query.get("sx"), "sx")
        camera_y = _number(query.get("sy"), "sy")
    else:
        if query.get("lat") is None or query.get("lon") is None:
            raise ValueError("supply either sx/sy or lat/lon")
        latitude = _number(query.get("lat"), "lat")
        longitude = _number(query.get("lon"), "lon")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("lat must be between -90 and 90")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("lon must be between -180 and 180")
        camera_x, camera_y = map(float, to_stereo(latitude, longitude))

    altitude_value = query.get("agl", query.get("alt", "0"))
    altitude = _number(altitude_value, "agl")
    if altitude < 0.0:
        raise ValueError("agl must be non-negative")
    max_range = _number(query.get("range", "16000"), "range")
    if max_range <= 0.0:
        raise ValueError("range must be greater than zero")

    origin_x = _optional_number(query.get("ox"), "ox")
    origin_y = _optional_number(query.get("oy"), "oy")
    if (origin_x is None) != (origin_y is None):
        raise ValueError("ox and oy must be supplied together")

    previous_depth = body.get("depthCap")
    if previous_depth is not None:
        if isinstance(previous_depth, bool):
            raise ValueError("depthCap must be an integer")
        try:
            parsed_depth = int(previous_depth)
        except (TypeError, ValueError) as exc:
            raise ValueError("depthCap must be an integer") from exc
        if parsed_depth != previous_depth or not 0 <= parsed_depth <= MAX_TILE_DEPTH:
            raise ValueError(
                f"depthCap must be between 0 and {MAX_TILE_DEPTH}"
            )
        previous_depth = parsed_depth

    known = body.get("known", {})
    if known is None:
        known = {}
    if not isinstance(known, dict):
        raise ValueError("known must be an object")

    return {
        "camera_x": camera_x,
        "camera_y": camera_y,
        "max_range": max_range,
        "max_depth": MAX_TILE_DEPTH,
        "altitude": altitude,
        "previous_depth": previous_depth,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "known_digests": known,
    }


def compose_tiles_response(arguments: dict) -> Response:
    """Run the camera pipeline and return its raw browser wire payload."""

    payload, _ = compose_camera_demand_binary_from_ready_data(
        db(), **arguments
    )
    return binary_response(payload)


def binary_response(payload: bytes) -> Response:
    """Wrap one binary-v1 payload with the legacy endpoint headers."""

    return Response(
        payload,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Terrain-Format": "binary-v1",
        },
    )


def _crop_ancestor_texture(payload: bytes, requested: str, resolved: str) -> bytes:
    child_depth, child_column, child_row = require_tile_id(requested)
    parent_depth, _, _ = require_tile_id(resolved)
    depth_delta = child_depth - parent_depth
    if depth_delta <= 0:
        raise ValueError("texture fallback must resolve to an ancestor")
    divisions = 1 << depth_delta
    sub_column = child_column % divisions
    sub_row = child_row % divisions
    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
        width, height = image.size
        x0 = sub_column * width // divisions
        x1 = (sub_column + 1) * width // divisions
        y0 = (divisions - 1 - sub_row) * height // divisions
        y1 = (divisions - sub_row) * height // divisions
        crop = image.crop((x0, y0, x1, y1)).resize(
            (256, 256), Image.Resampling.BILINEAR
        )
        output = io.BytesIO()
        crop.save(output, format="JPEG", quality=85)
    return output.getvalue()


def texture_response(
    connection: sqlite3.Connection,
    tile_id: str,
    *,
    schedule: Callable[[str], object] = submit_texture_demand,
    if_none_match: str | None = None,
) -> Response:
    """Serve exact imagery or a temporary nearest-ancestor crop."""

    depth, _, _ = require_tile_id(tile_id)
    if depth > MAX_TILE_DEPTH:
        raise ValueError(f"texture depth exceeds {MAX_TILE_DEPTH}")
    # This performs canonical and per-depth range validation before scheduling.
    ancestor_tile_ids(tile_id, include_self=True)
    texture = read_texture_with_ancestor(connection, tile_id)
    if texture is None:
        schedule(tile_id)
        return Response(
            b"",
            status_code=202,
            headers={
                "Cache-Control": "no-store",
                "X-Tex-Status": "fetching",
            },
        )

    payload = bytes(texture["texture"])
    if texture["exact"]:
        etag = f'"{hashlib.sha256(payload).hexdigest()}"'
        headers = {
            "Cache-Control": "public, max-age=86400",
            "ETag": etag,
            "X-Tex-Source": texture["source"],
            "X-Tex-Status": "ready",
            "X-Tex-Quality": "full",
            "X-Tex-Temporary": "0",
        }
        if if_none_match == etag:
            return Response(b"", status_code=304, headers=headers)
        return Response(payload, media_type="image/jpeg", headers=headers)

    schedule(tile_id)
    resolved = texture["resolved_tile_id"]
    crop = _crop_ancestor_texture(payload, tile_id, resolved)
    return Response(
        crop,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Tex-Ancestor": resolved,
            "X-Tex-Source": texture["source"],
            "X-Tex-Status": "ancestor_fallback",
            "X-Tex-Quality": "ancestor_crop",
            "X-Tex-Temporary": "1",
        },
    )


def serve_texture(tile_id: str, if_none_match: str | None = None) -> Response:
    return texture_response(db(), tile_id, if_none_match=if_none_match)
