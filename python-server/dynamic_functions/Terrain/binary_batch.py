"""Browser-compatible binary-v1 encoding for ready terrain composition."""

from __future__ import annotations

import base64
import copy
import hashlib
import re
import sqlite3
import zlib

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.composition import compose_tiles_from_ready_data
from dynamic_functions.Terrain.serve_flask import TileResult, encode_tiles_binary


_FORMAT = "binary-v1"
_MEDIA_TYPE = "application/octet-stream"
_CRC32 = re.compile(r"^[0-9a-f]{8}$")


def _validated_known_digests(
    known_digests: dict[str, str] | None,
) -> dict[str, str]:
    if known_digests is None:
        return {}
    if not isinstance(known_digests, dict):
        raise TypeError("known_digests must be an object")
    result = {}
    for tile_id, digest in known_digests.items():
        if not isinstance(tile_id, str) or not isinstance(digest, str):
            raise TypeError("known_digests must map tile ID strings to strings")
        normalized = digest.lower()
        if not _CRC32.fullmatch(normalized):
            raise ValueError(
                f"known heightmap digest for {tile_id!r} is not eight hex digits"
            )
        result[tile_id] = normalized
    return result


def _heightmap_block(tile: dict) -> tuple[bytes | None, str | None, int | None]:
    dem = tile.get("dem")
    if not isinstance(dem, dict):
        return None, None, None
    heightmap = dem.get("heightmap")
    if not isinstance(heightmap, dict) or heightmap.get("state") != "ready":
        return None, None, None
    encoded = heightmap.get("contentBase64")
    if not isinstance(encoded, str):
        raise ValueError(f"ready heightmap for {tile['tileId']} has no content")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(
            f"ready heightmap for {tile['tileId']} is not valid base64"
        ) from exc
    shape = heightmap.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in shape)
    ):
        raise ValueError(f"ready heightmap for {tile['tileId']} has invalid shape")
    expected_length = shape[0] * shape[1] * 4
    if len(payload) != expected_length:
        raise ValueError(
            f"ready heightmap for {tile['tileId']} has {len(payload)} bytes; "
            f"expected {expected_length}"
        )
    expected_digest = heightmap.get("digest")
    actual_digest = hashlib.sha256(payload).hexdigest()
    if expected_digest != actual_digest:
        raise ValueError(f"ready heightmap digest mismatch for {tile['tileId']}")
    return payload, f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}", shape[0]


def _without_embedded_content(tile: dict) -> dict:
    result = copy.deepcopy(tile)
    dem = result.get("dem")
    if isinstance(dem, dict):
        heightmap = dem.get("heightmap")
        if isinstance(heightmap, dict):
            heightmap.pop("contentBase64", None)
    texture = result.get("texture")
    if isinstance(texture, dict):
        texture.pop("contentBase64", None)
    return result


def encode_composed_tiles_binary(
    composition: dict,
    known_digests: dict[str, str] | None = None,
) -> tuple[bytes, dict]:
    """Encode a Step 6 composition result using the browser binary-v1 layout."""

    known = _validated_known_digests(known_digests)
    source_tiles = composition.get("tiles")
    if not isinstance(source_tiles, list):
        raise TypeError("composition must contain a tiles list")

    tiles = []
    blobs = []
    reused = 0
    for source_tile in source_tiles:
        if not isinstance(source_tile, dict) or not isinstance(
            source_tile.get("tileId"), str
        ):
            raise TypeError("composition tiles must contain string tileId values")
        tile = _without_embedded_content(source_tile)
        tile_id = tile["tileId"]
        tile["id"] = tile_id
        try:
            blob, digest, resolution = _heightmap_block(source_tile)
        except Exception as exc:
            dem = tile.get("dem")
            if not isinstance(dem, dict):
                dem = {}
                tile["dem"] = dem
            dem.update(
                {
                    "state": "error",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
            )
            blob, digest, resolution = None, None, None

        tile["heightmap"] = digest
        tile["resolution"] = resolution
        if blob is None:
            tile["heightmapBytes"] = None
        elif known.get(tile_id) == digest:
            tile["heightmapBytes"] = 0
            reused += 1
        else:
            tile["heightmapBytes"] = len(blob)
            blobs.append(blob)
        tiles.append(tile)

    header = {
        key: copy.deepcopy(value)
        for key, value in composition.items()
        if key != "tiles"
    }
    header.update(
        {
            "format": _FORMAT,
            "tiles": tiles,
            "tileCount": len(tiles),
            "tilesReused": reused,
        }
    )
    body = encode_tiles_binary(TileResult(header, blobs))
    return body, header


def compose_tiles_binary_from_ready_data(
    connection: sqlite3.Connection,
    tile_ids: list[str],
    known_digests: dict[str, str] | None = None,
) -> tuple[bytes, dict]:
    """Compose ready local data and encode it without provider or scheduler work."""

    composition = compose_tiles_from_ready_data(connection, tile_ids)
    return encode_composed_tiles_binary(composition, known_digests)


@visible
def compose_tiles_binary(
    tile_ids: list[str],
    known_digests: dict[str, str] | None = None,
) -> dict:
    """Return a base64-wrapped browser binary-v1 batch from ready local data."""

    body, header = compose_tiles_binary_from_ready_data(
        db(), tile_ids, known_digests
    )
    return {
        "format": _FORMAT,
        "mediaType": _MEDIA_TYPE,
        "contentLength": len(body),
        "digest": hashlib.sha256(body).hexdigest(),
        "tileCount": header["tileCount"],
        "tilesReused": header["tilesReused"],
        "contentBase64": base64.b64encode(body).decode("ascii"),
    }
