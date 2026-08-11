"""Directly callable acquisition and reads for persisted terrain textures."""

import atlantis

import base64
import hashlib
import os

from dynamic_functions.Terrain.dataforsyningen import (
    _fetch_metatile,
    _split_metatile,
)
from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.textures import (
    read_texture_payload,
    write_texture_metatile,
)


_SOURCE = "dataforsyningen"
_MEDIA_TYPE = "image/jpeg"


@visible
async def list_textures() -> list[tuple]:
    """Return the tile ID, source, and update time for every texture row."""

    return await atlantis.client_command("@Database/query 'SELECT tile_id, source, updated_at FROM textures'");


def _read_texture(connection, tile_id: str, *, include_data: bool) -> dict:
    """Build one JSON-safe response from exact stored texture bytes."""

    payload = read_texture_payload(connection, tile_id)
    if payload is None:
        return {"tileId": tile_id, "found": False}

    texture = payload["texture"]
    result = {
        "tileId": tile_id,
        "found": True,
        "source": payload["source"],
        "updatedAt": payload["updated_at"],
        "mediaType": _MEDIA_TYPE,
        "contentLength": len(texture),
        "digest": hashlib.sha256(texture).hexdigest(),
    }
    if include_data:
        result["contentBase64"] = base64.b64encode(texture).decode("ascii")
    return result


@visible
def read_texture(tile_id: str) -> dict:
    """Return one stored JPEG with provenance, digest, and exact base64 bytes."""

    return _read_texture(db(), tile_id, include_data=True)


@visible
def fetch_texture(tile_id: str) -> dict:
    """Fetch and atomically persist one aligned Dataforsyningen metatile.

    Provider acquisition, reprojection, validation, and splitting all finish
    before the shared database connection is requested. Provider failures
    therefore cannot create or alter texture rows.
    """

    token = os.environ.get("DATAFORSYNINGEN_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "DATAFORSYNINGEN_TOKEN is required for live imagery requests"
        )

    metatile_bytes, provider = _fetch_metatile(tile_id, token)
    if metatile_bytes is None:
        return {
            **provider,
            "written": False,
            "databaseTouched": False,
        }

    # This can still reject corrupt dimensions before database acquisition.
    children = _split_metatile(metatile_bytes, tile_id)

    connection = db()
    written = write_texture_metatile(
        connection,
        children,
        _SOURCE,
    )
    child_summaries = [
        _read_texture(connection, child_id, include_data=False)
        for child_id in sorted(children)
    ]
    requested = _read_texture(connection, tile_id, include_data=False)
    if not requested["found"]:
        raise AssertionError(
            f"requested texture {tile_id} was absent after metatile write"
        )

    return {
        **provider,
        "written": written,
        "databaseTouched": True,
        "childCount": len(child_summaries),
        "children": child_summaries,
        "requested": requested,
    }
