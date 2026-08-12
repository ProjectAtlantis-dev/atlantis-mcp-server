"""Explicit Government of Greenland WMS hydrography acquisition."""

from __future__ import annotations

import datetime
import hashlib
import io
import sqlite3
import urllib.parse
import urllib.request
import zlib

import atlantis
import numpy as np
from PIL import Image, UnidentifiedImageError

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import ensure_tile_row
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import require_tile_id, tile_bounds


SOURCE = "govmin_gl_aabent_land"
VERSION = 1
GRID_N = 65
OVERSAMPLE = 8
WMS_ENDPOINT = "https://gis.govmin.gl/geoserver/wms"
WMS_LAYER = "Greenland:gl_aabent_land"


class HydrographyClobberError(RuntimeError):
    """Raised when a hydrography write would replace different source data."""


def _request_spec(tile_id: str) -> dict:
    """Describe one rendered hydrography request without network or DB I/O."""

    require_tile_id(tile_id)
    bbox = tile_bounds(tile_id, GREENLAND_BBOX)
    sample_resolution = GRID_N * OVERSAMPLE
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "SRS": "EPSG:3413",
        "BBOX": ",".join(str(float(value)) for value in bbox),
        "WIDTH": str(sample_resolution),
        "HEIGHT": str(sample_resolution),
        "FORMAT": "image/png",
    }
    return {
        "provider": "Government of Greenland",
        "dataset": "gl_aabent_land",
        "artifact": "rendered_hydrography_mask",
        "tileId": tile_id,
        "endpoint": WMS_ENDPOINT,
        "crs": "EPSG:3413",
        "bbox": list(bbox),
        "resolution": GRID_N,
        "oversample": OVERSAMPLE,
        "width": sample_resolution,
        "height": sample_resolution,
        "layer": WMS_LAYER,
        "format": "image/png",
        "source": SOURCE,
        "version": VERSION,
        "params": params,
    }


def _water_pixels(rgb: np.ndarray) -> np.ndarray:
    """Identify the WMS's blue water cartography without accepting white."""

    values = np.asarray(rgb, dtype=np.int16)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("hydrography image must contain RGB pixels")
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    return (
        (blue >= 145)
        & ((blue - red) >= 18)
        & ((green - red) >= 10)
        & ((blue - green) >= 12)
    )


def _decode_mask(payload: bytes, resolution: int = GRID_N) -> np.ndarray:
    """Decode an oversampled WMS PNG into a south-first boolean mask."""

    sample_resolution = int(resolution) * OVERSAMPLE
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.load()
    except (OSError, TypeError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError("corrupt hydrography WMS image") from exc
    if image.size != (sample_resolution, sample_resolution):
        raise ValueError(
            f"expected {sample_resolution}x{sample_resolution} hydrography "
            f"image, got {image.width}x{image.height}"
        )
    high_resolution = _water_pixels(np.asarray(image, dtype=np.uint8))
    fractions = high_resolution.reshape(
        resolution,
        OVERSAMPLE,
        resolution,
        OVERSAMPLE,
    ).mean(axis=(1, 3))
    # WMS image rows are north-first; terrain heightmaps are south-first.
    return np.flipud(fractions >= 0.45)


def _fetch_url(url: str, timeout: int = 30) -> tuple[bytes, dict]:
    """Fetch one WMS response without retries or database access."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "greenland-terrain/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), {
            "httpStatus": response.status,
            "contentType": response.headers.get_content_type(),
        }


def _acquire_mask(tile_id: str, fetcher=_fetch_url) -> tuple[np.ndarray, dict]:
    """Fetch and decode one WMS mask without opening the terrain database."""

    request = _request_spec(tile_id)
    url = f"{request['endpoint']}?{urllib.parse.urlencode(request['params'])}"
    payload, response = fetcher(url)
    mask = _decode_mask(payload, request["resolution"])
    values = mask.astype(np.uint8)
    return mask, {
        **{key: value for key, value in request.items() if key != "params"},
        "status": "success",
        "httpStatus": response.get("httpStatus"),
        "responseContentType": response.get("contentType"),
        "responseContentLength": len(payload),
        "responseDigest": hashlib.sha256(payload).hexdigest(),
        "waterCount": int(values.sum()),
        "landCount": int(values.size - values.sum()),
        "digest": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _encoded_mask(mask: np.ndarray) -> tuple[np.ndarray, bytes]:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("hydrography mask must be a 2D array")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("hydrography mask dimensions must be positive")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("hydrography mask must contain only boolean values")
    canonical = values.astype(np.uint8)
    return canonical, zlib.compress(canonical.tobytes(), level=6)


def write_hydrography_mask(
    connection: sqlite3.Connection,
    tile_id: str,
    mask: np.ndarray,
    source: str,
    version: int,
    *,
    commit: bool = True,
) -> bool:
    """Store one exact WMS hydrography mask without replacing source data."""

    require_tile_id(tile_id)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("hydrography source must be a non-empty string")
    if not isinstance(version, int) or version < 1:
        raise ValueError("hydrography version must be a positive integer")
    canonical, encoded = _encoded_mask(mask)
    ensure_tile_row(connection, tile_id)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor = connection.execute(
        "INSERT OR IGNORE INTO hydrography_masks "
        "(tile_id, width, height, mask, source, version, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            tile_id,
            int(canonical.shape[1]),
            int(canonical.shape[0]),
            encoded,
            source,
            version,
            now,
        ),
    )
    if cursor.rowcount == 1:
        if commit:
            connection.commit()
        return True

    row = connection.execute(
        "SELECT width, height, mask, source, version FROM hydrography_masks "
        "WHERE tile_id = ?",
        (tile_id,),
    ).fetchone()
    incoming = (
        int(canonical.shape[1]),
        int(canonical.shape[0]),
        encoded,
        source,
        version,
    )
    if row == incoming:
        return False
    raise HydrographyClobberError(
        f"Refusing to clobber hydrography mask {tile_id}: "
        f"existing source={row[3] if row else None}, "
        f"version={row[4] if row else None}; incoming source={source}, "
        f"version={version}"
    )


def read_hydrography_mask(
    connection: sqlite3.Connection,
    tile_id: str,
) -> dict | None:
    """Read one exact stored hydrography mask without deriving connectivity."""

    require_tile_id(tile_id)
    row = connection.execute(
        "SELECT width, height, mask, source, version, updated_at "
        "FROM hydrography_masks WHERE tile_id = ?",
        (tile_id,),
    ).fetchone()
    if row is None:
        return None
    width, height = int(row[0]), int(row[1])
    values = np.frombuffer(zlib.decompress(row[2]), dtype=np.uint8)
    if values.size != width * height:
        raise ValueError(
            f"hydrography mask for {tile_id} has {values.size} values; "
            f"expected {width * height}"
        )
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"hydrography mask for {tile_id} is not boolean")
    return {
        "tile_id": tile_id,
        "width": width,
        "height": height,
        "mask": values.reshape((height, width)).astype(bool),
        "source": row[3],
        "version": int(row[4]),
        "updated_at": row[5],
        "digest": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _read_response(connection: sqlite3.Connection, tile_id: str) -> dict:
    payload = read_hydrography_mask(connection, tile_id)
    if payload is None:
        return {"tileId": tile_id, "found": False}
    mask = payload["mask"]
    return {
        "tileId": tile_id,
        "found": True,
        "source": payload["source"],
        "version": payload["version"],
        "updatedAt": payload["updated_at"],
        "shape": [payload["height"], payload["width"]],
        "waterCount": int(mask.sum()),
        "landCount": int(mask.size - mask.sum()),
        "digest": payload["digest"],
    }


@visible
async def list_hydrography() -> list[tuple]:
    """Return storage metadata for every persisted WMS hydrography mask."""

    return await atlantis.client_command(
        "@Database/query 'SELECT tile_id,width,height,source,version,updated_at FROM hydrography_masks'"
    )


@visible
def hydrography_request(tile_id: str) -> dict:
    """Describe one WMS hydrography request without network or DB access."""

    return _request_spec(tile_id)


@visible
def read_hydrography(tile_id: str) -> dict:
    """Return metadata for one stored raw hydrography mask."""

    return _read_response(db(), tile_id)


@visible
def fetch_hydrography(tile_id: str) -> dict:
    """Acquire, decode, and persist one raw WMS hydrography mask."""

    # Acquisition and decoding complete before requesting the shared DB.
    # A provider failure therefore cannot create or alter a source row.
    mask, acquisition = _acquire_mask(tile_id)
    connection = db()
    written = write_hydrography_mask(
        connection,
        tile_id,
        mask,
        SOURCE,
        VERSION,
    )
    return {**acquisition, "written": written, **_read_response(connection, tile_id)}
