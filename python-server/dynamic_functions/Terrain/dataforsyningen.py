"""Dataforsyningen WMS request construction without provider I/O."""

import io
import hashlib
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, UnidentifiedImageError
from rasterio.crs import CRS
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import require_tile_id, tile_bounds


_WMS_ENDPOINT = "https://api.dataforsyningen.dk/wms/gl_satellitfoto"
_WMS_LAYERS = "ortofoto_0_2m_regional,ortofoto_1_6m_regional"
_SOURCE_CRS = "EPSG:3413"
_WMS_CRS = "EPSG:3184"
_CHILD_RESOLUTION = 256


def _metatile_spec(tile_id: str) -> dict:
    """Describe the aligned 4-by-4 group containing ``tile_id``."""

    depth, column, row = require_tile_id(tile_id)
    # Resolve once to enforce the quadtree range checks before provider work.
    tile_bounds(tile_id, GREENLAND_BBOX)

    depth_step = min(2, depth)
    grid_size = 1 << depth_step
    parent_depth = depth - depth_step
    parent_column = column // grid_size
    parent_row = row // grid_size
    metatile_id = f"{parent_depth}-{parent_column}-{parent_row}"
    bbox = tile_bounds(metatile_id, GREENLAND_BBOX)
    resolution = _CHILD_RESOLUTION * grid_size
    children = [
        {
            "tileId": (
                f"{depth}-{parent_column * grid_size + column_offset}-"
                f"{parent_row * grid_size + row_offset}"
            ),
            "columnOffset": column_offset,
            "rowOffset": row_offset,
            "crop": [
                column_offset * _CHILD_RESOLUTION,
                (grid_size - 1 - row_offset) * _CHILD_RESOLUTION,
                (column_offset + 1) * _CHILD_RESOLUTION,
                (grid_size - row_offset) * _CHILD_RESOLUTION,
            ],
        }
        for column_offset in range(grid_size)
        for row_offset in range(grid_size)
    ]
    return {
        "metatileId": metatile_id,
        "bbox": bbox,
        "resolution": resolution,
        "gridSize": grid_size,
        "children": children,
    }


def _wms_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Transform EPSG:3413 bounds to padded EPSG:3184 WMS bounds."""

    transformed = transform_bounds(
        CRS.from_epsg(3413),
        CRS.from_epsg(3184),
        *bbox,
        densify_pts=21,
    )
    width_padding = (transformed[2] - transformed[0]) * 0.05
    height_padding = (transformed[3] - transformed[1]) * 0.05
    return (
        transformed[0] - width_padding,
        transformed[1] - height_padding,
        transformed[2] + width_padding,
        transformed[3] + height_padding,
    )


def _request_spec(tile_id: str) -> dict:
    metatile = _metatile_spec(tile_id)
    wms_bbox = _wms_bbox(metatile["bbox"])
    resolution = metatile["resolution"]
    return {
        "provider": "dataforsyningen",
        "dataset": "gl_satellitfoto",
        "tileId": tile_id,
        "metatileId": metatile["metatileId"],
        "endpoint": _WMS_ENDPOINT,
        "tokenEnvironmentVariable": "DATAFORSYNINGEN_TOKEN",
        "sourceCrs": _SOURCE_CRS,
        "sourceBbox": list(metatile["bbox"]),
        "crs": _WMS_CRS,
        "bbox": list(wms_bbox),
        "width": resolution,
        "height": resolution,
        "childResolution": _CHILD_RESOLUTION,
        "gridSize": metatile["gridSize"],
        "layers": _WMS_LAYERS.split(","),
        "format": "image/jpeg",
        "params": {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": _WMS_LAYERS,
            "CRS": _WMS_CRS,
            "BBOX": ",".join(str(value) for value in wms_bbox),
            "WIDTH": resolution,
            "HEIGHT": resolution,
            "FORMAT": "image/jpeg",
            "STYLES": "",
        },
        "children": metatile["children"],
    }


def _decode_metatile(image_bytes: bytes, expected_size: int) -> Image.Image:
    """Decode an RGB metatile or raise an explicit corrupt/dimension error."""

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
    except (OSError, TypeError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError("corrupt Dataforsyningen metatile image") from exc
    if image.size != (expected_size, expected_size):
        raise ValueError(
            f"expected {expected_size}x{expected_size} metatile, got "
            f"{image.width}x{image.height}"
        )
    return image


def _no_coverage_kind(
    image: Image.Image,
    warp_void_limit: float = 50.0,
) -> str | None:
    """Classify provider white-fill or reprojection void images."""

    pixels = np.asarray(image, dtype=np.uint8)
    white_pct = float((pixels.min(axis=2) >= 250).mean() * 100.0)
    if white_pct > 98.0 and float(pixels.std()) < 2.0:
        return "white_fill"
    black_pct = float((pixels.max(axis=2) == 0).mean() * 100.0)
    if black_pct > warp_void_limit:
        return "warp_void"
    return None


def _split_metatile(image_bytes: bytes, tile_id: str) -> dict[str, bytes]:
    """Split one aligned metatile into north/south-correct child JPEGs."""

    request = _request_spec(tile_id)
    image = _decode_metatile(image_bytes, request["width"])
    children = {}
    for child in request["children"]:
        cropped = image.crop(tuple(child["crop"]))
        output = io.BytesIO()
        cropped.save(output, format="JPEG", quality=85)
        children[child["tileId"]] = output.getvalue()
    return children


def _http_get(
    url: str,
    *,
    timeout: int = 30,
    retries: int = 3,
) -> tuple[bytes | None, dict]:
    """Fetch provider bytes while keeping credential-bearing URLs private."""

    retryable = {408, 425, 429, 500, 502, 503, 504}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "greenland-terrain/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), {
                    "httpStatus": response.status,
                    "contentType": response.headers.get_content_type(),
                }
        except urllib.error.HTTPError as exc:
            if exc.code in retryable and attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            if exc.code in {401, 403}:
                status = "authentication_error"
            elif exc.code == 429:
                status = "rate_limited"
            elif exc.code in retryable:
                status = "transient_error"
            else:
                status = "provider_error"
            return None, {
                "status": status,
                "httpStatus": exc.code,
                "message": str(exc.reason),
            }
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            reason = getattr(exc, "reason", exc)
            return None, {
                "status": "network_error",
                "message": str(reason),
            }
    raise AssertionError("unreachable provider retry state")


def _fetch_metatile(tile_id: str, token: str) -> tuple[bytes | None, dict]:
    """Fetch and reproject one WMS metatile without database access."""

    request = _request_spec(tile_id)
    query = urllib.parse.urlencode({"token": token, **request["params"]})
    response_bytes, response = _http_get(f"{request['endpoint']}?{query}")
    base = {
        "provider": request["provider"],
        "dataset": request["dataset"],
        "tileId": tile_id,
        "metatileId": request["metatileId"],
    }
    if response_bytes is None:
        return None, {**base, **response}

    response_metadata = {
        "httpStatus": response["httpStatus"],
        "responseContentType": response["contentType"],
        "responseContentLength": len(response_bytes),
        "responseDigest": hashlib.sha256(response_bytes).hexdigest(),
    }
    try:
        source_image = Image.open(io.BytesIO(response_bytes)).convert("RGB")
        source_image.load()
    except (OSError, TypeError, ValueError, UnidentifiedImageError):
        return None, {
            **base,
            **response_metadata,
            "status": "corrupt_response",
        }

    source_pixels = np.asarray(source_image, dtype=np.uint8)
    resolution = request["width"]
    source_transform = transform_from_bounds(
        *request["bbox"],
        source_image.width,
        source_image.height,
    )
    destination_transform = transform_from_bounds(
        *request["sourceBbox"],
        resolution,
        resolution,
    )
    destination = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=source_pixels[:, :, band],
            destination=destination[:, :, band],
            src_transform=source_transform,
            src_crs=CRS.from_epsg(3184),
            dst_transform=destination_transform,
            dst_crs=CRS.from_epsg(3413),
            resampling=Resampling.lanczos,
        )

    image = Image.fromarray(destination)
    no_coverage = _no_coverage_kind(image, warp_void_limit=99.0)
    output = io.BytesIO()
    image.save(output, format="PNG")
    image_bytes = output.getvalue()
    zero_percent = float((destination.max(axis=2) == 0).mean() * 100.0)
    metadata = {
        **base,
        **response_metadata,
        "status": "no_coverage" if no_coverage else "success",
        "coverage": no_coverage or "imagery",
        "width": image.width,
        "height": image.height,
        "format": "PNG",
        "contentLength": len(image_bytes),
        "digest": hashlib.sha256(image_bytes).hexdigest(),
        "zeroPercent": zero_percent,
    }
    if no_coverage:
        return None, metadata
    return image_bytes, metadata


@visible
def dataforsyningen_request(tile_id: str) -> dict:
    """Describe one aligned imagery request without network or DB access.

    Example:
        dataforsyningen_request("10-328-212")
    """

    return _request_spec(tile_id)


@visible
def dataforsyningen_fetch(tile_id: str) -> dict:
    """Fetch one live metatile and return metadata without persistence."""

    token = os.environ.get("DATAFORSYNINGEN_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "DATAFORSYNINGEN_TOKEN is required for live imagery requests"
        )
    _, metadata = _fetch_metatile(tile_id, token)
    return metadata
