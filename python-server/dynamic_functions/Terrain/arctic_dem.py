"""ArcticDEM request construction for explicit terrain tile IDs."""

import hashlib
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds as window_from_bounds

from dynamic_functions.Terrain.Database.tiles import GRID_N
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import tile_bounds


_GRID_ORIGIN = -4_000_000
_SOURCE_TILE_SIZE = 100_000
_URL_TEMPLATE = (
    "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
    "arcticdem/mosaics/v4.1/10m/{row}_{column}/"
    "{row}_{column}_10m_v4.1_dem.tif"
)
_NODATA = -9999.0


def _source_tile_for_point(x: float, y: float) -> tuple[int, int]:
    column = math.floor((x - _GRID_ORIGIN) / _SOURCE_TILE_SIZE) + 1
    row = math.floor((y - _GRID_ORIGIN) / _SOURCE_TILE_SIZE) + 1
    return row, column


def _sources_for_bbox(
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    x0, y0, x1, y1 = bbox
    first_row, first_column = _source_tile_for_point(x0, y0)
    last_row, last_column = _source_tile_for_point(x1, y1)
    return [
        {
            "row": row,
            "column": column,
            "url": _URL_TEMPLATE.format(row=row, column=column),
        }
        for row in range(first_row, last_row + 1)
        for column in range(first_column, last_column + 1)
    ]


def _resample_native(
    data: np.ndarray,
    transform,
    bbox: tuple[float, float, float, float],
    resolution: int,
) -> np.ndarray:
    """Bilinearly sample a raster in world coordinates, matching Flask."""

    height, width = data.shape
    output_x = np.linspace(bbox[0], bbox[2], resolution)
    output_y = np.linspace(bbox[1], bbox[3], resolution)
    source_columns = (output_x - transform.c) / transform.a
    source_rows = (output_y - transform.f) / transform.e

    column0 = np.clip(np.floor(source_columns).astype(int), 0, width - 2)
    column1 = column0 + 1
    column_fraction = np.clip(source_columns - column0, 0, 1).astype(np.float32)
    row0 = np.clip(np.floor(source_rows).astype(int), 0, height - 2)
    row1 = row0 + 1
    row_fraction = np.clip(source_rows - row0, 0, 1).astype(np.float32)

    value00 = data[np.ix_(row0, column0)]
    value01 = data[np.ix_(row0, column1)]
    value10 = data[np.ix_(row1, column0)]
    value11 = data[np.ix_(row1, column1)]
    result = (
        value00 * (1 - row_fraction[:, None]) * (1 - column_fraction[None, :])
        + value01 * (1 - row_fraction[:, None]) * column_fraction[None, :]
        + value10 * row_fraction[:, None] * (1 - column_fraction[None, :])
        + value11 * row_fraction[:, None] * column_fraction[None, :]
    )

    result[
        np.isnan(value00)
        | np.isnan(value01)
        | np.isnan(value10)
        | np.isnan(value11)
    ] = np.nan
    result[:, (source_columns < 0) | (source_columns > width - 1)] = np.nan
    result[(source_rows < 0) | (source_rows > height - 1), :] = np.nan
    return result.astype(np.float32)


def _decode_source(
    source: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: int = GRID_N,
) -> np.ndarray:
    """Decode one raster source into the canonical terrain grid."""

    with rasterio.open(source) as dataset:
        window = window_from_bounds(*bbox, dataset.transform)
        column0 = int(np.floor(window.col_off)) - 1
        row0 = int(np.floor(window.row_off)) - 1
        column1 = int(np.ceil(window.col_off + window.width)) + 1
        row1 = int(np.ceil(window.row_off + window.height)) + 1
        integer_window = Window.from_slices(
            rows=(row0, row1),
            cols=(column0, column1),
            boundless=True,
        )
        data = dataset.read(1, window=integer_window, boundless=True).astype(
            np.float32
        )
        data[data <= _NODATA] = np.nan
        return _resample_native(
            data,
            dataset.window_transform(integer_window),
            bbox,
            resolution,
        )


def _heightmap_summary(heightmap: np.ndarray) -> dict:
    """Return stable, JSON-safe metadata for a decoded heightmap."""

    valid = heightmap[np.isfinite(heightmap)]
    return {
        "shape": list(heightmap.shape),
        "dtype": str(heightmap.dtype),
        "minimum": float(np.min(valid)) if valid.size else None,
        "maximum": float(np.max(valid)) if valid.size else None,
        "nanCount": int(np.isnan(heightmap).sum()),
        "digest": hashlib.sha256(heightmap.tobytes()).hexdigest(),
    }


def _fetch_heightmap(tile_id: str) -> tuple[np.ndarray, list[dict]]:
    """Fetch and merge the ArcticDEM COG windows needed by one tile."""

    bbox = tile_bounds(tile_id, GREENLAND_BBOX)
    sources = _sources_for_bbox(bbox)
    heightmap = None

    for source in sources:
        decoded = _decode_source(source["url"], bbox)
        if heightmap is None:
            heightmap = decoded
        else:
            fill = np.isnan(heightmap) & np.isfinite(decoded)
            heightmap[fill] = decoded[fill]

    if heightmap is None:
        raise RuntimeError(f"No ArcticDEM sources found for {tile_id}")
    return heightmap, sources


@visible
def arcticdem_request(tile_id: str) -> dict:
    """Describe the ArcticDEM COG request for one terrain tile.

    This function performs no network or database access.

    Example:
        >>> request = arcticdem_request("10-334-192")
        >>> request["bbox"]
        [-358377.4375, -2839827.5, -355740.71875, -2837190.78125]
        >>> request["sources"][0]["url"]
        'https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/10m/12_37/12_37_10m_v4.1_dem.tif'
    """

    bbox = tile_bounds(tile_id, GREENLAND_BBOX)
    return {
        "provider": "arcticdem",
        "dataset": "mosaics/v4.1/10m",
        "tileId": tile_id,
        "crs": "EPSG:3413",
        "bbox": list(bbox),
        "resolution": GRID_N,
        "sources": _sources_for_bbox(bbox),
    }


@visible
def arcticdem_fetch(tile_id: str) -> dict:
    """Fetch and decode one ArcticDEM tile without database access.

    Network and provider failures are raised to the caller. No result from
    this function is treated as persisted terrain data.

    Example:
        arcticdem_fetch("10-334-192")
    """

    heightmap, sources = _fetch_heightmap(tile_id)
    return {
        "provider": "arcticdem",
        "dataset": "mosaics/v4.1/10m",
        "tileId": tile_id,
        "sources": sources,
        **_heightmap_summary(heightmap),
    }
