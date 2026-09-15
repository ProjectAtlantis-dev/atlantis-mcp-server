"""Copernicus GLO-30 request construction and DEM decoding."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from dynamic_functions.Terrain.Database.tiles import GRID_N
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import tile_bounds


_DATASET = "Copernicus DEM GLO-30"
_URL_ROOT = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
_EPSG_3413 = CRS.from_epsg(3413)
_EPSG_4326 = CRS.from_epsg(4326)
_TO_WGS84 = Transformer.from_crs(3413, 4326, always_xy=True)


def _source_name(latitude_degree: int, longitude_degree: int) -> str:
    if not -90 <= latitude_degree <= 89:
        raise ValueError(f"Copernicus latitude cell is invalid: {latitude_degree}")
    if not -180 <= longitude_degree <= 179:
        raise ValueError(
            f"Copernicus longitude cell is invalid: {longitude_degree}"
        )
    north_south = "N" if latitude_degree >= 0 else "S"
    east_west = "E" if longitude_degree >= 0 else "W"
    return (
        "Copernicus_DSM_COG_10_"
        f"{north_south}{abs(latitude_degree):02d}_00_"
        f"{east_west}{abs(longitude_degree):03d}_00_DEM"
    )


def _source(latitude_degree: int, longitude_degree: int) -> dict:
    name = _source_name(latitude_degree, longitude_degree)
    return {
        "latitudeDegree": latitude_degree,
        "longitudeDegree": longitude_degree,
        "name": name,
        "url": f"{_URL_ROOT}/{name}/{name}.tif",
    }


def _sources_for_bbox(
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    west, south, east, north = transform_bounds(
        _EPSG_3413,
        _EPSG_4326,
        *bbox,
        densify_pts=21,
    )
    first_latitude = math.floor(south)
    last_latitude = math.ceil(north) - 1
    first_longitude = math.floor(west)
    last_longitude = math.ceil(east) - 1
    return [
        _source(latitude, longitude)
        for latitude in range(first_latitude, last_latitude + 1)
        for longitude in range(first_longitude, last_longitude + 1)
    ]


def _decode_source(
    source: str,
    bbox: tuple[float, float, float, float],
    resolution: int = GRID_N,
) -> np.ndarray:
    """Reproject one WGS84 Copernicus COG onto the terrain grid."""

    north_up = np.full((resolution, resolution), np.nan, dtype=np.float32)
    destination_transform = transform_from_bounds(*bbox, resolution, resolution)
    with rasterio.open(source) as dataset:
        reproject(
            source=rasterio.band(dataset, 1),
            destination=north_up,
            src_nodata=dataset.nodata,
            dst_transform=destination_transform,
            dst_crs=_EPSG_3413,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )
    # Terrain heightmaps are south-up: row zero corresponds to bbox y_min.
    return np.flipud(north_up).copy()


def _fetch_heightmap(tile_id: str) -> tuple[np.ndarray, list[dict]]:
    """Fetch and merge every Copernicus COG intersecting one terrain tile."""

    bbox = tile_bounds(tile_id, GREENLAND_BBOX)
    sources = _sources_for_bbox(bbox)
    heightmap = np.full((GRID_N, GRID_N), np.nan, dtype=np.float32)
    for source in sources:
        decoded = _decode_source(source["url"], bbox)
        fill = np.isnan(heightmap) & np.isfinite(decoded)
        heightmap[fill] = decoded[fill]
    return heightmap, sources


def _heightmap_summary(heightmap: np.ndarray) -> dict:
    valid = heightmap[np.isfinite(heightmap)]
    return {
        "shape": list(heightmap.shape),
        "dtype": str(heightmap.dtype),
        "minimum": float(np.min(valid)) if valid.size else None,
        "maximum": float(np.max(valid)) if valid.size else None,
        "nanCount": int(np.isnan(heightmap).sum()),
        "digest": hashlib.sha256(heightmap.tobytes()).hexdigest(),
    }


@visible
def copernicus_request(tile_id: str) -> dict:
    """Describe the Copernicus GLO-30 requests for one terrain tile."""

    bbox = tile_bounds(tile_id, GREENLAND_BBOX)
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    longitude, latitude = _TO_WGS84.transform(center_x, center_y)
    return {
        "provider": "copernicus",
        "dataset": _DATASET,
        "tileId": tile_id,
        "crs": "EPSG:3413",
        "bbox": list(bbox),
        "center": {"latitude": latitude, "longitude": longitude},
        "resolution": GRID_N,
        "verticalDatum": "EGM2008",
        "sources": _sources_for_bbox(bbox),
    }


@visible
def copernicus_fetch(tile_id: str) -> dict:
    """Fetch and decode one Copernicus tile without database access."""

    heightmap, sources = _fetch_heightmap(tile_id)
    return {
        "provider": "copernicus",
        "dataset": _DATASET,
        "tileId": tile_id,
        "sources": sources,
        "verticalDatum": "EGM2008",
        **_heightmap_summary(heightmap),
    }
