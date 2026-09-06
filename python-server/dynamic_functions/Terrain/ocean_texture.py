"""Conservative, read-only detection of uniform white ocean imagery gaps.

This is a candidate detector, not proof of provider NoData. In particular,
uniform sea ice can satisfy the same criteria.
"""

import numpy as np
from scipy.ndimage import binary_dilation, label

from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import tile_bounds


MINIMUM_GAP_AREA_M2 = 10_000.0


def texture_pixel_area_m2(tile_id: str, shape: tuple[int, int]) -> float:
    """Projected EPSG:3413 area of one texture pixel at this tile's LOD."""

    if len(shape) != 2 or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError("texture shape must contain two positive integers")
    x0, y0, x1, y1 = tile_bounds(tile_id, GREENLAND_BBOX)
    return (x1 - x0) * (y1 - y0) / (shape[0] * shape[1])


def texture_ocean_mask(coastline: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Map south-first coastline vertices to north-first texture pixel centres.

    Require all four surrounding vertices to be sea so interpolation cannot
    extend a repair into coastline-defined land.
    """

    coast = np.asarray(coastline)
    if coast.dtype != np.bool_ or coast.ndim != 2 or min(coast.shape) < 2:
        raise ValueError("coastline must be a 2-D boolean vertex grid, at least 2x2")
    if len(shape) != 2 or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError("texture shape must contain two positive integers")
    height, width = shape
    x = np.floor((np.arange(width) + 0.5) / width * (coast.shape[1] - 1)).astype(int)
    y = np.floor((np.arange(height) + 0.5) / height * (coast.shape[0] - 1)).astype(int)
    north_first = np.flipud(coast)
    return (
        north_first[np.ix_(y, x)]
        & north_first[np.ix_(y + 1, x)]
        & north_first[np.ix_(y, x + 1)]
        & north_first[np.ix_(y + 1, x + 1)]
    )


def detect_white_ocean_gaps(
    rgb: np.ndarray, ocean: np.ndarray, *, pixel_area_m2: float,
) -> np.ndarray:
    """Flag sizable uniform near-white components wholly inside known ocean.

    Thresholds are in decoded 8-bit RGB: every channel >=245, channel spread
    <=4, per-channel component standard deviation <=2. Require at least 164
    pixels and one hectare of projected area. The pixel minimum is fixed so
    adjoining a neighbour does not change the test's significance threshold.
    Supply the
    actual pixel area so finer LOD cannot inflate a small object's size.
    No boundary alignment is assumed, and no
    pixels are expanded or filled beyond those satisfying the colour test.
    """

    pixels = np.asarray(rgb)
    sea = np.asarray(ocean)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("texture must be a uint8 RGB array")
    if sea.dtype != np.bool_ or sea.shape != pixels.shape[:2] or not sea.size:
        raise ValueError("ocean must be a nonempty boolean mask matching the texture")
    if not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("pixel area must be finite and positive")
    candidate = (
        sea
        & (pixels.min(axis=2) >= 245)
        & (np.ptp(pixels.astype(np.int16), axis=2) <= 4)
    )
    components, count = label(candidate)
    sizes = np.bincount(components.ravel())
    minimum = 164
    accepted = np.zeros(count + 1, dtype=bool)
    for component in np.flatnonzero(sizes[1:] >= minimum) + 1:
        if sizes[component] * pixel_area_m2 < MINIMUM_GAP_AREA_M2:
            continue
        values = pixels[components == component]
        accepted[component] = bool(np.all(values.std(axis=0) <= 2.0))
    return accepted[components]


def white_ocean_repair_mask(
    rgb: np.ndarray, ocean: np.ndarray, *, pixel_area_m2: float,
) -> np.ndarray:
    """Include the bright JPEG fringe within three pixels of an accepted gap.

    The fringe cannot initiate a detection, cross known land, or include dark
    ocean. This removes compressed grey/white edge pixels left by the strict
    uniform-white core test without indiscriminately dilating its footprint.
    """

    core = detect_white_ocean_gaps(rgb, ocean, pixel_area_m2=pixel_area_m2)
    nearby = binary_dilation(core, iterations=3)
    pixels = np.asarray(rgb)
    fringe = (
        nearby & ocean
        & (pixels.min(axis=2) >= 64)
        & (np.ptp(pixels.astype(np.int16), axis=2) <= 32)
    )
    return core | fringe
