"""Reversible ocean-gap correction in the southern Greenland trial area.

One fixed-resolution mosaic classifies connected gaps across tile boundaries.
Every requested LOD samples that same approval footprint, then gates repairs
with its current coastline and texture colours. Provider bytes are never written.
"""

from dataclasses import dataclass
from functools import lru_cache
import io
import sqlite3
import zlib

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_propagation

from dynamic_functions.Terrain.coastline import read_coastline_mask
from dynamic_functions.Terrain.ocean_texture import (
    texture_ocean_mask, texture_pixel_area_m2, white_ocean_repair_mask,
)
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import ancestor_tile_ids, tile_bounds


VERSION = "ocean-gap-v1"
FILL_RGB = (10, 20, 25)
# Depth-9 tiles 230..233 / 6..7 include the wedge, offshore block, and real ice.
# Detect at depth 10 with a one-tile halo. No provider acquisition is performed.
DEPTH = 10
SIZE = 256
_COLUMNS = range(459, 469)
_ROWS = range(11, 17)
_IDS = tuple(f"10-{x}-{y}" for y in _ROWS for x in _COLUMNS)
_SW = tile_bounds("10-460-12", GREENLAND_BBOX)
_NE = tile_bounds("10-467-15", GREENLAND_BBOX)
SCOPE_BBOX = (_SW[0], _SW[1], _NE[2], _NE[3])
_HALO_SW = tile_bounds(_IDS[0], GREENLAND_BBOX)
_HALO_NE = tile_bounds(_IDS[-1], GREENLAND_BBOX)
_MOSAIC_BBOX = (_HALO_SW[0], _HALO_SW[1], _HALO_NE[2], _HALO_NE[3])


def in_trial_area(tile_id: str) -> bool:
    x0, y0, x1, y1 = tile_bounds(tile_id, GREENLAND_BBOX)
    sx0, sy0, sx1, sy1 = SCOPE_BBOX
    return x0 < sx1 and x1 > sx0 and y0 < sy1 and y1 > sy0


@dataclass(eq=False, frozen=True)
class _Snapshot:
    approved: np.ndarray


@lru_cache(maxsize=2)
def _classify(rows: tuple) -> _Snapshot:
    shape = (len(_ROWS) * SIZE, len(_COLUMNS) * SIZE)
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    ocean = np.zeros(shape, dtype=bool)
    for tile_id, texture, width, height, mask in rows:
        # Missing evidence is explicitly unclassified, never inferred from RGB.
        if mask is None:
            continue
        coast = np.frombuffer(zlib.decompress(mask), dtype=np.uint8)
        if coast.size != width * height or not np.all((coast == 0) | (coast == 1)):
            raise ValueError(f"invalid coastline mask for {tile_id}")
        coast = coast.reshape(height, width).astype(bool)
        with Image.open(io.BytesIO(texture)) as image:
            if image.size != (SIZE, SIZE):
                raise ValueError(f"expected 256x256 reference texture for {tile_id}")
            pixels = np.asarray(image.convert("RGB"))
        _, column, row = map(int, tile_id.split("-"))
        x = (column - _COLUMNS.start) * SIZE
        y = (_ROWS.stop - 1 - row) * SIZE
        rgb[y:y + SIZE, x:x + SIZE] = pixels
        ocean[y:y + SIZE, x:x + SIZE] = texture_ocean_mask(coast, (SIZE, SIZE))
    approved = white_ocean_repair_mask(
        rgb, ocean, pixel_area_m2=texture_pixel_area_m2(_IDS[0], (SIZE, SIZE)),
    )
    # One reference-pixel allowance for footprint sampling between imagery LODs.
    # The final repair still requires current authoritative ocean and bright RGB.
    approved = binary_dilation(approved, iterations=1)
    approved.setflags(write=False)
    # Release render entries referencing older mosaics as evidence arrives.
    _render.cache_clear()
    return _Snapshot(approved)


def _snapshot(connection: sqlite3.Connection) -> _Snapshot:
    marks = ",".join("?" for _ in _IDS)
    rows = connection.execute(
        "SELECT x.tile_id,x.texture,c.width,c.height,c.mask FROM textures x "
        "LEFT JOIN coastline_masks c ON c.tile_id=x.tile_id "
        f"WHERE x.tile_id IN ({marks}) ORDER BY x.tile_id", _IDS,
    ).fetchall()
    # Content-keyed caching also invalidates when late coastline/texture arrives.
    return _classify(tuple(tuple(row) for row in rows))


def _centres(tile_id: str, shape: tuple[int, int]):
    x0, y0, x1, y1 = tile_bounds(tile_id, GREENLAND_BBOX)
    height, width = shape
    return (
        x0 + (np.arange(width) + 0.5) * (x1 - x0) / width,
        y1 - (np.arange(height) + 0.5) * (y1 - y0) / height,
    )


@lru_cache(maxsize=128)
def _render(tile_id: str, payload: bytes, snapshot: _Snapshot,
            coast_id: str, coast_shape: tuple[int, int], coast_bytes: bytes) -> tuple:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    x, y = _centres(tile_id, (height, width))
    mx0, my0, mx1, my1 = _MOSAIC_BBOX
    mh, mw = snapshot.approved.shape
    ix = np.clip(((x - mx0) / (mx1 - mx0) * mw).astype(int), 0, mw - 1)
    iy = np.clip(((my1 - y) / (my1 - my0) * mh).astype(int), 0, mh - 1)
    approved = snapshot.approved[np.ix_(iy, ix)].copy()
    sx0, sy0, sx1, sy1 = SCOPE_BBOX
    approved &= ((y >= sy0) & (y < sy1))[:, None] & ((x >= sx0) & (x < sx1))[None, :]

    # Sample the actual ancestor vertex grid at target texel centres, requiring
    # all four vertices. Do not enlarge an ancestor's land/sea pixels first.
    coast = np.frombuffer(coast_bytes, dtype=bool).reshape(coast_shape)
    cx0, cy0, cx1, cy1 = tile_bounds(coast_id, GREENLAND_BBOX)
    ch, cw = coast.shape
    vx = np.clip(np.floor((x - cx0) / (cx1 - cx0) * (cw - 1)).astype(int), 0, cw - 2)
    vy = np.clip(np.floor((y - cy0) / (cy1 - cy0) * (ch - 1)).astype(int), 0, ch - 2)
    ocean = (coast[np.ix_(vy, vx)] & coast[np.ix_(vy + 1, vx)]
             & coast[np.ix_(vy, vx + 1)] & coast[np.ix_(vy + 1, vx + 1)])
    bright_ocean = (ocean & (rgb.min(axis=2) >= 64)
                    & (np.ptp(rgb.astype(np.int16), axis=2) <= 32))
    # Refine the shared approval against this LOD's actual connected footprint.
    # This includes narrow tips missed by coarse sampling, but cannot jump a
    # dark-water gap to a separate iceberg or cross coastline-defined land.
    repair = binary_propagation(approved & bright_ocean, mask=bright_ocean)
    repair &= ((y >= sy0) & (y < sy1))[:, None] & ((x >= sx0) & (x < sx1))[None, :]
    count = int(repair.sum())
    if not count:
        return payload, "image/jpeg", 0
    corrected = rgb.copy()
    corrected[repair] = FILL_RGB
    output = io.BytesIO()
    # Lossless encoding preserves every unmodified decoded source pixel.
    Image.fromarray(corrected).save(output, format="PNG")
    return output.getvalue(), "image/png", count


def repair_texture(connection: sqlite3.Connection, tile_id: str, payload: bytes) -> tuple:
    """Return served bytes, media type, and repaired pixel count, without writes."""
    if not in_trial_area(tile_id):
        return payload, "image/jpeg", 0
    snapshot = _snapshot(connection)
    if not snapshot.approved.any():
        return payload, "image/jpeg", 0
    for candidate in ancestor_tile_ids(tile_id, include_self=True):
        coastline = read_coastline_mask(connection, candidate)
        if coastline is not None:
            coast = coastline["mask"]
            if min(coast.shape) < 2:
                raise ValueError(f"coastline vertex grid too small for {candidate}")
            return _render(tile_id, payload, snapshot, candidate, coast.shape, coast.tobytes())
    return payload, "image/jpeg", 0
