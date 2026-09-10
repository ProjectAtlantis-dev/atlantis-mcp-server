"""Reversible ocean-gap correction using bounded local reference mosaics.

Local mosaics classify connected gaps across tile boundaries. Fine LODs share
depth-10 reference imagery; coarser requests use their native depth. Repairs are
gated by current coastline and texture colours. Provider bytes are never written.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import sqlite3

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_propagation

from dynamic_functions.Terrain.coastline import read_coastline_mask
from dynamic_functions.Terrain.Database.textures import read_texture_with_ancestor
from dynamic_functions.Terrain.ocean_texture import (
    texture_pixel_area_m2, white_ocean_repair_mask,
)
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import ancestor_tile_ids, tile_bounds


VERSION = "ocean-gap-v2"
FILL_RGB = (10, 20, 25)
# Use a common depth for fine imagery, and native depth for coarser requests.
# A 3x3 halo bounds work and includes components crossing the requested tile.
DEPTH = 10
SIZE = 256


def _reference_ids(tile_id: str) -> tuple[str, ...]:
    depth, column, row = map(int, tile_id.split("-"))
    shift = max(0, depth - DEPTH)
    depth -= shift
    column >>= shift
    row >>= shift
    limit = 1 << depth
    return tuple(
        f"{depth}-{x}-{y}"
        for y in range(max(0, row - 1), min(limit, row + 2))
        for x in range(max(0, column - 1), min(limit, column + 2))
    )


@dataclass(eq=False, frozen=True)
class _Snapshot:
    approved: np.ndarray
    bbox: tuple[float, float, float, float]


@lru_cache(maxsize=16)
def _classify(ids: tuple[str, ...], rows: tuple) -> _Snapshot:
    addresses = [tuple(map(int, tile_id.split("-"))) for tile_id in ids]
    west = min(a[1] for a in addresses)
    east = max(a[1] for a in addresses)
    south = min(a[2] for a in addresses)
    north = max(a[2] for a in addresses)
    sw = tile_bounds(ids[0], GREENLAND_BBOX)
    ne = tile_bounds(ids[-1], GREENLAND_BBOX)
    bbox = (sw[0], sw[1], ne[2], ne[3])
    shape = ((north - south + 1) * SIZE, (east - west + 1) * SIZE)
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    ocean = np.zeros(shape, dtype=bool)
    for tile_id, source_id, texture, coast_id, coast_shape, coast_bytes in rows:
        with Image.open(io.BytesIO(texture)) as image:
            image = image.convert("RGB")
            if source_id != tile_id or image.size != (SIZE, SIZE):
                tx0, ty0, tx1, ty1 = tile_bounds(tile_id, GREENLAND_BBOX)
                sx0, sy0, sx1, sy1 = tile_bounds(source_id, GREENLAND_BBOX)
                width, height = image.size
                extent = (
                    (tx0 - sx0) / (sx1 - sx0) * width,
                    (sy1 - ty1) / (sy1 - sy0) * height,
                    (tx1 - sx0) / (sx1 - sx0) * width,
                    (sy1 - ty0) / (sy1 - sy0) * height,
                )
                image = image.transform((SIZE, SIZE), Image.Transform.EXTENT,
                                        extent, Image.Resampling.BILINEAR)
            pixels = np.asarray(image)
        _, column, row = map(int, tile_id.split("-"))
        x = (column - west) * SIZE
        y = (north - row) * SIZE
        rgb[y:y + SIZE, x:x + SIZE] = pixels
        ocean[y:y + SIZE, x:x + SIZE] = _ocean_at(
            tile_id, (SIZE, SIZE), coast_id, coast_shape, coast_bytes,
        )
    approved = white_ocean_repair_mask(
        rgb, ocean, pixel_area_m2=texture_pixel_area_m2(ids[0], (SIZE, SIZE)),
    )
    # One reference-pixel allowance for footprint sampling between imagery LODs.
    # The final repair still requires current authoritative ocean and bright RGB.
    approved = binary_dilation(approved, iterations=1)
    approved.setflags(write=False)
    return _Snapshot(approved, bbox)


def _coastline(connection: sqlite3.Connection, tile_id: str):
    for candidate in ancestor_tile_ids(tile_id, include_self=True):
        coastline = read_coastline_mask(connection, candidate)
        if coastline is not None:
            coast = coastline["mask"]
            if min(coast.shape) < 2:
                raise ValueError(f"coastline vertex grid too small for {candidate}")
            return candidate, coast.shape, coast.tobytes()
    return None


def _reference_evidence(connection: sqlite3.Connection, tile_id: str) -> tuple:
    ids = _reference_ids(tile_id)
    rows = []
    for reference_id in ids:
        texture = read_texture_with_ancestor(connection, reference_id)
        if texture is None:
            continue
        coast = _coastline(connection, reference_id)
        if coast is None:
            continue
        rows.append((reference_id, texture["resolved_tile_id"],
                     bytes(texture["texture"]), *coast))
    # Content keys invalidate approvals when imagery or coastline arrives or changes.
    # No provider acquisition, database writes, or process-wide mosaic is needed.
    return ids, tuple(rows)


def _centres(tile_id: str, shape: tuple[int, int]):
    x0, y0, x1, y1 = tile_bounds(tile_id, GREENLAND_BBOX)
    height, width = shape
    return (
        x0 + (np.arange(width) + 0.5) * (x1 - x0) / width,
        y1 - (np.arange(height) + 0.5) * (y1 - y0) / height,
    )


def _ocean_at(tile_id, shape, coast_id, coast_shape, coast_bytes):
    """Require four sea vertices at each target texel, including ancestor masks."""
    x, y = _centres(tile_id, shape)
    coast = np.frombuffer(coast_bytes, dtype=bool).reshape(coast_shape)
    cx0, cy0, cx1, cy1 = tile_bounds(coast_id, GREENLAND_BBOX)
    ch, cw = coast.shape
    vx = np.clip(np.floor((x - cx0) / (cx1 - cx0) * (cw - 1)).astype(int), 0, cw - 2)
    vy = np.clip(np.floor((y - cy0) / (cy1 - cy0) * (ch - 1)).astype(int), 0, ch - 2)
    return (coast[np.ix_(vy, vx)] & coast[np.ix_(vy + 1, vx)]
             & coast[np.ix_(vy, vx + 1)] & coast[np.ix_(vy + 1, vx + 1)])


@lru_cache(maxsize=128)
def _render(tile_id: str, payload: bytes, snapshot: _Snapshot,
            coast_id: str, coast_shape: tuple[int, int], coast_bytes: bytes) -> tuple:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    x, y = _centres(tile_id, (height, width))
    mx0, my0, mx1, my1 = snapshot.bbox
    mh, mw = snapshot.approved.shape
    ix = np.clip(((x - mx0) / (mx1 - mx0) * mw).astype(int), 0, mw - 1)
    iy = np.clip(((my1 - y) / (my1 - my0) * mh).astype(int), 0, mh - 1)
    approved = snapshot.approved[np.ix_(iy, ix)].copy()

    ocean = _ocean_at(tile_id, (height, width), coast_id, coast_shape, coast_bytes)
    bright_ocean = (ocean & (rgb.min(axis=2) >= 64)
                    & (np.ptp(rgb.astype(np.int16), axis=2) <= 32))
    # Refine the shared approval against this LOD's actual connected footprint.
    # This includes narrow tips missed by coarse sampling, but cannot jump a
    # dark-water gap to a separate iceberg or cross coastline-defined land.
    repair = binary_propagation(approved & bright_ocean, mask=bright_ocean)
    count = int(repair.sum())
    if not count:
        return payload, "image/jpeg", 0
    corrected = rgb.copy()
    corrected[repair] = FILL_RGB
    output = io.BytesIO()
    # Lossless encoding preserves every unmodified decoded source pixel.
    Image.fromarray(corrected).save(output, format="PNG")
    return output.getvalue(), "image/png", count


def _evidence_digest(tile_id, payload, coastline, evidence):
    """Hash actual content, including missing neighbors and algorithm settings."""
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, tuple):
            digest.update(b"T" + len(value).to_bytes(8, "big"))
            for item in value:
                add(item)
        else:
            data = value if isinstance(value, bytes) else str(value).encode("utf-8")
            digest.update(b"B" + len(data).to_bytes(8, "big") + data)

    add((VERSION, DEPTH, SIZE, FILL_RGB, tile_id, payload, coastline, evidence))
    return digest.hexdigest()


def repair_texture(connection: sqlite3.Connection, tile_id: str, payload: bytes,
                   *, persist: bool = False) -> tuple:
    """Read cached repairs; HTTP serving may persist derived results only.

    Composition remains read-only. One row per requested tile replaces stale
    revisions, including negative detections without duplicating source bytes.
    """
    ancestor_tile_ids(tile_id, include_self=True)
    coastline = _coastline(connection, tile_id)
    if coastline is None:
        return payload, "image/jpeg", 0
    evidence = _reference_evidence(connection, tile_id)
    key = _evidence_digest(tile_id, payload, coastline, evidence)
    cached = connection.execute(
        "SELECT texture,media_type,repaired_pixels FROM ocean_texture_repairs "
        "WHERE tile_id=? AND evidence_digest=?", (tile_id, key),
    ).fetchone()
    if cached is not None:
        return (bytes(cached[0]) if cached[0] is not None else payload,
                cached[1], cached[2])
    snapshot = _classify(*evidence)
    result = (_render(tile_id, payload, snapshot, *coastline)
              if snapshot.approved.any() else (payload, "image/jpeg", 0))
    if persist:
        # RELEASE commits only when there is no enclosing caller transaction.
        # Never commit or roll back unrelated work on the shared connection.
        connection.execute("SAVEPOINT ocean_texture_cache")
        try:
            connection.execute(
                "INSERT INTO ocean_texture_repairs "
                "(tile_id,evidence_digest,texture,media_type,repaired_pixels) "
                "VALUES (?,?,?,?,?) ON CONFLICT(tile_id) DO UPDATE SET "
                "evidence_digest=excluded.evidence_digest,texture=excluded.texture,"
                "media_type=excluded.media_type,repaired_pixels=excluded.repaired_pixels",
                (tile_id, key, result[0] if result[2] else None, result[1], result[2]),
            )
            connection.execute("RELEASE ocean_texture_cache")
        except Exception:
            connection.execute("ROLLBACK TO ocean_texture_cache")
            connection.execute("RELEASE ocean_texture_cache")
            raise
    return result
