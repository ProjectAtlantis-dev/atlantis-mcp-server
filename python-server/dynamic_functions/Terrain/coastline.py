"""Explicit GTK50 authoritative coastline acquisition for terrain tiles."""

from __future__ import annotations

import datetime
import hashlib
import os
import sqlite3
import subprocess
import uuid
import zlib
from pathlib import Path

import atlantis
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely import wkb as shapely_wkb
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shapely_transform

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.tiles import ensure_tile_row
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import require_tile_id, tile_bounds


SOURCE = "gtk50_vector"
VERSION = 2
GRID_N = 65
BLOCK_SIZE_M = 100_000
OVERSAMPLE = 8
EDGE_PAD_M = 100.0
FTP_HOST = "ftp.dataforsyningen.dk"
FTP_DIRECTORY = "/DATABOKS_GROENLAND/Vektor_50000"
FTP_BASE_URL = f"ftps://{FTP_HOST}{FTP_DIRECTORY}"
BLOCK_DIR = Path(__file__).with_name("Coastline") / "blocks"

_TO_UTM = Transformer.from_crs(3413, 3184, always_xy=True)
_TO_STEREO = Transformer.from_crs(3184, 3413, always_xy=True)


class CoastlineClobberError(RuntimeError):
    """Raised when a coastline write would replace different source data."""


def _encoded_mask(mask: np.ndarray) -> tuple[np.ndarray, bytes]:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("coastline mask must be a 2D array")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("coastline mask dimensions must be positive")
    canonical = values.astype(np.uint8)
    if not np.all((canonical == 0) | (canonical == 1)):
        raise ValueError("coastline mask must contain only boolean values")
    return canonical, zlib.compress(canonical.tobytes(), level=6)


def write_coastline_mask(
    connection: sqlite3.Connection,
    tile_id: str,
    mask: np.ndarray,
    source: str,
    version: int,
    *,
    commit: bool = True,
) -> bool:
    """Store one exact authoritative mask without replacing valid payloads."""

    require_tile_id(tile_id)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("coastline source must be a non-empty string")
    if not isinstance(version, int) or version < 1:
        raise ValueError("coastline version must be a positive integer")
    canonical, encoded = _encoded_mask(mask)
    ensure_tile_row(connection, tile_id)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor = connection.execute(
        "INSERT OR IGNORE INTO coastline_masks "
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
        "SELECT width, height, mask, source, version FROM coastline_masks "
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
    raise CoastlineClobberError(
        f"Refusing to clobber coastline mask {tile_id}: "
        f"existing source={row[3] if row else None}, "
        f"version={row[4] if row else None}; incoming source={source}, "
        f"version={version}"
    )


def read_coastline_mask(
    connection: sqlite3.Connection,
    tile_id: str,
) -> dict | None:
    """Read one authoritative coastline mask without transforming it."""

    require_tile_id(tile_id)
    row = connection.execute(
        "SELECT width, height, mask, source, version, updated_at "
        "FROM coastline_masks WHERE tile_id = ?",
        (tile_id,),
    ).fetchone()
    if row is None:
        return None
    width, height = int(row[0]), int(row[1])
    values = np.frombuffer(zlib.decompress(row[2]), dtype=np.uint8)
    if values.size != width * height:
        raise ValueError(
            f"coastline mask for {tile_id} has {values.size} values; "
            f"expected {width * height}"
        )
    mask = values.reshape((height, width)).astype(bool)
    return {
        "tile_id": tile_id,
        "width": width,
        "height": height,
        "mask": mask,
        "source": row[3],
        "version": int(row[4]),
        "updated_at": row[5],
        "digest": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _block_name(north: int, east: int) -> str:
    return f"{north}_{east:02d}"


def _block_filename(block: str) -> str:
    return f"GL50_Vektordata_100km_{block}.gpkg"


def _block_path(block: str) -> Path:
    return BLOCK_DIR / _block_filename(block)


def _blocks_for_bbox(
    bbox: tuple[float, float, float, float],
) -> list[str]:
    """Return every 100 km GTK50 block intersecting an EPSG:3413 bbox."""

    x0, y0, x1, y1 = map(float, bbox)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x0, x1, 5),
        np.linspace(y0, y1, 5),
    )
    utm_x, utm_y = _TO_UTM.transform(grid_x.ravel(), grid_y.ravel())
    east_min = int((utm_x.min() - EDGE_PAD_M) // BLOCK_SIZE_M)
    east_max = int((utm_x.max() + EDGE_PAD_M) // BLOCK_SIZE_M)
    north_min = int((utm_y.min() - EDGE_PAD_M) // BLOCK_SIZE_M)
    north_max = int((utm_y.max() + EDGE_PAD_M) // BLOCK_SIZE_M)
    return [
        _block_name(north, east)
        for north in range(north_min, north_max + 1)
        for east in range(east_min, east_max + 1)
    ]


def _request_spec(tile_id: str) -> dict:
    bbox = tile_bounds(tile_id, GREENLAND_BBOX)
    blocks = _blocks_for_bbox(bbox)
    return {
        "provider": "dataforsyningen",
        "dataset": "Vektor_50000",
        "artifact": "authoritative_tidal_sea_mask",
        "tileId": tile_id,
        "crs": "EPSG:3413",
        "bbox": list(bbox),
        "resolution": GRID_N,
        "source": SOURCE,
        "version": VERSION,
        "credentialEnvironmentVariables": [
            "DATAFORSYNINGEN_FTP_USER",
            "DATAFORSYNINGEN_FTP_PASS",
        ],
        "blocks": [
            {
                "blockId": block,
                "filename": _block_filename(block),
                "remoteDirectory": FTP_DIRECTORY,
            }
            for block in blocks
        ],
    }


def _gpkg_wkb(blob: bytes) -> bytes:
    if len(blob) < 8 or blob[:2] != b"GP":
        raise ValueError("invalid GeoPackage geometry header")
    envelope = (blob[3] >> 1) & 0x07
    envelope_length = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope)
    if envelope_length is None or len(blob) < 8 + envelope_length:
        raise ValueError("invalid GeoPackage geometry envelope")
    return bytes(blob[8 + envelope_length :])


def _read_block(path: Path) -> tuple[list[Polygon], list[Polygon]]:
    """Read provider GeoPackage polygons without touching the terrain DB."""

    if not path.is_file():
        raise FileNotFoundError(path)
    water: list[Polygon] = []
    islands: list[Polygon] = []
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for table, target in (("tidalwater_s", water), ("island_s", islands)):
            try:
                rows = source.execute(f'SELECT geom FROM "{table}"').fetchall()
            except sqlite3.OperationalError as exc:
                raise ValueError(f"GTK50 block is missing {table}") from exc
            for (blob,) in rows:
                if blob is None:
                    continue
                geometry = shapely_wkb.loads(_gpkg_wkb(blob))
                geometry = shapely_transform(_TO_STEREO.transform, geometry)
                parts = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
                target.extend(part for part in parts if isinstance(part, Polygon))
    finally:
        source.close()
    if not water and not islands:
        raise ValueError(f"GTK50 block contains no coastline geometry: {path}")
    return water, islands


def _rasterize(
    bbox: tuple[float, float, float, float],
    resolution: int,
    blocks: list[tuple[list[Polygon], list[Polygon]]],
) -> np.ndarray:
    """Rasterize GTK50 sea-minus-island polygons to a south-first mask."""

    x0, y0, x1, y1 = map(float, bbox)
    size = int(resolution) * OVERSAMPLE
    scale_x = size / (x1 - x0)
    scale_y = size / (y1 - y0)

    def pixels(ring) -> list[tuple[float, float]]:
        coordinates = np.asarray(ring.coords)
        return list(
            zip(
                (coordinates[:, 0] - x0) * scale_x,
                (y1 - coordinates[:, 1]) * scale_y,
            )
        )

    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    for water, _ in blocks:
        for polygon in water:
            if not polygon.intersects(Polygon.from_bounds(x0, y0, x1, y1)):
                continue
            draw.polygon(pixels(polygon.exterior), fill=1)
            for interior in polygon.interiors:
                draw.polygon(pixels(interior), fill=0)
    for _, islands in blocks:
        for polygon in islands:
            if not polygon.intersects(Polygon.from_bounds(x0, y0, x1, y1)):
                continue
            draw.polygon(pixels(polygon.exterior), fill=0)
    high_resolution = np.asarray(image, dtype=np.uint8)
    fractions = high_resolution.reshape(
        resolution,
        OVERSAMPLE,
        resolution,
        OVERSAMPLE,
    ).mean(axis=(1, 3))
    return np.flipud(fractions >= 0.5)


def _decode_blocks(tile_id: str, paths: dict[str, Path]) -> np.ndarray:
    request = _request_spec(tile_id)
    required = [item["blockId"] for item in request["blocks"]]
    missing = [block for block in required if block not in paths]
    if missing:
        raise ValueError("missing required GTK50 blocks: " + ", ".join(missing))
    parsed = [_read_block(paths[block]) for block in required]
    return _rasterize(tuple(request["bbox"]), request["resolution"], parsed)


def _credentials() -> tuple[str, str]:
    username = os.environ.get("DATAFORSYNINGEN_FTP_USER", "").strip()
    password = os.environ.get("DATAFORSYNINGEN_FTP_PASS", "").strip()
    if not username or not password:
        raise RuntimeError(
            "DATAFORSYNINGEN_FTP_USER and DATAFORSYNINGEN_FTP_PASS are "
            "required for GTK50 acquisition"
        )
    return username, password


def _download_block(block: str, username: str, password: str) -> Path:
    """Atomically cache one immutable GTK50 provider block."""

    target = _block_path(block)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f".{uuid.uuid4().hex}.part")
    try:
        subprocess.run(
            [
                "curl",
                "-sS",
                "--fail",
                "--max-time",
                "600",
                "--user",
                f"{username}:{password}",
                f"{FTP_BASE_URL}/{_block_filename(block)}",
                "-o",
                str(staging),
            ],
            check=True,
        )
        # Parse before publishing so corrupt/truncated downloads never become
        # valid cache entries.
        _read_block(staging)
        staging.replace(target)
        return target
    finally:
        staging.unlink(missing_ok=True)


def _acquire_mask(tile_id: str) -> tuple[np.ndarray, dict]:
    request = _request_spec(tile_id)
    blocks = [item["blockId"] for item in request["blocks"]]
    paths = {block: _block_path(block) for block in blocks if _block_path(block).is_file()}
    missing = [block for block in blocks if block not in paths]
    if missing:
        username, password = _credentials()
        paths.update(
            {
                block: _download_block(block, username, password)
                for block in missing
            }
        )
    mask = _decode_blocks(tile_id, paths)
    digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
    return mask, {
        **request,
        "status": "success",
        "blockCount": len(paths),
        "waterCount": int(mask.sum()),
        "landCount": int(mask.size - mask.sum()),
        "digest": digest,
    }


def _read_response(connection, tile_id: str) -> dict:
    payload = read_coastline_mask(connection, tile_id)
    if payload is None:
        return {"tileId": tile_id, "found": False}
    return {
        "tileId": tile_id,
        "found": True,
        "source": payload["source"],
        "version": payload["version"],
        "updatedAt": payload["updated_at"],
        "shape": [payload["height"], payload["width"]],
        "waterCount": int(payload["mask"].sum()),
        "landCount": int(payload["mask"].size - payload["mask"].sum()),
        "digest": payload["digest"],
    }


@visible
async def list_coastlines() -> list[tuple]:
    """Return storage metadata for every persisted coastline mask."""

    return await atlantis.client_command(
        "@Database/query 'SELECT tile_id,width,height,source,version,updated_at FROM coastline_masks'"
    )


@visible
def coastline_request(tile_id: str) -> dict:
    """Describe required GTK50 source blocks without network or DB access."""

    return _request_spec(tile_id)


@visible
def read_coastline(tile_id: str) -> dict:
    """Return metadata and digest for one stored authoritative sea mask."""

    return _read_response(db(), tile_id)


@visible
def fetch_coastline(tile_id: str) -> dict:
    """Acquire, rasterize, and persist one authoritative GTK50 sea mask."""

    # Provider acquisition and rasterization complete before the shared
    # terrain connection is requested. Failures cannot create or alter rows.
    mask, acquisition = _acquire_mask(tile_id)
    connection = db()
    written = write_coastline_mask(
        connection,
        tile_id,
        mask,
        SOURCE,
        VERSION,
    )
    return {**acquisition, "written": written, **_read_response(connection, tile_id)}
