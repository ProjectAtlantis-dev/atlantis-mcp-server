"""Read-only visual previews for persisted terrain tiles and textures."""

from __future__ import annotations

import html
import io
import tempfile

import atlantis
import numpy as np
from PIL import Image

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.textures import read_texture_payload
from dynamic_functions.Terrain.Database.tiles import read_dem_payload
from dynamic_functions.Terrain.coastline import read_coastline_mask
from dynamic_functions.Terrain.hydrography import read_hydrography_mask
from dynamic_functions.Terrain.tidal_connectivity import (
    connected_hydrography_for_tile,
)
from dynamic_functions.Terrain.tile_address import require_tile_id


_PREVIEW_SIZE = 520
_SEA_LEVEL_TOLERANCE_METERS = 1.0
_SEA_LEVEL_COLOR = np.asarray((5, 24, 64), dtype=np.uint8)
_MASK_BACKGROUND = np.asarray((24, 29, 35), dtype=np.uint8)
_COASTLINE_COLOR = np.asarray((255, 80, 210), dtype=np.uint8)
_HYDROGRAPHY_COLOR = np.asarray((0, 140, 255), dtype=np.uint8)
_CONNECTED_COLOR = np.asarray((255, 80, 210), dtype=np.uint8)
_TERRAIN_COLORS = np.asarray(
    [
        (31, 78, 82),
        (68, 112, 88),
        (142, 139, 91),
        (139, 125, 111),
        (225, 232, 235),
        (255, 255, 255),
    ],
    dtype=np.float32,
)


def _canonical_tile_id(tile_id: str) -> str:
    depth, column, row = require_tile_id(tile_id)
    if column >= 1 << depth or row >= 1 << depth:
        raise ValueError(
            f"terrain tile address is outside depth {depth}: {tile_id!r}"
        )
    canonical = f"{depth}-{column}-{row}"
    if tile_id != canonical:
        raise ValueError(f"terrain tile id is not canonical: {tile_id!r}")
    return canonical


def _terrain_png(heightmap: np.ndarray) -> tuple[bytes, float, float]:
    """Colorize one height grid and add relief without changing its values."""

    valid = np.isfinite(heightmap)
    if not np.any(valid):
        raise ValueError("stored terrain tile has no measured height samples")

    minimum = float(np.min(heightmap[valid]))
    maximum = float(np.max(heightmap[valid]))
    span = maximum - minimum
    if span == 0.0:
        normalized = np.zeros(heightmap.shape, dtype=np.float32)
    else:
        normalized = np.where(valid, (heightmap - minimum) / span, 0.0).astype(
            np.float32
        )

    color_position = normalized * (_TERRAIN_COLORS.shape[0] - 1)
    lower = np.floor(color_position).astype(np.intp)
    upper = np.minimum(lower + 1, _TERRAIN_COLORS.shape[0] - 1)
    fraction = (color_position - lower)[..., None]
    rgb = (
        _TERRAIN_COLORS[lower] * (1.0 - fraction)
        + _TERRAIN_COLORS[upper] * fraction
    )

    # A scale-independent hill shade makes the small 65x65 grid legible while
    # the color ramp continues to carry the actual elevation range.
    filled = np.where(valid, normalized, 0.0)
    gradient_y, gradient_x = np.gradient(filled)
    relief = np.clip(0.72 + (gradient_y - gradient_x) * 3.2, 0.48, 1.12)
    rgb = np.clip(rgb * relief[..., None], 0, 255).astype(np.uint8)
    near_sea_level = valid & (
        np.abs(heightmap) <= _SEA_LEVEL_TOLERANCE_METERS
    )
    rgb[near_sea_level] = _SEA_LEVEL_COLOR
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    # Stored DEM grids use row 0 as south, while image row 0 is the top
    # (north). Flip only at the rendering boundary so the persisted terrain
    # contract remains aligned with the rest of the terrain pipeline.
    rgba = np.flipud(np.dstack((rgb, alpha)))

    image = Image.fromarray(rgba).resize(
        (_PREVIEW_SIZE, _PREVIEW_SIZE),
        Image.Resampling.BILINEAR,
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), minimum, maximum


def _mask_png(
    mask: np.ndarray,
    color: np.ndarray,
    *,
    secondary: np.ndarray | None = None,
    secondary_color: np.ndarray | None = None,
) -> bytes:
    """Render south-first terrain masks without smoothing their topology."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("preview mask must be a 2D array")
    rgb = np.broadcast_to(_MASK_BACKGROUND, (*mask.shape, 3)).copy()
    rgb[mask] = color
    if secondary is not None:
        secondary = np.asarray(secondary, dtype=bool)
        if secondary.shape != mask.shape:
            raise ValueError("preview mask layers must have the same shape")
        if secondary_color is None:
            raise ValueError("secondary preview mask requires a color")
        rgb[secondary] = secondary_color
    image = Image.fromarray(np.flipud(rgb)).resize(
        (_PREVIEW_SIZE, _PREVIEW_SIZE),
        Image.Resampling.NEAREST,
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _missing_html(tile_id: str, payload_name: str) -> str:
    return f"""
<div style="box-sizing:border-box;padding:18px;border-radius:10px;
  background:#20252b;color:#e8edf2;font:14px system-ui,sans-serif">
  <strong>No stored {html.escape(payload_name)}</strong>
  <div style="margin-top:6px;color:#aeb8c2">Tile {html.escape(tile_id)}</div>
</div>
"""


async def _show_preview(widget_key: str, body: str) -> None:
    await atlantis.client_widget(
        body,
        widget_key=widget_key,
        manager_key="terrain-previews",
        manager_title="Terrain previews",
        shell="display",
    )


@visible
async def preview_tile(tile_id: str) -> None:
    """Show a colorized elevation preview of one stored DEM tile."""

    tile_id = _canonical_tile_id(tile_id)
    payload = read_dem_payload(db(), tile_id)
    if payload is None:
        await _show_preview(
            f"terrain-tile-{tile_id}",
            _missing_html(tile_id, "terrain tile"),
        )
        return

    png, minimum, maximum = _terrain_png(payload["heightmap"])
    with tempfile.NamedTemporaryFile(suffix=".png") as preview_file:
        preview_file.write(png)
        preview_file.flush()
        await atlantis.client_image(
            preview_file.name,
            image_format="image/png",
            content=(
                f"Terrain tile {tile_id} · {payload['source']} · "
                f"{payload['heightmap'].shape[1]}×{payload['heightmap'].shape[0]} · "
                f"{minimum:.1f}–{maximum:.1f} m"
            ),
            max_width=f"{_PREVIEW_SIZE}px",
        )


@visible
async def preview_texture(tile_id: str) -> None:
    """Show one stored terrain texture without fetching or rewriting it."""

    tile_id = _canonical_tile_id(tile_id)
    payload = read_texture_payload(db(), tile_id)
    if payload is None:
        await _show_preview(
            f"terrain-texture-{tile_id}",
            _missing_html(tile_id, "texture"),
        )
        return

    texture = payload["texture"]
    with Image.open(io.BytesIO(texture)) as stored_image:
        stored_width, stored_height = stored_image.size
        preview_image = stored_image.convert("RGB").resize(
            (_PREVIEW_SIZE, _PREVIEW_SIZE),
            Image.Resampling.BILINEAR,
        )

    with tempfile.NamedTemporaryFile(suffix=".jpg") as preview_file:
        preview_image.save(preview_file, format="JPEG", quality=90)
        preview_file.flush()
        await atlantis.client_image(
            preview_file.name,
            image_format="image/jpeg",
            content=(
                f"Terrain texture {tile_id} · {payload['source']} · "
                f"{stored_width}×{stored_height}"
            ),
            max_width=f"{_PREVIEW_SIZE}px",
        )


async def _show_mask_image(
    tile_id: str,
    label: str,
    source: str,
    mask: np.ndarray,
    color: np.ndarray,
    *,
    secondary: np.ndarray | None = None,
    secondary_color: np.ndarray | None = None,
) -> None:
    png = _mask_png(
        mask,
        color,
        secondary=secondary,
        secondary_color=secondary_color,
    )
    with tempfile.NamedTemporaryFile(suffix=".png") as preview_file:
        preview_file.write(png)
        preview_file.flush()
        await atlantis.client_image(
            preview_file.name,
            image_format="image/png",
            content=(
                f"{label} {tile_id} · {source} · "
                f"{mask.shape[1]}×{mask.shape[0]} · {int(mask.sum()):,} samples"
            ),
            max_width=f"{_PREVIEW_SIZE}px",
        )


@visible
async def preview_coastline(tile_id: str) -> None:
    """Show one stored authoritative GTK50 coastline mask in pink."""

    tile_id = _canonical_tile_id(tile_id)
    payload = read_coastline_mask(db(), tile_id)
    if payload is None:
        await _show_preview(
            f"terrain-coastline-{tile_id}",
            _missing_html(tile_id, "coastline mask"),
        )
        return
    await _show_mask_image(
        tile_id,
        "Coastline mask",
        payload["source"],
        payload["mask"],
        _COASTLINE_COLOR,
    )


@visible
async def preview_hydrography(tile_id: str) -> None:
    """Show one stored raw WMS hydrography mask in blue."""

    tile_id = _canonical_tile_id(tile_id)
    payload = read_hydrography_mask(db(), tile_id)
    if payload is None:
        await _show_preview(
            f"terrain-hydrography-{tile_id}",
            _missing_html(tile_id, "hydrography mask"),
        )
        return
    await _show_mask_image(
        tile_id,
        "Raw hydrography",
        payload["source"],
        payload["mask"],
        _HYDROGRAPHY_COLOR,
    )


@visible
async def preview_tidal_connectivity(tile_id: str) -> None:
    """Compare raw blue hydrography with accepted pink tidal connectivity."""

    tile_id = _canonical_tile_id(tile_id)
    connection = db()
    payload = read_hydrography_mask(connection, tile_id)
    if payload is None:
        await _show_preview(
            f"terrain-tidal-connectivity-{tile_id}",
            _missing_html(tile_id, "hydrography mask"),
        )
        return
    connected = connected_hydrography_for_tile(connection, tile_id)
    if connected is None:
        connected = np.zeros_like(payload["mask"])
    await _show_mask_image(
        tile_id,
        "Tidal connectivity (blue raw · pink accepted)",
        "derived_tidal_connectivity",
        payload["mask"],
        _HYDROGRAPHY_COLOR,
        secondary=connected,
        secondary_color=_CONNECTED_COLOR,
    )
