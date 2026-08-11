"""Read-only visual previews for persisted terrain tiles and textures."""

from __future__ import annotations

import base64
import hashlib
import html
import io

import atlantis
import numpy as np
from PIL import Image

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.textures import read_texture_payload
from dynamic_functions.Terrain.Database.tiles import read_dem_payload
from dynamic_functions.Terrain.tile_address import require_tile_id


_PREVIEW_SIZE = 520
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
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.dstack((rgb, alpha))

    image = Image.fromarray(rgba).resize(
        (_PREVIEW_SIZE, _PREVIEW_SIZE),
        Image.Resampling.BILINEAR,
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), minimum, maximum


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
async def preview_tile(tile_id: str) -> dict:
    """Show a colorized elevation preview of one stored DEM tile."""

    tile_id = _canonical_tile_id(tile_id)
    payload = read_dem_payload(db(), tile_id)
    if payload is None:
        await _show_preview(
            f"terrain-tile-{tile_id}",
            _missing_html(tile_id, "terrain tile"),
        )
        return {"tileId": tile_id, "found": False}

    png, minimum, maximum = _terrain_png(payload["heightmap"])
    encoded = base64.b64encode(png).decode("ascii")
    escaped_id = html.escape(tile_id)
    escaped_source = html.escape(str(payload["source"]))
    body = f"""
<div style="box-sizing:border-box;padding:12px;border-radius:10px;
  background:#171b20;color:#edf2f5;font:13px system-ui,sans-serif">
  <div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:9px">
    <strong>Terrain tile {escaped_id}</strong>
    <span style="color:#aeb8c2">{escaped_source}</span>
  </div>
  <div style="overflow:hidden;border:1px solid #46515c;border-radius:6px;
    background-color:#303840;background-image:linear-gradient(45deg,#3b444d 25%,transparent 25%),
    linear-gradient(-45deg,#3b444d 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#3b444d 75%),
    linear-gradient(-45deg,transparent 75%,#3b444d 75%);background-size:20px 20px;
    background-position:0 0,0 10px,10px -10px,-10px 0">
    <img alt="Elevation preview for tile {escaped_id}"
      src="data:image/png;base64,{encoded}"
      style="display:block;width:100%;max-width:{_PREVIEW_SIZE}px;aspect-ratio:1" />
  </div>
  <div style="height:8px;margin-top:10px;border-radius:4px;
    background:linear-gradient(90deg,rgb(31,78,82),rgb(68,112,88),rgb(142,139,91),
    rgb(139,125,111),rgb(225,232,235),white)"></div>
  <div style="display:flex;justify-content:space-between;margin-top:4px;color:#aeb8c2">
    <span>{minimum:.1f} m</span><span>{maximum:.1f} m</span>
  </div>
</div>
"""
    await _show_preview(
        f"terrain-tile-{tile_id}",
        body,
    )
    return {
        "tileId": tile_id,
        "found": True,
        "source": payload["source"],
        "updatedAt": payload["updated_at"],
        "minimum": minimum,
        "maximum": maximum,
        "mediaType": "image/png",
        "contentLength": len(png),
        "digest": hashlib.sha256(png).hexdigest(),
    }


@visible
async def preview_texture(tile_id: str) -> dict:
    """Show one stored terrain texture without fetching or rewriting it."""

    tile_id = _canonical_tile_id(tile_id)
    payload = read_texture_payload(db(), tile_id)
    if payload is None:
        await _show_preview(
            f"terrain-texture-{tile_id}",
            _missing_html(tile_id, "texture"),
        )
        return {"tileId": tile_id, "found": False}

    texture = payload["texture"]
    encoded = base64.b64encode(texture).decode("ascii")
    escaped_id = html.escape(tile_id)
    escaped_source = html.escape(str(payload["source"]))
    body = f"""
<div style="box-sizing:border-box;padding:12px;border-radius:10px;
  background:#171b20;color:#edf2f5;font:13px system-ui,sans-serif">
  <div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:9px">
    <strong>Terrain texture {escaped_id}</strong>
    <span style="color:#aeb8c2">{escaped_source}</span>
  </div>
  <img alt="Texture preview for tile {escaped_id}"
    src="data:image/jpeg;base64,{encoded}"
    style="display:block;width:100%;max-width:{_PREVIEW_SIZE}px;aspect-ratio:1;
      object-fit:contain;border:1px solid #46515c;border-radius:6px;background:#303840" />
</div>
"""
    await _show_preview(
        f"terrain-texture-{tile_id}",
        body,
    )
    return {
        "tileId": tile_id,
        "found": True,
        "source": payload["source"],
        "updatedAt": payload["updated_at"],
        "mediaType": "image/jpeg",
        "contentLength": len(texture),
        "digest": hashlib.sha256(texture).hexdigest(),
    }
