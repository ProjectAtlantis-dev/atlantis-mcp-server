"""Composite user interface for terrain tools."""

import base64
import struct
import zlib
from functools import lru_cache

import atlantis

from dynamic_functions.Terrain.Database.database import ux_status


@lru_cache(maxsize=1)
def _brushed_metal_texture(line_count: int = 2_048) -> str:
    """Build a compact PNG containing non-repeating horizontal grey lines."""
    rows = bytearray()
    grain = 0x6D2B79F5
    cluster_offset = 0

    for index in range(line_count):
        # Use a deterministic long-period sequence so the grain is stable but
        # does not expose a short visual cycle.
        if index % 8 == 0:
            grain = (1_664_525 * grain + 1_013_904_223) & 0xFFFFFFFF
            cluster_offset = ((grain >> 16) % 17) - 8

        grain = (1_664_525 * grain + 1_013_904_223) & 0xFFFFFFFF
        fine_offset = ((grain >> 16) % 11) - 5
        shade = 128 + cluster_offset + fine_offset
        rows.extend((0, shade, shade, shade))

    def png_chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    png = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, line_count, 8, 2, 0, 0, 0)),
            png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)),
            png_chunk(b"IEND", b""),
        )
    )
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@visible
async def dashboard() -> None:
    """Render the composite terrain dashboard."""
    components = [
        ux_status(),
    ]
    composite_html = f"""
<style>
  .dashboard-widget-manager:has(.terrain-dashboard-composite) {{
    align-content: end !important;
  }}
  .terrain-dashboard-composite {{
    box-sizing: border-box;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    grid-auto-rows: max-content;
    align-content: end;
    gap: 12px;
    width: 100%;
    padding: 12px;
    background:
      linear-gradient(90deg, rgba(255, 255, 255, 0.08), transparent 28%, rgba(255, 255, 255, 0.12) 52%, transparent 76%, rgba(0, 0, 0, 0.08)),
      url("{_brushed_metal_texture()}");
    background-repeat: no-repeat;
    background-size: 100% 100%;
    border: 2px solid;
    border-color: #d2d2d2 #565656 #424242 #c4c4c4;
    border-radius: 8px;
    box-shadow:
      inset 2px 2px 1px rgba(255, 255, 255, 0.42),
      inset -2px -2px 1px rgba(0, 0, 0, 0.38),
      inset 0 5px 8px rgba(255, 255, 255, 0.12),
      inset 0 -6px 9px rgba(0, 0, 0, 0.2),
      0 1px 0 rgba(255, 255, 255, 0.18),
      0 4px 8px rgba(0, 0, 0, 0.48);
  }}
</style>
<div class="terrain-dashboard-composite">
  {''.join(components)}
</div>
"""

    await atlantis.client_widget(
        composite_html,
        widget_key="terrain-dashboard-content",
        manager_key="terrain-dashboard",
        manager_title="Terrain dashboard",
        shell="display",
    )
