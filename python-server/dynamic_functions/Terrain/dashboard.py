"""Composite user interface for terrain tools."""

import base64
import struct
import uuid
import zlib
from functools import lru_cache

import atlantis

from dynamic_functions.Terrain.Database.database import ux_status
from dynamic_functions.Terrain.viewer_server import server_status


def _server_status_bar() -> str:
    """Build the dashboard component for the Terrain viewer server status."""
    uid = uuid.uuid4().hex[:8]
    running = bool(server_status()["running"])
    if running:
        light_color = "#22c55e"
        light_glow = "34, 197, 94"
    else:
        light_color = "#ef4444"
        light_glow = "239, 68, 68"
    state_label = "on" if running else "off"

    return f"""
<style>
  #terrain-server-status-{uid} {{
    box-sizing: border-box;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 14px;
    align-items: center;
    width: 100%;
    padding: 4.8px;
    color: #fffaf0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #terrain-server-status-{uid} .terrain-server-label {{
    justify-self: start;
    margin: 0;
    color: rgba(42, 42, 42, 0.92);
    font-family: "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
    font-size: 17px;
    font-stretch: condensed;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-shadow:
      0 -1px 0 rgba(0, 0, 0, 0.72),
      0 1px 0 rgba(255, 255, 255, 0.52);
  }}
  #terrain-server-status-{uid} .terrain-server-light {{
    justify-self: end;
    width: 34px;
    height: 5px;
    background: {light_color};
    border-radius: 1px;
    box-shadow:
      0 0 5px rgba({light_glow}, 0.72),
      0 0 11px rgba({light_glow}, 0.34);
  }}
</style>
<div id="terrain-server-status-{uid}" aria-label="Terrain viewer server status">
  <span class="terrain-server-label">TERRAIN SERVER</span>
  <span class="terrain-server-light" role="status" aria-label="{state_label}"></span>
</div>
"""


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
        f"""
        <div class="terrain-dashboard-status-stack">
          {_server_status_bar()}
          {ux_status()}
        </div>
        """,
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
  .terrain-dashboard-status-stack {{
    box-sizing: border-box;
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    grid-auto-rows: max-content;
    gap: 3px;
    width: 100%;
    padding: 2px;
    background: rgba(30, 30, 30, 0.5);
    border: 1px solid;
    border-color: rgba(52, 52, 52, 0.92) rgba(220, 220, 220, 0.42) rgba(225, 225, 225, 0.48) rgba(48, 48, 48, 0.9);
    border-radius: 4px;
    box-shadow:
      inset 0 1px 2px rgba(0, 0, 0, 0.72),
      0 1px 0 rgba(255, 255, 255, 0.24);
  }}
  .terrain-dashboard-status-stack > div[aria-label] {{
    background:
      linear-gradient(90deg, rgba(255, 255, 255, 0.12), transparent 36%, rgba(255, 255, 255, 0.08) 68%, rgba(0, 0, 0, 0.08)),
      rgba(148, 148, 148, 0.82);
    border: 1px solid;
    border-color: rgba(222, 222, 222, 0.8) rgba(67, 67, 67, 0.92) rgba(55, 55, 55, 0.94) rgba(210, 210, 210, 0.72);
    border-radius: 2px;
    box-shadow:
      inset 1px 1px 0 rgba(255, 255, 255, 0.26),
      inset -1px -1px 0 rgba(0, 0, 0, 0.3),
      0 1px 2px rgba(0, 0, 0, 0.5);
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
