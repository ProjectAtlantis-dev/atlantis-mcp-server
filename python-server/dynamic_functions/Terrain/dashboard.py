"""Composite user interface for terrain tools."""

import atlantis

from dynamic_functions.Terrain.Database.database import ux_status


def _brushed_metal_gradient(line_count: int = 2_048) -> str:
    """Build a non-repeating field of subtly varied horizontal grey lines."""
    stops = []
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
        start = index * 100 / line_count
        end = (index + 1) * 100 / line_count
        color = f"rgb({shade}, {shade}, {shade})"
        stops.extend((f"{color} {start:.3f}%", f"{color} {end:.3f}%"))

    return f"linear-gradient(to bottom, {', '.join(stops)})"


@visible
async def dashboard() -> None:
    """Render the composite terrain dashboard."""
    components = [
        ux_status(),
    ]
    composite_html = f"""
<style>
  .terrain-dashboard-composite {{
    box-sizing: border-box;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    width: 100%;
    padding: 12px;
    background:
      linear-gradient(90deg, rgba(255, 255, 255, 0.08), transparent 28%, rgba(255, 255, 255, 0.12) 52%, transparent 76%, rgba(0, 0, 0, 0.08)),
      {_brushed_metal_gradient()};
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
