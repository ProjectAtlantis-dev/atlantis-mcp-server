"""Composite user interface for terrain tools."""

import atlantis

from dynamic_functions.Terrain.Database.database import ux_status


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
        shell="user",
    )
