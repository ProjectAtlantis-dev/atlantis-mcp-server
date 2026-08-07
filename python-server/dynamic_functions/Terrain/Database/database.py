"""Connection lifecycle for the terrain heightmap SQLite database."""

import sqlite3
import uuid
from pathlib import Path

import atlantis

from dynamic_functions.Terrain.Database import schema


DATABASE_PATH = Path(__file__).with_name("terrain.db")

_connection: sqlite3.Connection | None = None


@visible
async def start() -> None:
    """Open the terrain database and establish its schema."""
    global _connection

    if _connection is None:
        _connection = sqlite3.connect(DATABASE_PATH)
        try:
            schema.create(_connection)
        except Exception:
            _connection.close()
            _connection = None
            raise

    await atlantis.client_log(f"Terrain database started at {DATABASE_PATH}")


@visible
async def stop() -> None:
    """Commit pending work and close the terrain database connection."""
    global _connection

    if _connection is not None:
        _connection.commit()
        _connection.close()
        _connection = None

    await atlantis.client_log(f"Terrain database stopped at {DATABASE_PATH}")


@visible
def status() -> dict:
    """Report whether this process has a live, queryable database connection."""
    if _connection is None:
        return {
            "running": False,
            "path": str(DATABASE_PATH),
            "exists": DATABASE_PATH.exists(),
        }

    try:
        # A connection object can remain non-None after it has been closed or
        # become unusable. Executing against the schema verifies the actual
        # connection used by start(), stop(), and the terrain tools.
        journal_mode = _connection.execute("PRAGMA journal_mode").fetchone()[0]
        tile_count = _connection.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    except sqlite3.Error:
        return {
            "running": False,
            "path": str(DATABASE_PATH),
            "exists": DATABASE_PATH.exists(),
        }

    return {
        "running": True,
        "path": str(DATABASE_PATH),
        "exists": True,
        "journal_mode": journal_mode,
        "tile_count": tile_count,
    }


def ux_status() -> str:
    """Build the dashboard component for the terrain database status."""
    uid = uuid.uuid4().hex[:8]
    running = bool(status()["running"])
    light_color = "#22c55e" if running else "#ef4444"
    state_label = "on" if running else "off"

    return f"""
<style>
  #terrain-db-status-{uid} {{
    box-sizing: border-box;
    display: flex;
    gap: 14px;
    align-items: center;
    justify-content: center;
    width: 100%;
    padding: 24px;
    color: #fffaf0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #terrain-db-status-{uid} .terrain-db-label {{
    margin: 0;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.08em;
  }}
  #terrain-db-status-{uid} .terrain-db-light {{
    flex: 0 0 auto;
    width: 16px;
    height: 16px;
    background: {light_color};
    border-radius: 50%;
    box-shadow: 0 0 12px {light_color};
  }}
</style>
<section id="terrain-db-status-{uid}" aria-label="Terrain database status">
  <span class="terrain-db-label">TERRAIN DB</span>
  <span class="terrain-db-light" role="status" aria-label="{state_label}"></span>
</section>
"""
