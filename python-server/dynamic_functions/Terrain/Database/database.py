"""Connection lifecycle for the terrain heightmap SQLite database."""

import sqlite3
import uuid
from pathlib import Path

import atlantis

from dynamic_functions.Terrain.Database import schema


DATABASE_PATH = Path(__file__).with_name("terrain.db")
_CONNECTION_KEY = "Terrain.Database.connection"


def _connect() -> sqlite3.Connection:
    """Open the one process-wide connection with terrain runtime settings."""
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30.0,
        # The source service opened worker-specific connections. The Atlantis
        # port deliberately owns one server_shared connection instead, so it
        # must be eligible for the scheduler threads that will use it later.
        check_same_thread=False,
    )
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _get_connection() -> sqlite3.Connection | None:
    """Return the reload-safe terrain database connection, if started."""
    return atlantis.server_shared.get(_CONNECTION_KEY)


def db() -> sqlite3.Connection:
    """Return the terrain database connection, starting it when needed."""
    connection = _get_connection()
    if connection is None:
        connection = _connect()
        try:
            schema.create(connection)
        except Exception:
            connection.close()
            raise
        atlantis.server_shared.set(_CONNECTION_KEY, connection)
    return connection


async def _update_dashboard() -> None:
    """Re-render the Terrain dashboard after a database state change."""
    server = atlantis.get_server_instance()
    context = atlantis.get_context()
    if server is None or context is None:
        raise RuntimeError("Updating the Terrain dashboard requires an active tool call")

    await server.function_manager.function_call(
        "dashboard",
        context,
        app="Terrain",
        args={},
        setup_context=False,
    )


@visible
async def start() -> None:
    """Open the terrain database and establish its schema."""
    db()
    await atlantis.client_log(f"Terrain database started")
    await _update_dashboard()


@visible
async def stop() -> None:
    """Commit pending work and close the terrain database connection."""
    connection = _get_connection()
    if connection is not None:
        connection.commit()
        connection.close()
        atlantis.server_shared.remove(_CONNECTION_KEY)

    await atlantis.client_log(f"Terrain database stopped")
    await _update_dashboard()


@visible
def status() -> dict:
    """Report whether this process has a live, queryable database connection."""
    connection = _get_connection()
    if connection is None:
        return {
            "running": False,
            "path": str(DATABASE_PATH),
            "exists": DATABASE_PATH.exists(),
        }

    try:
        # A connection object can remain non-None after it has been closed or
        # become unusable. Executing against the schema verifies the actual
        # connection used by start(), stop(), and the terrain tools.
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        tile_count = connection.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
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


def _table_names() -> list[str]:
    """Return the application table names in the terrain database."""
    rows = db().execute(
        """
        SELECT name
        FROM sqlite_schema
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [row[0] for row in rows]


@visible
def tables() -> list[dict]:
    """Return each application table name and its number of rows."""
    connection = db()
    result = []
    for table_name in _table_names():
        quoted_name = '"' + table_name.replace('"', '""') + '"'
        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {quoted_name}"
        ).fetchone()[0]
        result.append({"table_name": table_name, "row_count": row_count})
    return result


@visible
def describe(table_name: str) -> list[dict]:
    """Return SQLite column metadata for an application table."""
    if table_name not in _table_names():
        raise ValueError(f"Unknown terrain database table: {table_name}")

    cursor = db().execute(
        """
        SELECT
            cid,
            name,
            CASE WHEN "notnull" THEN type || ' NOT NULL' ELSE type END AS type,
            dflt_value,
            pk
        FROM pragma_table_info(?)
        ORDER BY cid
        """,
        (table_name,),
    )
    field_names = [column[0] for column in cursor.description]
    return [dict(zip(field_names, row)) for row in cursor.fetchall()]


@visible
def query(sql: str) -> list[dict]:
    """Execute one SQLite statement and return its result rows as objects."""
    connection = db()
    cursor = connection.execute(sql)
    if cursor.description is None:
        connection.commit()
        return []

    field_names = [column[0] for column in cursor.description]
    return [dict(zip(field_names, row)) for row in cursor.fetchall()]


def ux_status() -> str:
    """Build the dashboard component for the terrain database status."""
    uid = uuid.uuid4().hex[:8]
    running = bool(status()["running"])
    if running:
        light_color = "#22c55e"
        light_glow = "34, 197, 94"
    else:
        light_color = "#ef4444"
        light_glow = "239, 68, 68"
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
    padding: 4.8px;
    color: #fffaf0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #terrain-db-status-{uid} .terrain-db-label {{
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
  #terrain-db-status-{uid} .terrain-db-light {{
    flex: 0 0 auto;
    width: 34px;
    height: 5px;
    background: {light_color};
    border-radius: 1px;
    box-shadow:
      0 0 5px rgba({light_glow}, 0.72),
      0 0 11px rgba({light_glow}, 0.34);
  }}
</style>
<div id="terrain-db-status-{uid}" aria-label="Terrain database status">
  <span class="terrain-db-label">TERRAIN DB</span>
  <span class="terrain-db-light" role="status" aria-label="{state_label}"></span>
</div>
"""
