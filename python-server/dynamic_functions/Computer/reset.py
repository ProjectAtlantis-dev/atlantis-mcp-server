"""Reset the Computer to seed state. Run directly or as a tool."""

import atlantis
import logging
import os
from dynamic_functions.Computer.query import _connect

logger = logging.getLogger("mcp_server")

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def reset_db():
    import dynamic_functions.Computer.query as qmod
    import sqlite3

    # Bypass _connect to avoid bootstrap recursion
    conn = sqlite3.connect(qmod.DB_PATH)
    conn.row_factory = sqlite3.Row

    # nuke everything
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    for t in tables:
        conn.execute(f"DROP TABLE IF EXISTS [{t}]")
    conn.commit()

    # replay schema + seed
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.close()

    qmod._bootstrapped = True
    logger.info("Computer reset to seed state")


@visible
async def reset():
    """Wipe the Computer and reload schema + seed data from schema.sql."""
    reset_db()
    await atlantis.client_log("Computer reset. All tables rebuilt and seeded.")
