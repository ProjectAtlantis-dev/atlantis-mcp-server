"""The Computer — central sqlite database for game state facts.

Structured, queryable data lives here: visitor records, clearance flags,
location assignments, visit counts, etc.

Messy nested stuff (checklists, transcripts, debug artifacts) stays as
JSON files on disk where you can just cat them.
"""

import atlantis
import logging
import os
import sqlite3

logger = logging.getLogger("mcp_server")

DB_PATH = os.path.join(os.path.dirname(__file__), "computer.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _bootstrap()
    return conn


_bootstrapped = False

def _bootstrap():
    """Run schema.sql if the db has no tables yet."""
    global _bootstrapped
    if _bootstrapped:
        return
    conn = _connect()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    if not tables:
        logger.info("Computer: empty db, bootstrapping from schema.sql")
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())
        logger.info("Computer: bootstrap complete")
    _bootstrapped = True
    conn.close()


@visible
async def query(sql: str):
    """
    Run a SQL statement against the Computer.

    SELECT returns rows as a list of dicts.
    INSERT/UPDATE/DELETE returns the number of rows affected.
    Supports any valid sqlite SQL including CREATE TABLE, ALTER TABLE, etc.

    Args:
        sql: Any valid sqlite SQL statement
    """
    logger.info(f"Computer query: {sql}")
    conn = _connect()
    try:
        cursor = conn.execute(sql)
        if cursor.description:
            rows = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Computer returned {len(rows)} rows")
            return rows
        else:
            conn.commit()
            logger.info(f"Computer affected {cursor.rowcount} rows")
            return f"{cursor.rowcount} rows affected"
    except Exception as e:
        logger.error(f"Computer error: {e}")
        return f"Error: {e}"
    finally:
        conn.close()
