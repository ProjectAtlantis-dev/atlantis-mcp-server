"""Schema setup for the terrain heightmap SQLite database."""

import sqlite3


def create(db: sqlite3.Connection) -> None:
    """Idempotently configure SQLite and create the heightmap schema."""
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS tiles (
            tile_id          TEXT PRIMARY KEY,
            depth            INTEGER NOT NULL,
            col              INTEGER NOT NULL,
            row              INTEGER NOT NULL,
            x_min            REAL NOT NULL,
            y_min            REAL NOT NULL,
            x_max            REAL NOT NULL,
            y_max            REAL NOT NULL,
            parent_id        TEXT,
            geometric_error  REAL NOT NULL DEFAULT 0.0,
            source           TEXT NOT NULL DEFAULT 'empty',
            updated_at       TEXT NOT NULL,
            dem_demanded_at  TEXT,
            dem_requested_at TEXT,
            cog_requested_at TEXT,
            heightmap        BLOB,
            confidence_map   BLOB
        );

        CREATE INDEX IF NOT EXISTS tiles_depth_col_row
            ON tiles(depth, col, row);

        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    db.commit()
