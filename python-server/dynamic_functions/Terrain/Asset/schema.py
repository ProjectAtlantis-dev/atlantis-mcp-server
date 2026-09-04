"""Schema setup for the Terrain asset catalog."""

import sqlite3


def create(connection: sqlite3.Connection) -> None:
    """Idempotently configure SQLite and create the asset schema."""
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            enabled     INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            lat         REAL NOT NULL,
            lon         REAL NOT NULL,
            heading_deg REAL NOT NULL DEFAULT 0,
            z           REAL,
            properties  TEXT NOT NULL DEFAULT '{}',
            saved_at    REAL,
            updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            cx          REAL,
            cy          REAL,
            min_x       REAL,
            min_y       REAL,
            max_x       REAL,
            max_y       REAL
        );

        CREATE INDEX IF NOT EXISTS assets_type_enabled
            ON assets(type, enabled);

        CREATE INDEX IF NOT EXISTS assets_center
            ON assets(cx, cy);

        CREATE INDEX IF NOT EXISTS assets_bounds
            ON assets(type, min_x, max_x, min_y, max_y);

        CREATE TABLE IF NOT EXISTS asset_metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    connection.commit()
