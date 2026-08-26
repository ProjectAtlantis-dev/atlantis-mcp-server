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
            vertical_datum   TEXT,
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

        CREATE TABLE IF NOT EXISTS textures (
            tile_id    TEXT PRIMARY KEY,
            source     TEXT NOT NULL,
            texture    BLOB NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS coastline_masks (
            tile_id    TEXT PRIMARY KEY,
            width      INTEGER NOT NULL CHECK (width > 0),
            height     INTEGER NOT NULL CHECK (height > 0),
            mask       BLOB NOT NULL,
            source     TEXT NOT NULL,
            version    INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (tile_id) REFERENCES tiles(tile_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS hydrography_masks (
            tile_id    TEXT PRIMARY KEY,
            width      INTEGER NOT NULL CHECK (width > 0),
            height     INTEGER NOT NULL CHECK (height > 0),
            mask       BLOB NOT NULL,
            source     TEXT NOT NULL,
            version    INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (tile_id) REFERENCES tiles(tile_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tidal_connectivity_masks (
            tile_id    TEXT PRIMARY KEY,
            width      INTEGER NOT NULL CHECK (width > 0),
            height     INTEGER NOT NULL CHECK (height > 0),
            mask       BLOB NOT NULL,
            source     TEXT NOT NULL,
            version    INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (tile_id) REFERENCES tiles(tile_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS bathymetry (
            tile_id    TEXT PRIMARY KEY,
            heightmap  BLOB NOT NULL,
            water_px   INTEGER NOT NULL,
            min_z      REAL NOT NULL,
            max_z      REAL NOT NULL,
            source     TEXT NOT NULL,
            version    INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (tile_id) REFERENCES tiles(tile_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS bathymetry_source
            ON bathymetry(source, version);
        """
    )
    tile_columns = {
        row[1] for row in db.execute("PRAGMA table_info(tiles)").fetchall()
    }
    if "vertical_datum" not in tile_columns:
        db.execute("ALTER TABLE tiles ADD COLUMN vertical_datum TEXT")
    db.commit()
