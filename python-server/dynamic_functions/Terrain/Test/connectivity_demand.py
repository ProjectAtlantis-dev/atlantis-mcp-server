"""Offline regression for immutable background connectivity publication."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dynamic_functions.Terrain.demand import _connectivity_worker
from dynamic_functions.Terrain.tidal_connectivity import (
    read_connectivity_snapshot,
    write_connectivity_snapshot,
)


@visible
def connectivity_demand_offline() -> dict:
    """Prove a newer generation skips snapshots an older worker published."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "connectivity.db"
        connection = sqlite3.connect(database_path)
        connection.execute(
            "CREATE TABLE tidal_connectivity_masks ("
            "tile_id TEXT PRIMARY KEY, width INTEGER NOT NULL, "
            "height INTEGER NOT NULL, mask BLOB NOT NULL, "
            "source TEXT NOT NULL, version INTEGER NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        original = np.zeros((3, 3), dtype=bool)
        replacement = np.ones((3, 3), dtype=bool)
        new_mask = np.eye(3, dtype=bool)
        write_connectivity_snapshot(
            connection, "11-683-374", original
        )
        connection.close()

        with (
            patch(
                "dynamic_functions.Terrain.demand.DATABASE_PATH",
                database_path,
            ),
            patch(
                "dynamic_functions.Terrain.demand._build_connected_hydrography",
                return_value={
                    "11-683-374": replacement,
                    "11-684-374": new_mask,
                },
            ),
        ):
            result = _connectivity_worker("11:generation")

        connection = sqlite3.connect(database_path)
        try:
            preserved = read_connectivity_snapshot(
                connection, "11-683-374"
            )
            published = read_connectivity_snapshot(
                connection, "11-684-374"
            )
        finally:
            connection.close()
        return {
            "existingSnapshotPreserved": bool(
                preserved is not None
                and np.array_equal(preserved["mask"], original)
            ),
            "missingSnapshotPublished": bool(
                published is not None
                and np.array_equal(published["mask"], new_mask)
            ),
            "boundedPublicationCounts": result
            == {
                "depth": 11,
                "derived": 2,
                "published": 1,
                "alreadyReady": 1,
            },
        }
