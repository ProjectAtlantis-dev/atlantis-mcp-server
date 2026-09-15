"""Deterministic, rollback-only checks for explicit ancestor fallback."""

import datetime

import numpy as np

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.textures import (
    read_texture_with_ancestor,
)
from dynamic_functions.Terrain.Database.tiles import (
    read_dem_with_ancestor,
    write_dem,
)
from dynamic_functions.Terrain.tile_address import ancestor_tile_ids


_PARENT_ID = "13-8191-8191"
_CHILD_ID = "15-32767-32767"
_EXACT_ID = "15-32766-32766"
_MISSING_ID = "15-0-0"


def _delete_fixture_rows(connection) -> None:
    fixture_ids = {
        _PARENT_ID,
        _CHILD_ID,
        _EXACT_ID,
        *ancestor_tile_ids(_MISSING_ID, include_self=True),
    }
    marks = ",".join("?" for _ in fixture_ids)
    connection.execute(
        f"DELETE FROM textures WHERE tile_id IN ({marks})",
        tuple(fixture_ids),
    )
    connection.execute(
        f"DELETE FROM tiles WHERE tile_id IN ({marks})",
        tuple(fixture_ids),
    )


@visible
def parent_fallback() -> dict:
    """Prove nearest-ancestor, exact-precedence, miss, and read-only behavior."""

    connection = db()
    connection.execute("SAVEPOINT parent_fallback_test")
    try:
        _delete_fixture_rows(connection)
        before_changes = connection.total_changes
        parent_heightmap = np.arange(65 * 65, dtype=np.float32).reshape(65, 65)
        exact_heightmap = parent_heightmap + 10000.0
        write_dem(
            connection,
            _PARENT_ID,
            parent_heightmap,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )
        write_dem(
            connection,
            _EXACT_ID,
            exact_heightmap,
            "arcticdem_10m",
            "EGM2008",
            commit=False,
        )

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        connection.executemany(
            "INSERT INTO textures (tile_id, source, texture, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (
                (_PARENT_ID, "fixture", b"parent-texture", now),
                (_EXACT_ID, "fixture", b"exact-texture", now),
            ),
        )
        after_setup = connection.total_changes

        dem_parent = read_dem_with_ancestor(connection, _CHILD_ID)
        dem_exact = read_dem_with_ancestor(connection, _EXACT_ID)
        dem_missing = read_dem_with_ancestor(connection, _MISSING_ID)
        texture_parent = read_texture_with_ancestor(connection, _CHILD_ID)
        texture_exact = read_texture_with_ancestor(connection, _EXACT_ID)
        texture_missing = read_texture_with_ancestor(connection, _MISSING_ID)

        return {
            "demNearestAncestor": (
                dem_parent is not None
                and dem_parent["resolved_tile_id"] == _PARENT_ID
                and dem_parent["depth_delta"] == 2
                and not dem_parent["exact"]
            ),
            "demExactPrecedence": (
                dem_exact is not None
                and dem_exact["resolved_tile_id"] == _EXACT_ID
                and dem_exact["depth_delta"] == 0
                and dem_exact["exact"]
            ),
            "demMiss": dem_missing is None,
            "textureNearestAncestor": (
                texture_parent is not None
                and texture_parent["resolved_tile_id"] == _PARENT_ID
                and texture_parent["depth_delta"] == 2
                and not texture_parent["exact"]
                and texture_parent["texture"] == b"parent-texture"
            ),
            "textureExactPrecedence": (
                texture_exact is not None
                and texture_exact["resolved_tile_id"] == _EXACT_ID
                and texture_exact["depth_delta"] == 0
                and texture_exact["exact"]
                and texture_exact["texture"] == b"exact-texture"
            ),
            "textureMiss": texture_missing is None,
            "readsChangedNoRows": connection.total_changes == after_setup,
            "fixtureRowsWritten": after_setup - before_changes,
        }
    finally:
        connection.execute("ROLLBACK TO parent_fallback_test")
        connection.execute("RELEASE parent_fallback_test")
