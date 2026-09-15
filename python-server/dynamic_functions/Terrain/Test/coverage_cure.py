"""Offline checks for exact-depth terrain cure evidence."""

from __future__ import annotations

import sqlite3

from dynamic_functions.Terrain.Database import schema
from dynamic_functions.Terrain.Database.tiles import ensure_tile_row
from dynamic_functions.Terrain.coverage_cure import coverage_cure_inventory


@visible
def coverage_cure_offline() -> dict:
    """Verify curing requires DEM, current coastline, and exact texture."""

    coverage_connection = sqlite3.connect(":memory:")
    schema.create(coverage_connection)
    coverage_tiles = (
        "10-2-2", "11-4-4", "11-5-4", "11-4-5", "11-5-5", "11-6-4", "12-8-8",
    )
    for tile_id in coverage_tiles:
        ensure_tile_row(coverage_connection, tile_id)
    coverage_connection.execute(
        "UPDATE tiles SET heightmap = ? WHERE tile_id IN (?,?,?,?,?,?)",
        (b"dem", "10-2-2", "11-4-4", "11-5-4", "11-5-5", "11-6-4", "12-8-8"),
    )
    coverage_connection.executemany(
        "INSERT INTO coastline_masks "
        "(tile_id, width, height, mask, source, version, updated_at) "
        "VALUES (?, 1, 1, ?, ?, ?, 'now')",
        (
            ("10-2-2", b"mask", "gtk50_vector", 2),
            ("11-4-4", b"mask", "gtk50_vector", 2),
            ("11-5-4", b"mask", "gtk50_vector", 1),
            ("11-4-5", b"mask", "gtk50_vector", 2),
            ("11-5-5", b"mask", "gtk50_vector", 2),
            ("11-6-4", b"mask", "gtk50_vector", 2),
            ("12-8-8", b"mask", "gtk50_vector", 2),
        ),
    )
    coverage_connection.executemany(
        "INSERT INTO textures (tile_id, texture, source, updated_at) "
        "VALUES (?, ?, 'dataforsyningen', 'now')",
        (
            ("10-2-2", b"ancestor-texture"),
            ("11-4-4", b"exact-texture"),
            ("11-5-4", b"exact-texture"),
            ("11-4-5", b"exact-texture"),
            ("11-6-4", b""),
        ),
    )
    coverage = coverage_cure_inventory(coverage_connection)
    coverage_connection.close()

    return {
        "coverageCure": bool(
            coverage["cureDepth"] == 11
            and coverage["cureVersion"] == 2
            and coverage["definition"]
            == "depth-11 DEM, current authoritative coastline mask, "
            "and exact depth-11 texture (no ancestor imagery)"
            and coverage["summary"]
            == {"cured": 1, "partial": 4, "coarse": 1}
            and {
                tile["tile"]: tile["status"] for tile in coverage["tiles"]
            }
            == {
                "10-2-2": "partial",
                "11-4-4": "cured",
                "11-4-5": "partial",
                "11-5-4": "partial",
                "11-5-5": "partial",
                "11-6-4": "partial",
            }
        ),
        "ancestorTextureDoesNotCure": any(
            tile["tile"] == "11-5-5"
            and tile["dem"] and tile["coastline"]
            and tile["texture"] is None and tile["status"] == "partial"
            for tile in coverage["tiles"]
        ),
    }
