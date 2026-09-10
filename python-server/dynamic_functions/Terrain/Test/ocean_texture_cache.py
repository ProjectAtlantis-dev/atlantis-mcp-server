"""Disk-cache reuse, invalidation, and transaction ownership regression."""

import io
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from dynamic_functions.Terrain.Database import schema
from dynamic_functions.Terrain.Database.tiles import ensure_tile_row
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain import ocean_texture_serving as serving


@visible
def ocean_texture_cache_offline() -> dict:
    tile_id = "10-443-34"
    image = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), 255, dtype=np.uint8)).save(image, "JPEG")
    payload = image.getvalue()
    checks = {}
    with tempfile.TemporaryDirectory(prefix="ocean-cache-") as directory:
        path = Path(directory) / "textures.db"
        connection = sqlite3.connect(path)
        schema.create(connection)
        ensure_tile_row(connection, tile_id)
        connection.execute("INSERT INTO textures (tile_id,source,texture,updated_at,acquisition_dates) VALUES (?,?,?,?,?)",
                           (tile_id, "fixture", payload, "now", None))
        ocean = np.ones((65, 65), dtype=bool)
        write_coastline_mask(connection, tile_id, ocean, "fixture", 2, commit=False)
        connection.commit()
        first = serving.repair_texture(connection, tile_id, payload, persist=True)
        checks["repairPersisted"] = first[2] == 256 * 256 and not connection.in_transaction
        connection.close()
        serving._classify.cache_clear()
        serving._render.cache_clear()
        connection = sqlite3.connect(path)
        with patch.object(serving, "_classify", side_effect=AssertionError("reclassified")), \
             patch.object(serving, "_render", side_effect=AssertionError("rerendered")):
            checks["reopenReusesDiskWithoutRepairWork"] = (
                serving.repair_texture(connection, tile_id, payload, persist=True) == first
            )
        original_key = connection.execute("SELECT evidence_digest FROM ocean_texture_repairs").fetchone()[0]
        with patch.object(serving, "VERSION", "cache-test-new-version"):
            serving.repair_texture(connection, tile_id, payload, persist=True)
        checks["versionInvalidates"] = original_key != connection.execute(
            "SELECT evidence_digest FROM ocean_texture_repairs").fetchone()[0]
        connection.execute("DELETE FROM coastline_masks WHERE tile_id=?", (tile_id,))
        write_coastline_mask(connection, tile_id, np.zeros_like(ocean), "fixture", 3, commit=False)
        result = serving.repair_texture(connection, tile_id, payload, persist=True)
        checks["coastlineInvalidates"] = result == (payload, "image/jpeg", 0)
        checks["callerTransactionPreserved"] = connection.in_transaction
        checks["negativeResultAvoidsSourceDuplication"] = connection.execute(
            "SELECT texture FROM ocean_texture_repairs").fetchone()[0] is None
        connection.rollback()
        checks["rollbackRestoresRepair"] = serving.repair_texture(connection, tile_id, payload)[2] > 0
        changed = io.BytesIO()
        Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(changed, "JPEG")
        dark = changed.getvalue()
        checks["requestedPayloadInvalidates"] = serving.repair_texture(
            connection, tile_id, dark, persist=True) == (dark, "image/jpeg", 0)
        checks["oneRevisionPerTile"] = connection.execute(
            "SELECT COUNT(*) FROM ocean_texture_repairs").fetchone()[0] == 1
        checks["sourcePreserved"] = connection.execute(
            "SELECT texture FROM textures WHERE tile_id=?", (tile_id,)).fetchone()[0] == payload
        before = connection.total_changes
        serving.repair_texture(connection, tile_id, payload)
        checks["compositionMissDoesNotWrite"] = connection.total_changes == before
        connection.close()
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks
