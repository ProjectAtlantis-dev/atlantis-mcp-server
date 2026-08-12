"""Offline deterministic checks for GTK50 coastline acquisition gates."""

from __future__ import annotations

import hashlib

import numpy as np
from shapely.geometry import Polygon

from dynamic_functions.Terrain.coastline import (
    CoastlineClobberError,
    _rasterize,
    _request_spec,
    read_coastline_mask,
    write_coastline_mask,
)
from dynamic_functions.Terrain.Database.database import db


@visible
def coastline_offline(tile_id: str) -> dict:
    """Verify request, orientation, persistence, idempotence, and no-clobber."""

    request = _request_spec(tile_id)
    x0, y0, x1, y1 = request["bbox"]
    mid_y = (y0 + y1) / 2.0
    # Water occupies the geographic north half; output rows are south-first,
    # so water must appear in the latter half of the returned array.
    northern_water = Polygon([(x0, mid_y), (x1, mid_y), (x1, y1), (x0, y1)])
    island = Polygon(
        [
            (x0 + (x1 - x0) * 0.4, y0 + (y1 - y0) * 0.7),
            (x0 + (x1 - x0) * 0.6, y0 + (y1 - y0) * 0.7),
            (x0 + (x1 - x0) * 0.6, y0 + (y1 - y0) * 0.9),
            (x0 + (x1 - x0) * 0.4, y0 + (y1 - y0) * 0.9),
        ]
    )
    mask = _rasterize(tuple(request["bbox"]), 65, [([northern_water], [island])])
    digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
    orientation = not bool(mask[:20].any()) and bool(mask[-20:].any())

    connection = db()
    connection.execute("SAVEPOINT coastline_offline_test")
    try:
        connection.execute("DELETE FROM coastline_masks WHERE tile_id = ?", (tile_id,))
        first = write_coastline_mask(
            connection, tile_id, mask, "fixture", 1, commit=False
        )
        duplicate = write_coastline_mask(
            connection, tile_id, mask, "fixture", 1, commit=False
        )
        before_first_read = connection.total_changes
        stored = read_coastline_mask(connection, tile_id)
        first_read_only = connection.total_changes == before_first_read
        if stored is None:
            raise AssertionError("stored coastline mask could not be read")
        exact = np.array_equal(mask, stored["mask"])
        clobber = False
        changed = mask.copy()
        changed[0, 0] = ~changed[0, 0]
        try:
            write_coastline_mask(
                connection, tile_id, changed, "fixture", 1, commit=False
            )
        except CoastlineClobberError:
            clobber = True
        before_second_read = connection.total_changes
        after = read_coastline_mask(connection, tile_id)
        second_read_only = connection.total_changes == before_second_read
        preserved = after is not None and np.array_equal(mask, after["mask"])
        return {
            "tileId": tile_id,
            "blocks": [item["blockId"] for item in request["blocks"]],
            "shape": list(mask.shape),
            "waterCount": int(mask.sum()),
            "digest": digest,
            "southFirstOrientation": orientation,
            "firstWrite": first,
            "duplicateWrite": duplicate,
            "exactRoundTrip": exact,
            "clobberBlocked": clobber,
            "existingPreserved": preserved,
            "readOnlyReads": first_read_only and second_read_only,
        }
    finally:
        connection.execute("ROLLBACK TO coastline_offline_test")
        connection.execute("RELEASE coastline_offline_test")
