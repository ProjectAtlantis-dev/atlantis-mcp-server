"""Offline endpoint regression for shared ocean-gap approval and cache validity."""

import hashlib
import io
import sqlite3

import numpy as np
from PIL import Image

from dynamic_functions.Terrain.Database import schema
from dynamic_functions.Terrain.Database.tiles import ensure_tile_row
from dynamic_functions.Terrain.coastline import write_coastline_mask
from dynamic_functions.Terrain.composition import _compose_texture
from dynamic_functions.Terrain.http_adapter import _crop_ancestor_texture, texture_response
from dynamic_functions.Terrain.ocean_texture_serving import FILL_RGB


def _jpeg(pixels):
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=100, subsampling=0)
    return output.getvalue()


def _pixels(response):
    return np.asarray(Image.open(io.BytesIO(response.body)).convert("RGB"))


@visible
def ocean_texture_serving_offline() -> dict:
    connection = sqlite3.connect(":memory:")
    schema.create(connection)
    ocean = np.ones((65, 65), dtype=bool)
    left_id, right_id = "10-463-13", "10-464-13"
    left = np.full((256, 256, 3), FILL_RGB, dtype=np.uint8)
    right = left.copy()
    # Each half has fewer than 164 core pixels. Together they form one
    # significant patch across the seam (200 samples, about 21,000 m²).
    left[100:120, -5:] = 255
    right[100:120, :5] = 255

    def store(tile_id, payload, coast=ocean):
        ensure_tile_row(connection, tile_id)
        connection.execute(
            "INSERT INTO textures (tile_id,source,texture,updated_at) VALUES (?,?,?,?)",
            (tile_id, "fixture", payload, "now"),
        )
        if coast is not None:
            write_coastline_mask(connection, tile_id, coast, "fixture", 2, commit=False)

    def response(tile_id, etag=None):
        return texture_response(connection, tile_id, schedule=lambda _: None, if_none_match=etag)

    try:
        left_bytes, right_bytes = _jpeg(left), _jpeg(right)
        store(left_id, left_bytes)
        isolated = response(left_id)
        store(right_id, right_bytes)
        connected = response(left_id, isolated.headers["etag"])
        neighbor = response(right_id)
        checks = {
            "smallIsolatedFragmentUnchanged": isolated.body == left_bytes,
            "lateNeighborInvalidatesDerivedCache": bool(
                connected.status_code == 200
                and connected.headers["etag"] != isolated.headers["etag"]
                and connected.headers["content-type"] == "image/png"
            ),
            "sharedBoundaryHasSameFill": bool(
                np.all(_pixels(connected)[100:120, -1] == FILL_RGB)
                and np.array_equal(_pixels(connected)[100:120, -1], _pixels(neighbor)[100:120, 0])
            ),
            "compositionDigestMatchesResponse": (
                connected.headers["etag"] == '"' + _compose_texture(connection, left_id)["digest"] + '"'
            ),
            "derivedEtagIsContentHash": connected.headers["etag"] == '"' + hashlib.sha256(connected.body).hexdigest() + '"',
            "derivedEtag304": response(left_id, connected.headers["etag"]).status_code == 304,
            "repairsRevalidateCache": connected.headers["cache-control"] == "no-cache",
        }

        child_id = "11-927-27"
        ancestor = response(child_id)
        child_bytes = _crop_ancestor_texture(left_bytes, child_id, left_id)
        store(child_id, child_bytes, coast=None)
        exact = response(child_id)
        checks["ancestorAndExactAtSameLodAgree"] = bool(
            ancestor.headers["x-tex-ancestor"] == left_id and ancestor.body == exact.body
        )
        checks["fineLodUsesSharedSignificance"] = int(exact.headers["x-tex-repaired-pixels"]) > 0

        raw_child = np.asarray(Image.open(io.BytesIO(child_bytes)).convert("RGB"))
        dark = raw_child.max(axis=2) < 64
        checks["unmodifiedPixelsPreservedLosslessly"] = bool(
            np.array_equal(_pixels(exact)[dark], raw_child[dark])
        )
        write_coastline_mask(connection, child_id, np.zeros_like(ocean), "fixture", 2, commit=False)
        land = response(child_id, exact.headers["etag"])
        checks["lateExactLandVetoesAncestorApproval"] = bool(
            land.status_code == 200 and land.body == child_bytes
        )
        unproven_id = "10-465-13"
        all_white = _jpeg(np.full_like(left, 255))
        store(unproven_id, all_white, coast=None)
        checks["missingCoastlinePreservesSource"] = response(unproven_id).body == all_white
        outside_id = "10-470-13"
        store(outside_id, all_white)
        checks["outsideFormerTrialRepaired"] = bool(np.all(_pixels(response(outside_id)) == FILL_RGB))
        screenshot_id = "10-443-34"
        store(screenshot_id, all_white)
        checks["screenshotLocationLargeGapRepaired"] = bool(
            np.all(_pixels(response(screenshot_id)) == FILL_RGB)
        )
        coarse_id = "8-100-20"
        store(coarse_id, all_white)
        checks["coarseGapRepaired"] = bool(np.all(_pixels(response(coarse_id)) == FILL_RGB))
        checks["missingReferenceUsesAvailableAncestor"] = bool(
            np.all(_pixels(response("11-803-163")) == FILL_RGB)
        )
        for boundary_id in ("0-0-0", "10-0-0", "10-1023-1023"):
            store(boundary_id, all_white)
            checks[f"rootBoundaryRepaired:{boundary_id}"] = bool(
                np.all(_pixels(response(boundary_id)) == FILL_RGB)
            )
        response(left_id)  # Refresh evidence after the root imagery arrives.
        before = connection.total_changes
        response(left_id)
        _compose_texture(connection, left_id)
        checks["cacheHitsAndCompositionAreReadOnly"] = connection.total_changes == before
        checks["storedSourceUnmodified"] = connection.execute(
            "SELECT texture FROM textures WHERE tile_id=?", (left_id,),
        ).fetchone()[0] == left_bytes
        if not all(checks.values()):
            raise AssertionError(checks)
        return checks
    finally:
        connection.close()
