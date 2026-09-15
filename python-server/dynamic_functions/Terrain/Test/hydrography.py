"""Offline deterministic checks for WMS hydrography acquisition."""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.hydrography import (
    GRID_N,
    OVERSAMPLE,
    SOURCE,
    VERSION,
    HydrographyClobberError,
    _acquire_mask,
    _decode_mask,
    _request_spec,
    read_hydrography_mask,
    write_hydrography_mask,
)


_FIXTURE_MASK_DIGEST = (
    "5d7a997b647903c2440c45bcae97745c2a452d2c389391ae541cb864b254b658"
)


def _fixture() -> bytes:
    """Build a north-first PNG with two water regions and one land hole."""

    size = GRID_N * OVERSAMPLE
    pixels = np.full((size, size, 3), 255, dtype=np.uint8)
    water_rgb = np.asarray([100, 160, 190], dtype=np.uint8)
    # The northern 32 terrain rows are water.
    pixels[: 32 * OVERSAMPLE, :, :] = water_rgb
    # One complete land sample within that water body.
    pixels[10 * OVERSAMPLE : 11 * OVERSAMPLE, 20 * OVERSAMPLE : 21 * OVERSAMPLE] = 255
    # One disconnected water sample in the southern half.
    pixels[50 * OVERSAMPLE : 51 * OVERSAMPLE, 40 * OVERSAMPLE : 41 * OVERSAMPLE] = water_rgb
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


@visible
def hydrography_offline(tile_id: str) -> dict:
    """Verify WMS request, decode, persistence, and failure isolation gates."""

    connection = db()
    before_request = connection.total_changes
    request = _request_spec(tile_id)
    request_read_only = connection.total_changes == before_request

    fixture = _fixture()
    mask = _decode_mask(fixture)
    digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
    if digest != _FIXTURE_MASK_DIGEST:
        raise AssertionError(f"hydrography fixture digest changed: {digest}")
    south_first = (
        not bool(mask[:14].any())
        and bool(mask[14, 40])
        and not bool(mask[54, 20])
        and bool(mask[-1].all())
    )

    fetched_urls: list[str] = []

    def fixture_fetcher(url: str):
        fetched_urls.append(url)
        return fixture, {"httpStatus": 200, "contentType": "image/png"}

    acquired, acquisition = _acquire_mask(tile_id, fetcher=fixture_fetcher)
    acquisition_exact = np.array_equal(mask, acquired)
    corrupt_rejected = False
    try:
        _decode_mask(b"not an image")
    except ValueError:
        corrupt_rejected = True
    wrong_size_rejected = False
    wrong = io.BytesIO()
    Image.new("RGB", (8, 8), (100, 160, 190)).save(wrong, format="PNG")
    try:
        _decode_mask(wrong.getvalue())
    except ValueError:
        wrong_size_rejected = True

    connection.execute("SAVEPOINT hydrography_offline_test")
    try:
        connection.execute(
            "DELETE FROM hydrography_masks WHERE tile_id = ?", (tile_id,)
        )
        first = write_hydrography_mask(
            connection, tile_id, mask, SOURCE, VERSION, commit=False
        )
        duplicate = write_hydrography_mask(
            connection, tile_id, mask, SOURCE, VERSION, commit=False
        )
        before_read = connection.total_changes
        stored = read_hydrography_mask(connection, tile_id)
        read_only = connection.total_changes == before_read
        exact = stored is not None and np.array_equal(mask, stored["mask"])

        changed = mask.copy()
        changed[0, 0] = ~changed[0, 0]
        clobber = False
        try:
            write_hydrography_mask(
                connection, tile_id, changed, SOURCE, VERSION, commit=False
            )
        except HydrographyClobberError:
            clobber = True
        after = read_hydrography_mask(connection, tile_id)
        preserved = after is not None and np.array_equal(mask, after["mask"])

        provider_failure = False

        def failed_fetcher(_url: str):
            raise TimeoutError("fixture timeout")

        try:
            _acquire_mask(tile_id, fetcher=failed_fetcher)
        except TimeoutError:
            provider_failure = True
        failure_preserved = np.array_equal(
            mask, read_hydrography_mask(connection, tile_id)["mask"]
        )
        return {
            "tileId": tile_id,
            "endpoint": request["endpoint"],
            "layer": request["layer"],
            "bbox": request["bbox"],
            "requestSize": [request["height"], request["width"]],
            "shape": list(mask.shape),
            "waterCount": int(mask.sum()),
            "digest": digest,
            "southFirstOrientation": south_first,
            "requestReadOnly": request_read_only,
            "fetchCalledOnce": len(fetched_urls) == 1,
            "acquisitionExact": acquisition_exact,
            "acquisitionStatus": acquisition["status"],
            "corruptRejected": corrupt_rejected,
            "wrongSizeRejected": wrong_size_rejected,
            "firstWrite": first,
            "duplicateWrite": duplicate,
            "exactRoundTrip": exact,
            "clobberBlocked": clobber,
            "existingPreserved": preserved,
            "readOnlyRead": read_only,
            "providerFailureRaised": provider_failure,
            "providerFailurePreserved": failure_preserved,
        }
    finally:
        connection.execute("ROLLBACK TO hydrography_offline_test")
        connection.execute("RELEASE hydrography_offline_test")
