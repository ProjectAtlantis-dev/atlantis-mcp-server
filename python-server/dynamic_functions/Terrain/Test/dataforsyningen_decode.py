"""Directly callable checks for Dataforsyningen metatile splitting."""

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from dynamic_functions.Terrain.dataforsyningen import (
    _decode_metatile,
    _no_coverage_kind,
    _request_spec,
    _split_metatile,
)


_FIXTURE_PATH = (
    Path(__file__).with_name("fixtures") / "dataforsyningen_metatile.png"
)
_FIXTURE_DIGEST = "e9cd577974627e389660c038b7b8379faace6a00ee7960437dac1c6908c4c250"
_COMBINED_CHILD_DIGEST = (
    "c576c13ebfd8a1151652c2edd8393e483c1d1033a04fbe245a7142dc387da39f"
)


def _expected_color(column: int, row: int) -> tuple[int, int, int]:
    return 30 + column * 50, 40 + row * 45, 80 + (column * 4 + row) * 7


@visible
def dataforsyningen_decode(tile_id: str) -> dict:
    """Decode and split the bundled 4-by-4 orientation fixture."""

    fixture = _FIXTURE_PATH.read_bytes()
    request = _request_spec(tile_id)
    image = _decode_metatile(fixture, request["width"])
    split = _split_metatile(fixture, tile_id)
    children = []
    combined_digest = hashlib.sha256()

    for child in request["children"]:
        encoded = split[child["tileId"]]
        decoded = Image.open(io.BytesIO(encoded)).convert("RGB")
        pixels = np.asarray(decoded, dtype=np.uint8)
        center = tuple(int(value) for value in pixels[128, 128])
        expected = _expected_color(
            child["columnOffset"],
            child["rowOffset"],
        )
        if decoded.size != (256, 256):
            raise AssertionError(
                f"{child['tileId']} decoded at unexpected size {decoded.size}"
            )
        if any(abs(actual - wanted) > 3 for actual, wanted in zip(center, expected)):
            raise AssertionError(
                f"{child['tileId']} orientation mismatch: "
                f"center={center}, expected={expected}"
            )
        pixel_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        combined_digest.update(child["tileId"].encode("ascii"))
        combined_digest.update(bytes.fromhex(pixel_digest))
        children.append(
            {
                **child,
                "width": decoded.width,
                "height": decoded.height,
                "centerRgb": list(center),
                "pixelDigest": pixel_digest,
                "noCoverage": _no_coverage_kind(decoded),
            }
        )

    white = Image.new("RGB", image.size, (255, 255, 255))
    white_buffer = io.BytesIO()
    white.save(white_buffer, format="PNG")
    white_status = _no_coverage_kind(
        _decode_metatile(white_buffer.getvalue(), request["width"])
    )

    corrupt_status = None
    try:
        _decode_metatile(b"not an image", request["width"])
    except ValueError:
        corrupt_status = "corrupt"

    fixture_digest = hashlib.sha256(fixture).hexdigest()
    child_digest = combined_digest.hexdigest()
    if fixture_digest != _FIXTURE_DIGEST:
        raise AssertionError(f"fixture digest changed: {fixture_digest}")
    if child_digest != _COMBINED_CHILD_DIGEST:
        raise AssertionError(f"split child digest changed: {child_digest}")
    if white_status != "white_fill":
        raise AssertionError(f"white fixture classified as {white_status}")
    if corrupt_status != "corrupt":
        raise AssertionError(f"corrupt fixture classified as {corrupt_status}")

    return {
        "tileId": tile_id,
        "metatileId": request["metatileId"],
        "fixtureStatus": _no_coverage_kind(image) or "valid",
        "fixtureDigest": fixture_digest,
        "width": image.width,
        "height": image.height,
        "childCount": len(children),
        "combinedChildDigest": child_digest,
        "whiteStatus": white_status,
        "corruptStatus": corrupt_status,
        "children": children,
    }
