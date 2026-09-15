"""Offline real-world and negative controls for ocean texture gap detection."""

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from dynamic_functions.Terrain.ocean_texture import (
    detect_white_ocean_gaps,
    texture_ocean_mask,
    texture_pixel_area_m2,
    white_ocean_repair_mask,
)


FIXTURE = Path(__file__).with_name("fixtures") / "ocean_white_wedge_11-934-28.jpg"
FIXTURE_DIGEST = "4c288129d624a5a503ae84358b6b65ad150a023a469e3f35754f7df788ff090c"
BLOCK_FIXTURE = FIXTURE.with_name("ocean_white_block_9-230-6.jpg")
BLOCK_DIGEST = "28de5bb6fb1f99c8da1597f67bfb83a71fad080c4872e5336562a6cd5eb0a202"
ICE_FIXTURE = FIXTURE.with_name("ocean_ice_12-1858-52.jpg")
ICE_DIGEST = "2d353423ba508bb96ba9944f6fc255cded69bbe84ca6eff1edf90ff25911d05c"


@visible
def ocean_texture_offline() -> dict:
    """Detect the diagonal wedge while protecting land and nonuniform imagery."""

    payload = FIXTURE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FIXTURE_DIGEST
    pixels = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    original = pixels.copy()
    # Saved GTK50 row for this tile has all 4,225 vertices classified as sea.
    ocean = texture_ocean_mask(np.ones((65, 65), dtype=bool), (256, 256))
    pixel_area = texture_pixel_area_m2("11-934-28", (256, 256))

    def detect(values, sea=ocean, area=pixel_area):
        return detect_white_ocean_gaps(values, sea, pixel_area_m2=area)

    detected = detect(pixels)
    block_payload = BLOCK_FIXTURE.read_bytes()
    assert hashlib.sha256(block_payload).hexdigest() == BLOCK_DIGEST
    block = np.asarray(Image.open(io.BytesIO(block_payload)).convert("RGB"))
    block_area = texture_pixel_area_m2("9-230-6", (256, 256))
    block_detected = detect(block, area=block_area)
    ice_payload = ICE_FIXTURE.read_bytes()
    assert hashlib.sha256(ice_payload).hexdigest() == ICE_DIGEST
    ice = np.asarray(Image.open(io.BytesIO(ice_payload)).convert("RGB"))
    ice_detected = detect(ice, area=texture_pixel_area_m2("12-1858-52", (256, 256)))

    varied = np.full_like(pixels, 245)
    varied[::2] = 255
    flecks = np.zeros_like(pixels)
    flecks[::8, ::8] = 255
    colored = np.full_like(pixels, 255)
    colored[:, :, 0] = 245
    coast = np.zeros((3, 3), dtype=bool)
    coast[1:, :2] = True  # Northwest cell is sea; other cells touch land.
    projected = texture_ocean_mask(coast, (8, 8))
    expected = np.zeros((8, 8), dtype=bool)
    expected[:4, :4] = True
    small_object = np.zeros_like(pixels)
    small_object[50:150, 50:150] = 255
    fringe_fixture = small_object.copy()
    fringe_fixture[49, 50:150] = 180
    fringe_fixture[20, 50:150] = 180
    restricted_ocean = ocean.copy()
    restricted_ocean[49, 50:80] = False
    fringe_repair = white_ocean_repair_mask(
        fringe_fixture, restricted_ocean, pixel_area_m2=pixel_area,
    )
    checks = {
        "realDiagonalWedgeDetected": bool(
            detected[200, 40] and not detected[20, 200]
            and 0.20 < detected.mean() < 0.23
            and np.any(np.diff(detected.sum(axis=1)) != 0)
        ),
        "realCoarseBlockDetected": bool(
            block_detected[20, 20] and not block_detected[200, 200]
            and 0.36 < block_detected.mean() < 0.39
        ),
        "realIceUntouched": not bool(ice_detected.any()),
        "realIceRepairUntouched": not bool(white_ocean_repair_mask(
            ice, ocean, pixel_area_m2=texture_pixel_area_m2("12-1858-52", (256, 256)),
        ).any()),
        "jpegFringeRepairedWithoutCrossingLandOrDarkOcean": bool(
            np.all(fringe_repair[49, 80:150])
            and not fringe_repair[49, 50:80].any()
            and not fringe_repair[20].any()
            and not fringe_repair[fringe_fixture.max(axis=2) == 0].any()
        ),
        "physicalAreaRejectsSmallObjectAtFineLod": not bool(
            detect(small_object, area=0.25).any()
        ),
        "significantAreaAccepted": bool(detect(small_object).sum() == 10_000),
        "landSnowPreserved": not bool(
            detect(pixels, np.zeros_like(ocean)).any()
        ),
        "variedBrightPixelsRejected": not bool(detect(varied).any()),
        "smallBrightFlecksRejected": not bool(detect(flecks).any()),
        "coloredBrightPixelsRejected": not bool(detect(colored).any()),
        "coastOrientationAndLandBoundary": bool(np.array_equal(projected, expected)),
        "onlyWhiteOceanFlagged": bool(np.all(ocean[detected]) and np.all(pixels[detected] >= 245)),
        "sourceUnmodified": bool(np.array_equal(pixels, original)),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "checks": checks,
        "tileId": "11-934-28",
        "flaggedPixels": int(detected.sum()),
        "flaggedPercent": round(float(detected.mean() * 100), 2),
        "flaggedAreaM2": round(float(detected.sum() * pixel_area)),
        "coarseTileId": "9-230-6",
        "coarseFlaggedPixels": int(block_detected.sum()),
        "coarseFlaggedPercent": round(float(block_detected.mean() * 100), 2),
        "coarseFlaggedAreaM2": round(float(block_detected.sum() * block_area)),
        "iceTileId": "12-1858-52",
        "iceFlaggedPixels": int(ice_detected.sum()),
    }
