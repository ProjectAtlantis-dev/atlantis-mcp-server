"""Visible offline checks that ArcticDEM acquisition failures stay visible."""

from unittest.mock import patch

from rasterio.errors import RasterioIOError

from dynamic_functions.Terrain.arctic_dem import _fetch_heightmap


def _surfaced_failure(tile_id: str, failure: Exception) -> dict:
    with patch(
        "dynamic_functions.Terrain.arctic_dem._decode_source",
        side_effect=failure,
    ) as decode:
        caught = None
        try:
            _fetch_heightmap(tile_id)
        except Exception as exc:
            caught = exc
    if caught is None:
        raise AssertionError(f"ArcticDEM swallowed {type(failure).__name__}")
    return {
        "raised": True,
        "type": type(caught).__name__,
        "message": str(caught),
        "decodeAttempts": decode.call_count,
        "sameException": caught is failure,
    }


@visible
def arcticdem_failures(tile_id: str) -> dict:
    """Inject ArcticDEM failures without network or database access."""

    cases = {
        "rateLimited": _surfaced_failure(
            tile_id,
            RasterioIOError("HTTP response code: 429 Too Many Requests"),
        ),
        "transient": _surfaced_failure(
            tile_id,
            RasterioIOError("HTTP response code: 503 Service Unavailable"),
        ),
        "network": _surfaced_failure(
            tile_id,
            TimeoutError("synthetic ArcticDEM timeout"),
        ),
        "corrupt": _surfaced_failure(
            tile_id,
            RasterioIOError("not recognized as a supported raster format"),
        ),
        "noData": _surfaced_failure(
            tile_id,
            RasterioIOError("HTTP response code: 404 Not Found"),
        ),
    }
    checks = {
        "rateLimitVisible": (
            cases["rateLimited"]["sameException"]
            and "429" in cases["rateLimited"]["message"]
        ),
        "transientVisible": (
            cases["transient"]["sameException"]
            and "503" in cases["transient"]["message"]
        ),
        "networkVisible": cases["network"]["sameException"],
        "corruptVisible": cases["corrupt"]["sameException"],
        "noDataVisible": (
            cases["noData"]["sameException"]
            and "404" in cases["noData"]["message"]
        ),
        "noImplicitRetry": all(
            case["decodeAttempts"] == 1 for case in cases.values()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(
            "ArcticDEM failure checks failed: " + ", ".join(failed)
        )

    return {
        "tileId": tile_id,
        "offline": True,
        "databaseAccess": False,
        "checks": checks,
        "cases": cases,
        "note": (
            "ArcticDEM currently surfaces provider failures unchanged and "
            "does not retry them. This test intentionally adds no production "
            "exception classification."
        ),
    }
