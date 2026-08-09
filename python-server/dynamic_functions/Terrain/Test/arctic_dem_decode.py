"""Directly callable checks for the ArcticDEM raster decoder."""

from pathlib import Path

from dynamic_functions.Terrain.arctic_dem import (
    _decode_source,
    _heightmap_summary,
)
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import tile_bounds


_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "arctic_dem.tif"


@visible
def arcticdem_decode(tile_id: str) -> dict:
    """Decode the bundled test TIFF without network or database access.

    Example:
        arcticdem_decode("10-334-192")
    """

    if not _FIXTURE_PATH.is_file():
        raise FileNotFoundError(
            f"ArcticDEM fixture does not exist: {_FIXTURE_PATH}"
        )

    heightmap = _decode_source(
        _FIXTURE_PATH,
        tile_bounds(tile_id, GREENLAND_BBOX),
    )
    return {
        "tileId": tile_id,
        **_heightmap_summary(heightmap),
    }
