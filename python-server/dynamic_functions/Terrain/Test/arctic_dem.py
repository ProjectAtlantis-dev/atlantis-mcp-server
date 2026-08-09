"""Directly callable checks for the ArcticDEM raster decoder."""

import hashlib
from pathlib import Path

import numpy as np

from dynamic_functions.Terrain.arctic_dem import _decode_source
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
    valid = heightmap[np.isfinite(heightmap)]
    return {
        "tileId": tile_id,
        "shape": list(heightmap.shape),
        "dtype": str(heightmap.dtype),
        "minimum": float(np.min(valid)) if valid.size else None,
        "maximum": float(np.max(valid)) if valid.size else None,
        "nanCount": int(np.isnan(heightmap).sum()),
        "digest": hashlib.sha256(heightmap.tobytes()).hexdigest(),
    }
