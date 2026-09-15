"""Google Maps links for canonical terrain tile addresses."""

import atlantis

from dynamic_functions.Terrain.coords import to_wgs84
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import require_tile_id, tile_bounds


def _google_maps_tile(tile_id: str) -> dict:
    """Build a satellite-view link centered on one terrain tile."""

    depth, _, _ = require_tile_id(tile_id)
    west, south, east, north = tile_bounds(tile_id, GREENLAND_BBOX)
    center_x = (west + east) / 2.0
    center_y = (south + north) / 2.0
    latitude, longitude = to_wgs84(center_x, center_y)
    latitude = float(latitude)
    longitude = float(longitude)
    zoom = max(2, min(22, depth + 4))
    url = (
        "https://www.google.com/maps/@"
        f"{latitude:.7f},{longitude:.7f},{zoom}z/data=!3m1!1e3"
    )
    return {
        "tileId": tile_id,
        "latitude": latitude,
        "longitude": longitude,
        "zoom": zoom,
        "url": url,
    }


@visible
async def google(tile_id: str) -> dict:
    """Show a Google Maps satellite link centered on a terrain tile.

    Example:
        google("10-334-192")
    """

    result = _google_maps_tile(tile_id)
    await atlantis.client_markdown(
        f"[Open tile {tile_id} in Google Maps]({result['url']})"
    )
    return result
