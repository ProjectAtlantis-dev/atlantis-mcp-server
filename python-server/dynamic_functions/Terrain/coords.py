"""Bidirectional coordinate conversion between WGS84 (lat/lon) and NSIDC Polar Stereographic North (EPSG:3413)."""

from pyproj import Transformer

# EPSG:3413 - NSIDC Sea Ice Polar Stereographic North
# Centered at 70°N, 45°W. Units: meters.
_to_stereo = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=False)
_to_wgs84 = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=False)

# nuuk
# % to_stereo 64.175, -51.7388

# % to_wgs84  -333722.40326186264,  -2824336.188842417


@visible
def to_stereo(lat, lon):
    """Convert WGS84 (lat, lon) to polar stereographic (x, y) in meters.

    Accepts scalars or arrays.

    Example:
        >>> x, y = to_stereo(64.175, -51.7388)
        >>> round(x, 1), round(y, 1)
        (-333722.4, -2824336.2)
    """
    x, y = _to_stereo.transform(lat, lon)
    return x, y


@visible
def to_wgs84(x, y):
    """Convert polar stereographic (x, y) in meters to WGS84 (lat, lon).

    Accepts scalars or arrays.

    Example:
        >>> lat, lon = to_wgs84(-333722.4, -2824336.2)
        >>> round(lat, 3), round(lon, 4)
        (64.175, -51.7388)
    """
    lat, lon = _to_wgs84.transform(x, y)
    return lat, lon
