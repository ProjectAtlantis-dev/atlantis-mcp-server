"""Canonical parsing and formatting for terrain quadtree tile identifiers."""

from __future__ import annotations

from typing import TypeAlias


TileAddress: TypeAlias = tuple[int, int, int]
TileBounds: TypeAlias = tuple[float, float, float, float]


def parse_tile_id(tile_id: object) -> TileAddress | None:
    """Return ``(depth, column, row)`` for a valid terrain tile identifier."""
    if not isinstance(tile_id, str):
        return None
    parts = tile_id.split("-")
    if len(parts) != 3:
        return None
    try:
        address = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    if any(value < 0 for value in address):
        return None
    return address


def require_tile_id(tile_id: object) -> TileAddress:
    """Parse a tile identifier or raise a consistent ``ValueError``."""
    address = parse_tile_id(tile_id)
    if address is None:
        raise ValueError(f"invalid terrain tile id: {tile_id!r}")
    return address


def format_tile_id(depth: int, column: int, row: int) -> str:
    """Format a terrain tile address using the repository's canonical form."""
    address = (int(depth), int(column), int(row))
    if any(value < 0 for value in address):
        raise ValueError(f"terrain tile address must be non-negative: {address!r}")
    return "-".join(str(value) for value in address)


def tile_bounds(tile_id: object, root_bbox: TileBounds) -> TileBounds:
    """Return one tile's bounds within the supplied quadtree root."""
    depth, column, row = require_tile_id(tile_id)
    tiles_per_axis = 1 << depth
    if column >= tiles_per_axis or row >= tiles_per_axis:
        raise ValueError(f"terrain tile address is outside depth {depth}: {tile_id!r}")
    root_x0, root_y0, root_x1, root_y1 = map(float, root_bbox)
    tile_width = (root_x1 - root_x0) / tiles_per_axis
    tile_height = (root_y1 - root_y0) / tiles_per_axis
    x0 = root_x0 + column * tile_width
    y0 = root_y0 + row * tile_height
    return x0, y0, x0 + tile_width, y0 + tile_height


def inset_tile_corners(
    tile_id: object,
    root_bbox: TileBounds,
) -> tuple[tuple[float, float], ...]:
    """Return SW, SE, NW, NE points infinitesimally inside one tile.

    The inset keeps east and north corners assigned to this tile instead of
    the adjacent quadtree cell while remaining immaterial to source rasters.
    """
    x0, y0, x1, y1 = tile_bounds(tile_id, root_bbox)
    west, south, east, north = x0, y0, x1, y1
    # ``nextafter`` would add a numpy dependency to this small address module.
    # One part per billion is sub-millimetric for the validation tiles.
    inset = min(x1 - x0, y1 - y0) * 1e-9
    west += inset
    south += inset
    east -= inset
    north -= inset
    return (
        (west, south),
        (east, south),
        (west, north),
        (east, north),
    )
