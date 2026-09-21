"""Deterministic shared DEM vertices; stored provider samples remain untouched.

Both Terrain composition and vehicle sampling use this derived surface. A shared
vertex is the mean of valid, same-datum samples from its same-depth neighbors.
Corners include all four incident tiles, independent of request order or batch.
No missing/confidence-zero sample is filled and interiors are unchanged.
"""
import numpy as np
from dynamic_functions.Terrain.Database.tiles import read_dem_payload
from dynamic_functions.Terrain.tile_address import require_tile_id


def read_continuous_dem(db, tile_id, cache=None):
    cache = {} if cache is None else cache

    def raw(key):
        if key not in cache:
            cache[key] = read_dem_payload(db, key)
        return cache[key]

    own = raw(tile_id)
    if own is None:
        return None
    depth, column, row = require_tile_id(tile_id)
    height = own['heightmap'].copy()
    n = height.shape[0]
    sources = {}

    def neighbor(dx, dy):
        x, y = column + dx, row + dy
        if not (0 <= x < 2**depth and 0 <= y < 2**depth):
            return None
        key = f'{depth}-{x}-{y}'
        value = raw(key)
        if value is None or value['vertical_datum'] != own['vertical_datum']:
            return None
        if value['heightmap'].shape != height.shape:
            raise ValueError(f'DEM neighbor grid shape differs: {tile_id}, {key}')
        sources[key] = value['updated_at']
        return value['heightmap']

    for dx, dy, edge, opposite in (
        (-1, 0, (slice(1, -1), 0), (slice(1, -1), -1)),
        (1, 0, (slice(1, -1), -1), (slice(1, -1), 0)),
        (0, -1, (0, slice(1, -1)), (-1, slice(1, -1))),
        (0, 1, (-1, slice(1, -1)), (0, slice(1, -1))),
    ):
        other = neighbor(dx, dy)
        if other is None:
            continue
        a, b = height[edge], other[opposite]
        valid = np.isfinite(a) & np.isfinite(b)
        a[valid] = (a[valid].astype(np.float64) + b[valid]) / 2
    for ix, iy, dx, dy in ((0, 0, -1, -1), (n-1, 0, 1, -1), (0, n-1, -1, 1), (n-1, n-1, 1, 1)):
        if not np.isfinite(height[iy, ix]):
            continue
        samples = [float(own['heightmap'][iy, ix])]
        for ox, oy in ((dx, 0), (0, dy), (dx, dy)):
            other = neighbor(ox, oy)
            if other is None:
                continue
            value = other[n-1-iy if oy else iy, n-1-ix if ox else ix]
            if np.isfinite(value):
                samples.append(float(value))
        height[iy, ix] = sum(sorted(samples)) / len(samples)
    return {**own, 'heightmap': height, 'seam_sources': sources}
