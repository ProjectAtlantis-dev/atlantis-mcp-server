"""Ordered rectangle lookup for verified terrain tiles and masks.

Indexing changes candidate lookup only. Highest-resolution ordering, inclusive
bounds and unknown/confidence handling remain owned by the sampling callers.
"""
import math


class RectangleIndex:
    def __init__(self, items, bounds, extent, cell_size=256):
        self.cell_size = cell_size
        self.cells = {}
        ex0, ey0, ex1, ey1 = extent
        for item in items:
            x0, y0, x1, y1 = bounds(item)
            lo_x, lo_y = max(x0, ex0), max(y0, ey0)
            hi_x, hi_y = min(x1, ex1), min(y1, ey1)
            if lo_x > hi_x or lo_y > hi_y:
                continue
            entry = (x0, y0, x1, y1, item)
            for row in range(math.floor(lo_y/cell_size), math.floor(hi_y/cell_size)+1):
                for col in range(math.floor(lo_x/cell_size), math.floor(hi_x/cell_size)+1):
                    self.cells.setdefault((col, row), []).append(entry)

    def find(self, x, y):
        for x0, y0, x1, y1, item in self.cells.get((math.floor(x/self.cell_size), math.floor(y/self.cell_size)), ()):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return item
        return None
