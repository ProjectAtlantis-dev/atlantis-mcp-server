"""Water-aware filtering of south-first child rasters into parent LODs."""

import numpy as np

GRID = 65

def parents_of(children):
    """Group child grids into the 2x2 blocks their parent tile covers."""
    groups = {}
    for (col, row), g in children.items():
        groups.setdefault((col // 2, row // 2), {})[(col & 1, row & 1)] = g
    return groups


def _paste(quad, fill, dtype):
    step = GRID - 1
    canvas = np.full((2 * step + 1, 2 * step + 1), fill, dtype=dtype)
    for (dc, dr), g in quad.items():
        canvas[dr * step:dr * step + GRID, dc * step:dc * step + GRID] = g
    return canvas


def decimate(quad, wet_quad, smooth=True):
    """Fold a 2x2 block of 65x65 grids into one 65x65 parent grid.

    Both the blobs and the tile row index run south-first, so a child's row
    parity is its offset up the canvas -- no flip is needed anywhere.

    Point-sampling the canvas aliases: near a shoreline one fine sample can be
    far deeper than the cell it would come to represent, and promoting it
    carves a spurious trench along the coast. So the canvas is low-passed with
    a 3x3 tent before subsampling.

    The filter is water-aware. Land sits hundreds of metres above the carve, so
    averaging across the waterline would drag shoreline samples upward and
    replace a trench artifact with a shelf artifact. Only samples the finer
    level calls water contribute, and coarse samples with no wet support stay
    NaN for restore_land to fill from the DEM.
    """
    canvas = _paste(quad, np.nan, np.float32)
    if not smooth:
        return canvas[::2, ::2]

    wet = _paste(wet_quad, False, bool) & np.isfinite(canvas)
    values = np.where(wet, canvas, 0.0).astype(np.float64)
    weights = wet.astype(np.float64)

    tent = np.array([1.0, 2.0, 1.0])
    for axis in (0, 1):
        values = _convolve1d(values, tent, axis)
        weights = _convolve1d(weights, tent, axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(weights > 0, values / np.maximum(weights, 1e-12),
                            np.nan)
    return smoothed[::2, ::2].astype(np.float32)


def _convolve1d(arr, kernel, axis):
    """Separable 3-tap convolution with edge clamping."""
    a = np.moveaxis(arr, axis, 0)
    padded = np.concatenate([a[:1], a, a[-1:]], axis=0)
    out = (kernel[0] * padded[:-2] + kernel[1] * padded[1:-1]
           + kernel[2] * padded[2:])
    return np.moveaxis(out, 0, axis)
