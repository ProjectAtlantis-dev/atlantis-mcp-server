"""Bathymetry raster sampling and shoreline constraints."""

import numpy as np
from scipy import ndimage

def sample_global(field, meta, shape, x0, y1, res):
    """Bilinear-sample a world-referenced global raster onto a local grid."""
    gx0, gy1, gres = meta["x0"], meta["y1"], meta["res"]
    xs = x0 + np.arange(shape[1]) * res
    ys = y1 - np.arange(shape[0]) * res
    ax = (xs - gx0) / gres
    ay = (gy1 - ys) / gres
    coords = np.array(np.broadcast_arrays(ay[:, None], ax[None, :]))
    return ndimage.map_coordinates(field, coords, order=1, mode="nearest")


def pin_waterline(field, mask):
    """Pin the first wet vertex ring to sea level, in place.

    The global solve is pinned on its own coastline grid, but a finer output
    mask need not put the shoreline on exactly the same samples. Reassert the
    boundary condition after resampling so LOD and mask differences cannot
    move the flex grid's waterline below zero.
    """
    water = np.asarray(mask, dtype=bool)
    shore = water & np.isfinite(field) & ndimage.binary_dilation(
        ~water, structure=np.ones((3, 3), dtype=bool),
    )
    field[shore] = 0.0
    return shore


def repair_resampled_waterline(field, mask, res, source_res):
    """Remove the wide zero shelf made by resampling a coarser pinned grid.

    The global bed is solved at depth 9 (about 82 m/px). Its first wet vertex
    ring is correctly pinned to zero, but bilinear sampling that ring onto a
    depth-12 grid turns one boundary vertex into as many as eight consecutive
    zero vertices. At an oblique or narrow shoreline the plateau can be wider
    still. The destination coastline is finer and is the authority, so carry
    the nearby interior bed into its water polygon and taper it back to zero
    over one *source* sample. ``pin_waterline`` then fixes the exact first wet
    ring; no coarse-grid shoreline phase remains in the delivered surface.

    Only exact/non-negative sampled bed values are repaired. Real negative
    bathymetry is left byte-for-byte alone, so this cannot reshape the basin.
    """
    water = np.asarray(mask, dtype=bool)
    out = np.asarray(field)
    if not water.any() or source_res <= res:
        return np.zeros(water.shape, dtype=bool)

    # Negative samples are the trustworthy interior of the coarse solution.
    # Push-pull extrapolation avoids a nearest-neighbour Voronoi seam while
    # supplying the bed that lies beneath a fine water polygon which the
    # coarse shoreline row could not resolve.
    interior = water & np.isfinite(out) & (out < -1e-6)
    if not interior.any():
        return np.zeros(water.shape, dtype=bool)
    carried = extrapolate(out, interior, res, coarsest_m=max(4 * source_res, 600.0))

    if water.all():
        # scipy's EDT treats the array origin as an implicit exterior zero
        # when no real zero exists. A fully wet destination tile has no local
        # shoreline, so that convention would preserve one false zero at its
        # southwest corner.
        amount = np.ones(water.shape, dtype=np.float64)
    else:
        distance = ndimage.distance_transform_edt(water) * res
        amount = np.clip((distance - res) / max(source_res, res), 0.0, 1.0)
        amount = amount * amount * (3.0 - 2.0 * amount)
    candidate = np.minimum(carried, 0.0) * amount

    repair = water & np.isfinite(out) & (out >= -1e-6) & (amount > 0.0)
    out[repair] = np.minimum(out[repair], candidate[repair])
    return repair


def limit_shoreline_drop(field, mask, res, slope_cap=1.5):
    """Reassert the maximum descent from this grid's own shoreline.

    A fine coastline can put its first water vertex beside a coarse bed sample
    which is already tens of metres deep. Pinning the first vertex alone then
    turns the coarse-to-fine phase error into a cliff. The global eikonal solve
    already obeys this bound on its grid; reapply the same Lipschitz constraint
    after resampling and destination-mask restoration.
    """
    water = np.asarray(mask, dtype=bool)
    if not water.any() or water.all():
        return np.zeros(water.shape, dtype=bool)
    shore = water & ndimage.binary_dilation(
        ~water, structure=np.ones((3, 3), dtype=bool),
    )
    distance_from_pin = ndimage.distance_transform_edt(~shore)
    allowed_depth = slope_cap * distance_from_pin * res
    too_deep = water & np.isfinite(field) & (field < -allowed_depth)
    field[too_deep] = -allowed_depth[too_deep]
    return too_deep


def extrapolate(field, valid, res, coarsest_m=6000.0, levels=7, floor_frac=0.05):
    """Spread `field` from `valid` cells across the whole grid, smoothly.

    The obvious way to carry shoreline values inland-of-water is to sample
    through `distance_transform_edt(return_indices=True)`. Don't: that is a
    Voronoi partition, and every field read through it steps discontinuously
    wherever the closest source flips -- which is exactly along the rays
    radiating from headlands. Post-blurring cannot undo it, because the jump
    is already in the sampled values.

    Instead do a push-pull fill. Build a Gaussian pyramid of the masked field
    and its weight, then read each pixel from the finest level that has enough
    support. The result is C1-smooth everywhere, has no preferred direction,
    and still hugs the true values close to the sources.
    """
    f0 = np.where(valid, np.nan_to_num(field), 0.0)
    w0 = valid.astype(np.float64)

    if not valid.any():
        return np.zeros_like(f0)

    out = np.full(f0.shape, float(field[valid].mean()))
    sigma = coarsest_m / res
    for _ in range(levels):          # coarse -> fine, each blended over the last
        num = ndimage.gaussian_filter(f0, sigma)
        den = ndimage.gaussian_filter(w0, sigma)
        val = num / np.maximum(den, 1e-12)
        # Confidence ramps smoothly with local support. A hard threshold here
        # terraces the output -- visible as contour-parallel steps wherever a
        # level switches on -- so blend proportionally instead.
        # den is already a normalised support fraction in [0,1]; scaling it by
        # den.max() made the criterion depend on the window's contents
        a = np.clip(den / floor_frac, 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)      # smoothstep, C1 at both ends
        out = val * a + out * (1.0 - a)
        sigma /= 2.0
        if sigma < 0.8:
            break
    return out
