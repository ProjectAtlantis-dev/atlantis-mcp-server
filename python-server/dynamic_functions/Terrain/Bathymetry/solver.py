"""Slope-constrained regional bed solve, flow routing, and sediment deposition.

The production model uses fixed parameters. Survey scoring and experimental
calibration commands are deliberately outside the acquisition runtime.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .raster import extrapolate, pin_waterline

def channel_halfwidth(mask, res, rmax=6000.0, steps=14):
    """Radius of the largest disc of water that covers each pixel.

    `distance_transform_edt` alone gives distance to the nearest shore, which
    collapses to zero at the waterline -- useless as a channel width, since
    every shoreline pixel would read as infinitely narrow. What the depth model
    needs is the width of the channel a pixel *belongs to*, constant across the
    cross-section.

    So take the morphological covering radius: w(p) = max{ d(q) : |p-q| <= d(q) }.
    A pixel inherits the widest inscribed disc that reaches it. This does not
    leak across a headland the way a plain max-filter does -- a disc from the
    main fjord only reaches as far as its own radius, so a 300 m side inlet
    hanging off a 4 km fjord stays 300 m wide.
    """
    d = ndimage.distance_transform_edt(mask) * res
    w = np.zeros_like(d)
    for r in np.geomspace(res, rmax, steps):
        seed = d >= r
        if not seed.any():
            break
        reach = ndimage.binary_dilation(seed, ndimage.generate_binary_structure(2, 2),
                                        iterations=max(1, int(round(r / res))))
        w = np.where(reach, np.maximum(w, r), w)
    return np.maximum(w, d)


def shore_slope_field(z, mask, land, res, band_lo=100.0, band_hi=400.0,
                      smooth_m=120.0, reach_m=1200.0,
                      curvature_projection_m=0.0, w_near=0.5,
                      reach_local=250.0, local_blend_m=600.0,
                      min_land_elev=1.0):
    """Slope of the land rising away from the shore, carried into the water.

    Measure separate lower- and upper-hillside bands and weight the lower band
    more strongly: it is the part of the hill that continues below sea level.
    The samples immediately at the water edge and all nominal land below
    ``min_land_elev`` remain excluded because the source DEM artificially
    flattens there. In particular, samples below ``min_land_elev`` can be
    water returns or coastline-raster phase errors even when the polygon
    labels them land.

    Close to shore, a local push-pull field prevents an adjacent headland from
    lending its steep slope to a gentle bay. Offshore, that local field blends
    into the broad field needed to bridge gaps without Voronoi seams.

    ``curvature_projection_m`` optionally preserves a second-order signal:
    where the shoreline-normal land slope grows toward the water, continue
    that trend for the requested distance. Only positive acceleration is
    retained, so fine detail may relax the descent constraint but never make
    an established basin shallower.
    """
    d_land = ndimage.distance_transform_edt(~mask) * res
    gy, gx = np.gradient(np.nan_to_num(z), res)
    evidence = land & np.isfinite(z) & (z >= min_land_elev)
    sigma = max(smooth_m / res / 2.0, 0.5)
    raw_grad = np.hypot(gx, gy)
    grad_weight = ndimage.gaussian_filter(evidence.astype(np.float64), sigma)
    def smooth_evidence(values):
        return (
            ndimage.gaussian_filter(np.where(evidence, values, 0.0), sigma)
            / np.maximum(grad_weight, 1e-12)
        )

    grad = smooth_evidence(raw_grad)

    # The signed-distance gradient points from land toward water. Projecting
    # terrain gradient onto it removes along-shore relief before measuring how
    # quickly the cross-shore slope changes toward the coastline.
    signed_distance = (
        ndimage.distance_transform_edt(mask)
        - ndimage.distance_transform_edt(~mask)
    )
    normal_y, normal_x = np.gradient(signed_distance)
    normal_length = np.hypot(normal_x, normal_y)
    normal_x /= np.maximum(normal_length, 1e-12)
    normal_y /= np.maximum(normal_length, 1e-12)
    normal_slope = smooth_evidence(
        np.maximum(-(gx * normal_x + gy * normal_y), 0.0)
    )

    near = evidence & (d_land >= 0.5 * band_lo) & (d_land < 2.0 * band_lo)
    far = evidence & (d_land >= 2.0 * band_lo) & (d_land <= band_hi)
    if not near.any():
        near = evidence
    if not far.any():
        far = near

    d_shore = ndimage.distance_transform_edt(mask) * res
    local_weight = np.exp(-d_shore / max(local_blend_m, res))

    def carry(field, samples):
        local = extrapolate(
            field, samples, res, coarsest_m=max(reach_local, res),
        )
        broad = extrapolate(field, samples, res, coarsest_m=reach_m)
        return local * local_weight + broad * (1.0 - local_weight)

    s_near = carry(grad, near)
    s_far = carry(grad, far)
    base = w_near * s_near + (1.0 - w_near) * s_far
    if curvature_projection_m <= 0.0:
        return base

    normal_near = carry(normal_slope, near)
    normal_far = carry(normal_slope, far)
    near_midpoint = 0.5 * (0.5 * band_lo + 2.0 * band_lo)
    far_midpoint = 0.5 * (2.0 * band_lo + band_hi)
    sample_separation = max(far_midpoint - near_midpoint, res)
    acceleration = np.maximum(normal_near - normal_far, 0.0)
    return base + acceleration * (
        float(curvature_projection_m) / sample_separation
    )


def combine_multiscale_slopes(coarse, fine, factor):
    """Downsample an aligned fine slope field without weakening the coarse one."""
    factor = int(factor)
    if factor < 1:
        raise ValueError("multiscale factor must be positive")
    coarse = np.asarray(coarse, dtype=np.float64)
    fine = np.asarray(fine, dtype=np.float64)
    expected = tuple((size - 1) * factor + 1 for size in coarse.shape)
    if fine.shape != expected:
        raise ValueError(
            f"fine slope shape {fine.shape} does not align with "
            f"coarse shape {coarse.shape} at factor {factor}"
        )
    sampled = fine[::factor, ::factor]
    return np.maximum(coarse, sampled)


def ocean_distance(mask, res):
    """Geodesic distance through water to the open ocean, for flow routing."""
    ys, xs = np.nonzero(mask)
    idx = -np.ones(mask.shape, np.int64)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)
    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ay, ax = ys + dy, xs + dx
        ok = (ay >= 0) & (ay < mask.shape[0]) & (ax >= 0) & (ax < mask.shape[1])
        ok[ok] &= mask[ay[ok], ax[ok]]
        a, b = np.arange(n)[ok], idx[ay[ok], ax[ok]]
        w = np.full(ok.sum(), np.hypot(dy, dx) * res)
        rows += [a, b]; cols += [b, a]; vals += [w, w]
    g = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()

    edge = np.zeros(mask.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    lab, nl = ndimage.label(mask)
    el = set(np.unique(lab[mask & edge])) - {0}
    sizes = ndimage.sum(mask, lab, range(1, nl + 1))
    ocean = max(el, key=lambda L: sizes[L - 1]) if el else 0
    seeds = idx[(lab == ocean) & edge]
    seeds = seeds[seeds >= 0]
    d = dijkstra(g, indices=seeds, min_only=True)
    d[~np.isfinite(d)] = d[np.isfinite(d)].max() if np.isfinite(d).any() else 0.0
    out = np.zeros(mask.shape)
    out[ys, xs] = d
    return out


def flow_accumulation(z, mask, res):
    """Upstream catchment area, routed across land and on down the fjords.

    The bed solve on its own knows only how far a point is from the shore and
    how steep that shore is. It has no along-valley coordinate at all, which is
    why a canyon reaching the coast stopped dead there: its walls contributed
    some steepness to the allowance field, but nothing carried the valley's
    *axis* offshore.

    Flux does. Route drainage over one continuous surface -- the real DEM on
    land, and a gently ocean-ward ramp beneath the water -- and the routing
    crosses the coastline without noticing it. Tributaries arrive at their
    mouths carrying their catchments, merge into the trunk, and accumulate down
    the fjord exactly as ice did. The result is a field that is large along
    every valley axis and its submarine continuation, and small on headlands
    that drain nothing.
    """
    od = ocean_distance(mask, res)
    # one routing surface: land as measured, water as a ramp falling seaward.
    # Sits entirely below sea level so land always drains into it, and is
    # monotone in distance-to-ocean so water always drains out.
    surf = np.where(mask, -50.0 + od * 1e-4, np.nan_to_num(z).astype(np.float64))

    # priority-flood: fill pits so every land cell has a descending path out.
    # Seeded from the water, which is the true outlet of the whole domain.
    import heapq
    H, W = surf.shape
    filled = surf.copy()
    done = np.zeros(surf.shape, bool)
    heap = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        heap.append((surf[y, x], y, x))
    done[mask] = True
    heapq.heapify(heap)
    nb = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while heap:
        e, y, x = heapq.heappop(heap)
        for dy, dx in nb:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not done[ny, nx]:
                done[ny, nx] = True
                v = max(filled[ny, nx], e + 1e-4)
                filled[ny, nx] = v
                heapq.heappush(heap, (v, ny, nx))

    # D8 steepest descent on the filled surface
    best = np.zeros(surf.shape)
    rec_y = np.zeros(surf.shape, np.int32)
    rec_x = np.zeros(surf.shape, np.int32)
    yy, xx = np.mgrid[0:H, 0:W]
    rec_y[:], rec_x[:] = yy, xx
    for dy, dx in nb:
        sh = np.roll(np.roll(filled, -dy, 0), -dx, 1)
        if dy: sh[(0 if dy < 0 else -1), :] = np.inf
        if dx: sh[:, (0 if dx < 0 else -1)] = np.inf
        drop = (filled - sh) / np.hypot(dy, dx)
        take = drop > best
        best = np.where(take, drop, best)
        rec_y = np.where(take, yy + dy, rec_y)
        rec_x = np.where(take, xx + dx, rec_x)

    # accumulate in descending order of filled elevation: every cell is
    # processed after everything that drains into it
    acc = np.full(surf.shape, res * res, np.float64)
    order = np.argsort(filled, axis=None)[::-1]
    fy, fx = np.unravel_index(order, surf.shape)
    ry, rx = rec_y.ravel(), rec_x.ravel()
    flat = np.ravel_multi_index((ry, rx), surf.shape)
    a1 = acc.ravel()
    for i in order:
        j = flat[i]
        if j != i:
            a1[j] += a1[i]
    return acc, od


def solve_bed(z, mask, land, res, s_field, decay_m, s_min, s_max, flux=None,
              flux_p=0.0, flux_lo=0.4, flux_hi=2.5, flux_ref=None):
    """Deepest surface that never out-steepens its shoreline allowance.

    Multi-source shortest path over the water graph, with edge cost equal to
    the local slope allowance times the step length. A super-source carries
    each attach point's own elevation as its entry cost, so shores at different
    heights all pin correctly in one solve rather than needing a pass each.
    """
    d_shore = ndimage.distance_transform_edt(mask) * res
    # The allowance decays offshore: full land slope at the waterline, easing
    # to nothing in the basin. Without this the descent never flattens and
    # every fjord bottoms out in a V.
    #
    # Flux lengthens the decay rather than steepening the slope. That
    # distinction is the whole point: scaling the slope by flux would let a
    # high-discharge valley authorise a submarine wall steeper than the land
    # above it, which is the one thing this model exists to prevent. Stretching
    # the decay instead lets a big valley keep descending further out -- so it
    # goes deeper without ever going steeper, and its channel continues
    # offshore for as long as its flux does.
    dec = np.full(mask.shape, float(decay_m))
    if flux is not None and flux_p > 0:
        ref = (
            float(flux_ref)
            if flux_ref is not None
            else float(np.median(flux[mask]))
        )
        if not np.isfinite(ref) or ref <= 0.0:
            raise ValueError("flux reference must be finite and positive")
        dec = dec * np.clip((flux / max(ref, 1e-9)) ** flux_p, flux_lo, flux_hi)
    s_eff = np.clip(s_field, s_min, s_max) * np.exp(-d_shore / dec)

    ys, xs = np.nonzero(mask)
    idx = -np.ones(mask.shape, np.int64)
    idx[ys, xs] = np.arange(len(ys))
    n = len(ys)

    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ay, ax = ys + dy, xs + dx
        ok = (ay >= 0) & (ay < mask.shape[0]) & (ax >= 0) & (ax < mask.shape[1])
        ok[ok] &= mask[ay[ok], ax[ok]]
        a, b = np.arange(n)[ok], idx[ay[ok], ax[ok]]
        step = np.hypot(dy, dx) * res
        # average the allowance across the step so the cost is symmetric
        w = 0.5 * (s_eff[ys[a], xs[a]] + s_eff[ay[ok], ax[ok]]) * step
        rows += [a, b]; cols += [b, a]; vals += [w, w]

    # Attach points are the shoreline itself, and the shoreline is at sea
    # level by definition -- the mesh starts flat and is pushed down from
    # there. The land contributes its *slope* as the descent allowance, never
    # its elevation: pinning the mesh to the height of the ground behind the
    # coast lets a 900 m clifftop vote to keep the water shallow, which is both
    # wrong and the same category of mistake as the old z_shore anchor.
    att = mask & (ndimage.distance_transform_edt(mask) <= 1.5)
    if not att.any():
        raise ValueError("no shoreline: the water mask touches no land")
    a_nodes = idx[att]
    rows.append(np.full(len(a_nodes), n)); cols.append(a_nodes)
    # These nodes are the waterline itself, not the first offshore step. Pin
    # them to sea level; charging one edge here starts every shoreline below
    # zero and creates a cliff before the graph has travelled anywhere.
    vals.append(np.zeros(len(a_nodes), dtype=np.float64))

    g = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n + 1, n + 1)).tocsr()
    d = dijkstra(g, indices=n, min_only=True)[:n]

    bed = np.full(mask.shape, np.nan)
    bed[ys, xs] = -d                            # descent below sea level
    return bed, s_eff, d_shore


def deposit(bed, mask, res, frac, taper=0.5, radii=(2, 3, 4, 6, 8, 12, 16, 24)):
    """Fill the keel to a flat floor: erosion cuts a V, deposition makes it a U.

    The descent is pure erosion, so it bottoms out wherever the constraints
    from opposing shores meet -- a line, not a floor. That is why the deepest
    cells overshoot: two walls converging both keep descending right up to the
    point they cross. Real troughs do not look like that, because sediment
    fills the keel from the bottom up, and it is the fill that produces a flat
    floor and puts a lid on the maximum depth.

    Implemented as a grey-scale morphological closing, which is exactly "raise
    the surface until a disc of radius r can roll along it": the keel fills,
    anything already broader than the disc is untouched. The disc scales with
    the local channel half-width, so a wide fjord gets a wide floor and a
    narrow inlet a narrow one, rather than one authored floor width everywhere.

    Weighted to vanish at the shoreline. Without the taper the closing lifts
    the walls as readily as the floor and the whole trough shallows uniformly,
    which is flattening, not filling.
    """
    hw = channel_halfwidth(mask, res)
    d_shore = ndimage.distance_transform_edt(mask) * res
    z = np.where(mask, np.nan_to_num(bed), 0.0)

    # closing at a ladder of radii; each pixel reads the one matching its channel
    want = np.clip(0.5 * hw / res, radii[0], radii[-1])
    closed = z.copy()
    prev = None
    for r in radii:
        k = 2 * int(r) + 1
        c = ndimage.minimum_filter(ndimage.maximum_filter(z, size=k), size=k)
        if prev is None:
            closed = np.where(want <= r, c, closed)
        else:
            # linear blend between adjacent rungs so the floor width varies
            # smoothly instead of stepping where a bucket changes
            t = np.clip((want - prev[0]) / (r - prev[0]), 0.0, 1.0)
            band = (want > prev[0]) & (want <= r)
            closed = np.where(band, prev[1] * (1 - t) + c * t, closed)
        prev = (r, c)
    assert prev is not None
    closed = np.where(want > radii[-1], prev[1], closed)

    # taper: no fill at the waterline, full fill by mid-channel
    w = np.clip(d_shore / np.maximum(taper * hw, res), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)

    # A morphological closing meets the wall at a hard corner, and that corner
    # is steeper than the descent was ever allowed to be -- it took the max
    # submarine slope from 1.45 to 2.01, past the model's own ceiling. Real
    # sediment cannot stand at a corner either: it has an angle of repose, and
    # subaqueous silt's is very low. So smooth the deposited thickness before
    # laying it down. The fill still fills; it just cannot stack into a step
    # the erosional constraint would have forbidden.
    inc = ndimage.gaussian_filter(np.maximum(closed - z, 0.0) * w, 1.5)
    # Closing can lift a few shallow-water samples centimetres above sea
    # level. They are still inside the authoritative water polygon. Where the
    # deposit would cross zero, retain the pre-deposition erosional bed instead
    # of flattening an interior sample to sea level.
    deposited = z + frac * inc
    out = np.where(deposited > 0.0, z, deposited)
    # Gaussian smoothing of the deposited thickness otherwise bleeds fill
    # onto the zero-depth boundary and lifts shoreline vertices above sea
    # level. Every bed-producing pass must preserve the flex-grid pin.
    pin_waterline(out, mask)
    return np.where(mask, out, np.nan)
