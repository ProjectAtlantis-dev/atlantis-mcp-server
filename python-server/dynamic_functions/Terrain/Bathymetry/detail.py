"""World-coordinate sampling of the bundled seafloor detail atlas."""

import numpy as np
from scipy import ndimage

def sample_atlas(atlas, atlas_res, shape, x0, y1, res, scale=1.0, shift=(0.0, 0.0)):
    """Sample the detail atlas by world position, wrapping at its period.

    Indexing on EPSG:3413 coordinates rather than array indices is what makes
    the output deterministic: the same ground gets the same detail regardless
    of which mosaic window or LOD it was extracted at, so neighbouring tiles
    carved in separate runs still line up.
    """
    xs = x0 + np.arange(shape[1]) * res + shift[0]
    ys = y1 - np.arange(shape[0]) * res + shift[1]
    ax = (xs / (atlas_res * scale)) % atlas.shape[1]
    ay = (-ys / (atlas_res * scale)) % atlas.shape[0]
    coords = np.array(np.broadcast_arrays(ay[:, None], ax[None, :]))
    return ndimage.map_coordinates(atlas, coords, order=1, mode="grid-wrap")


def add_detail(zc, mask, floor, res, x0, y1, atlas, atlas_res,
               drape=0.06, lineation=0.03, gully=0.05):
    """Seafloor relief: basin drape, trough lineations, and wall gullies.

    Amplitudes are fractions of the local floor depth, so a 700 m basin gets
    proportionally more relief than a 60 m sandy shallow -- which is also what
    keeps the silted inner reaches reading as smooth.
    """
    d = ndimage.distance_transform_edt(mask) * res

    # deeper water carries more relief; the sandy shallows stay smooth
    strength = np.clip(floor / 400.0, 0.12, 1.0)

    def A(scale, shift=(0.0, 0.0)):
        return sample_atlas(atlas, atlas_res, zc.shape, x0, y1, res, scale, shift)

    # real glacial roughness at two scales: broad basin form plus finer drape
    detail = A(4.0) * floor * drape * strength
    detail += A(1.0, (5000.0, -3000.0)) * floor * drape * 0.45 * strength

    # lineations: ridges parallel to the trough, standing in for glacial
    # lineations and iceberg ploughmarks. The distance-to-shore field's
    # contours run along the channel, so its gradient gives the cross-axis.
    ds = ndimage.gaussian_filter(d, max(400.0 / res, 1.0))
    gy, gx = np.gradient(ds, res)
    ang = np.arctan2(gy, gx)
    yy, xx = np.mgrid[0:zc.shape[0], 0:zc.shape[1]] * res

    # A pure sine here reads as corduroy -- regular, obviously synthetic, and
    # worst around islands where the axis field swings through a full turn.
    # Warping the phase with noise and gating on an independent noise field
    # breaks up the periodicity into discontinuous ridge sets.
    warp = A(2.5, (11000.0, 7000.0)) * 900.0
    phase = (np.cos(ang) * xx + np.sin(ang) * yy + warp) * (2 * np.pi / 900.0)
    gate = np.clip(A(5.0, (-8000.0, 2000.0)) * 0.8 + 0.35, 0, 1)
    detail += np.sin(phase) * gate * floor * lineation * strength

    # gullies incised down the walls, only where the wall is tall enough to
    # have them; same treatment so they don't comb the shallow skerry channels
    wall = np.exp(-((d - 250.0) / 350.0) ** 2) * np.clip(floor / 300.0, 0, 1)
    gwarp = A(1.5, (3000.0, 9000.0)) * 500.0
    gphase = (-np.sin(ang) * xx + np.cos(ang) * yy + gwarp) * (2 * np.pi / 700.0)
    ggate = np.clip(A(3.0, (-2000.0, -6000.0)) * 0.9 + 0.2, 0, 1)
    detail -= np.abs(np.sin(gphase)) * ggate * floor * gully * wall

    # fade detail out at the shoreline so the coast stays crisp
    fade = np.clip(d / max(3 * res, 60.0), 0, 1)
    return np.where(mask, zc + detail * fade, zc)
