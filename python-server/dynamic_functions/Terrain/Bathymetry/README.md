# Regional bathymetry runtime

This directory owns bathymetry generation. It requires only the server's Python
environment, Terrain SQLite database, and the two bundled detail-atlas arrays.
There are no external checkout, shell-wrapper, viewer HTTP, or temporary survey
file dependencies.

`inputs.py` describes exact DEM/coastline coverage for a depth-8 job: the
depth-9 regional solve and halo, depth-11 slope evidence covering that halo,
depth-12 carving and apron, and intermediate parent LODs. Terrain's existing
provider lanes acquire missing inputs. An authoritative all-water mask can
supply the sea-level source surface without a DEM; land requires measured DEM.
Camera polls retain pending prerequisites without reopening completed ones;
moving away releases the old region's pending claims.

`solver.py` retains the established slope-constrained solve, flow routing,
multiscale shore slopes, and sediment deposition. The production parameters
remain decay 14,000 m, slope limits 0.02–1.5, flow exponent 0.35, maximum flow
multiplier 2.5, curvature projection 250 m, and sediment fill 0.7. Experimental
survey fitting/scoring commands are not part of acquisition: the production
solve uses fixed parameters and never reads calibration surveys.

`raster.py`, `detail.py`, and `lod.py` preserve world-coordinate detail,
shoreline pinning/drop limits, and water-aware parent filtering. The bundled
`data/detail_atlas.npy` and `data/detail_atlas_res.npy` are the original atlas
harvested from detrended ArcticDEM land patches; keep both files in source
control so a clean checkout reproduces the same seafloor detail.

`worker.py` computes from one read-only database snapshot, then publishes all
finest and parent rasters in one transaction. It never changes existing DEM
or coastline payloads. An existing depth-8 result makes a repeated job a no-op.
The demand wrapper executes it with the current interpreter and reports the
full captured error output on failure.

For a prepared database, run from the repository root:

```bash
PYTHONPATH=python-server python-server/venv/bin/python -m \
  dynamic_functions.Terrain.Bathymetry.worker \
  --db /path/to/terrain.db --tile 8-123-5
```

This command publishes derived bathymetry. `Test.bathymetry_worker_offline()`
uses temporary synthetic databases to verify the pipeline, atomic rollback,
source preservation, shoreline constraints, and prerequisite queue lifetime.
It also removes DEM/coastline inputs, exercises the real demand lanes and
retry path with fixture acquisitions, and runs the local worker subprocess
through successful publication.
