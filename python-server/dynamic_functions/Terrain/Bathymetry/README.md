# Regional bathymetry runtime

This directory owns bathymetry generation. It needs only the server's Python
environment, the Terrain SQLite database, and the two bundled detail-atlas
arrays. There are no external checkouts, shell wrappers, viewer HTTP calls or
survey files involved.

## Pipeline

1. **Inputs (`inputs.py`):** determine the DEM and coastline coverage a depth-8
   job needs. Terrain's normal provider lanes fetch whatever is missing. Water can
   come from an all-water mask, but land needs a measured DEM.
2. **Solve (`solver.py`):** runs with fixed production parameters and never reads
   calibration surveys. The survey fitting commands are experimental and not part
   of acquisition.
3. **Detail (`raster.py`, `detail.py`, `lod.py`):** adds detail from the atlas,
   pins shorelines, and builds parent LODs.
4. **Publish (`worker.py`):** computes from one read-only snapshot and publishes
   every raster in one transaction.

## Invariants

- Bathymetry never modifies DEM or coastline source data.
- An existing depth-8 result makes a repeated job a no-op.
- Moving the camera away releases that region's pending claims.
- Keep `data/detail_atlas.npy` and `data/detail_atlas_res.npy` in source
  control, so a clean checkout reproduces the same seafloor.

## Running

To run a job against a prepared database, from the repository root:

```bash
PYTHONPATH=python-server python-server/venv/bin/python -m \
  dynamic_functions.Terrain.Bathymetry.worker \
  --db /path/to/terrain.db --tile 8-123-5
```

This publishes derived data into that database. `Test.bathymetry_worker_offline()`
runs the whole pipeline against temporary synthetic databases.
