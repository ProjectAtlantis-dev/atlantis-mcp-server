# Terrain dynamic functions

DEM acquisition evaluates ArcticDEM 10 m and Copernicus GLO-30 independently,
then persists the candidate with the most finite samples; ArcticDEM wins an
equal-coverage tie because of its finer native resolution. A failure from one
provider remains visible in the acquisition record but does not suppress valid
data from the other. If neither provider resolves the tile, the worker raises
one error containing both named provider failures.

ArcticDEM supplies WGS84 ellipsoidal elevations. The Terrain tools convert
them to orthometric EGM2008 elevations before comparison and persistence so
that sea level is approximately zero metres. Copernicus GLO-30 is already
EGM2008 orthometric.

The conversion requires PROJ's open `us_nga_egm08_25.tif` grid. Install it in
the Python environment used to run Atlantis:

```bash
projsync --file us_nga_egm08_25.tif
```

The ArcticDEM candidate fails closed when the real grid-backed transformation
is not available; Copernicus may still resolve the request independently.
PROJ's ballpark zero-offset fallback must never be used because it would store
sea-level terrain roughly 28–49 metres too high around Greenland.

## Ready-data batch tools

`compose_tiles(tile_ids)` returns independently available DEM, water, and
texture state without provider access, scheduling, or writes.
DEM geometry is publishable as soon as its measured heightmap exists;
coastline, hydrography, connectivity, and bathymetry remain independent state
and may refine that geometry later without withholding it from the viewer.
Where a GTK50 coastline mask is available (including a projected depth-12
ancestor), it controls both land and sea. Connected WMS hydrography supplies
water only where that coastline coverage is absent; connectivity does not
authorize cutting into GTK50 land. Both source masks remain visible in water
status for diagnosis.

`compose_tiles_binary(tile_ids, known_digests)` encodes the same ready-data
batch with the browser's aligned `binary-v1` envelope. `known_digests` maps tile
IDs to the eight-digit CRC32 returned in each tile's `heightmap` field. Matching
heightmap blocks are omitted and reported with `heightmapBytes: 0`. Because MCP
tool results are JSON, the visible tool base64-wraps the complete binary
envelope in `contentBase64`; the decoded bytes are the exact browser payload.

`camera_lod(camera_x, camera_y, max_range, max_depth, altitude,
previous_depth)` applies the Flask viewer's radial and altitude LOD rules using
only supplied EPSG:3413 camera values. It returns a deterministic, 2:1-balanced
desired leaf set plus nearest-ready DEM fallback coverage and an explicit list
of exact target tiles still missing. Render coverage follows the current
desired leaves: exact tiles when ready, with ancestors retained only where
needed to fill unresolved gaps. Finer cached descendants outside that leaf set
remain in the database for later camera passes but do not override a coarser
radial or altitude LOD. The call is read-only and never performs provider access
or schedules work. Quality monotonicity is relative to the current camera
request: while that request remains current, coverage may improve from ancestor
fallback to the requested exact tiles but must never regress in the opposite
direction. A camera move creates a new distance-derived request and can
correctly select coarser leaves.

`compose_camera_binary(...)` composes that hierarchical ready coverage and
returns the same base64-wrapped `binary-v1` envelope as the explicit batch
tool. Tile bboxes are relative to the supplied origin (or camera by default),
while `stereoBbox` retains absolute EPSG:3413 bounds. Ready coverage is read in
bounded chunks, so a camera leaf set may safely exceed the explicit batch
tool's 256-ID input limit.

## Nonblocking demand

`submit_camera_demand(...)` compares the supplied camera's desired leaves with
ready local rows, submits only absent work, and returns without waiting for a
worker. DEM, texture, coastline, hydrography, and tidal-connectivity work use
separate bounded lanes, so a slow or failing provider cannot consume another
domain's capacity. Coastline and hydrography are dependency-staged behind a
ready DEM and normalized to the depth-12 WMS contract; connectivity is staged
behind ready hydrography and derived off the camera path.

`compose_camera_demand_binary(...)` first composes the best coherent ready
coverage, then performs the same quick submissions and returns the browser
binary envelope. `demand_status()` exposes active, pending, completed, and
failed lane state without waiting. Failed work is held rather than hot-looped;
bounded retry eligibility is a separate migration step.

Every new camera submission replaces each lane's unstarted queue with the
current nearest-first demand. Work already running is never cancelled and may
publish normally, while obsolete pending IDs are dropped before they consume
provider capacity. Ready ancestor coverage follows the priority of the nearest
desired leaf it represents.

Transient transport, timeout, rate-limit, and server failures receive bounded
retry deadlines (2 seconds, then 10 seconds). A worker performs one attempt and
releases its slot; no worker sleeps for backoff. A later camera refresh makes
eligible work runnable. Invalid inputs and payloads, credentials, dimensions,
clobber conflicts, and exhausted attempts remain terminal and visible in lane
status.

Responses summarize whether useful work is active/pending (`nextAction: poll`),
waiting for a future retry deadline (`nextAction: retry`), or terminal/complete
(`nextAction: idle`). Retry timing is explicit, and failures from obsolete
camera claims do not keep the current view polling. Repeated ready-data
composition retains coherent ancestor fallback while exact leaves converge.

## Ocean imagery repair trial

`Test.ocean_texture_offline()` checks a conservative white-gap detector against
the original cached textures for the diagonal gap `11-934-28`, offshore blank
area `9-230-6`, and real ice `12-1858-52`. It also checks land exclusion, small
bright flecks, and nonuniform bright pixels. All fixtures are offline.

The viewer texture endpoint enables derived repair in depth-9 tiles
`230..233 / 6..7`, covering the supplied southern Greenland examples. A shared
depth-10 mosaic with a one-tile halo classifies connected defects across tile
boundaries. Every requested LOD uses that same approval map, refines the gap
footprint through connected bright neutral pixels, and requires current GTK50
ocean coverage. Missing reference imagery or coastline is left unclassified;
serving does not fetch provider data or write the database.

Repair uses the fixed sampled ocean RGB `(10,20,25)` across the trial to avoid
different fill colours per tile. Corrected textures are lossless PNGs served
from the existing `.jpg` route with `image/png`; unchanged images retain their
original bytes. The tile-composition digest and HTTP ETag both describe the
actual derived bytes. Trial responses revalidate their cache, and derived
caches invalidate when reference textures or coastline change. Reload an
already-open viewer once to clear its existing GPU textures.

`X-Tex-Repair: ocean-gap-v1` and `X-Tex-Repaired-Pixels` identify corrected
responses. Outside the trial bounds, textures are unchanged. The original
provider JPEGs remain in the database. The heuristic is not a provider NoData
mask and has not been enabled nationwide.

`Test.ocean_texture_serving_offline()` checks shared-boundary significance,
ancestor/exact LOD agreement, late coastline and neighbour arrival, cache/ETag
invalidation, untouched source bytes, and trial-area limits.

Build the original/detection/candidate-fill comparison with:

```bash
PYTHONPATH=python-server python-server/venv/bin/python \
  python-server/dynamic_functions/Terrain/Test/ocean_texture_preview.py /tmp/ocean-preview
```

Open `/tmp/ocean-preview/index.html`. The preview uses a sampled ocean colour
as an experimental fill. A bounded repair mask includes bright neutral JPEG
fringes within three pixels of the accepted core, constrained to known ocean;
all pixels outside that mask are preserved. The detector
requires at least one hectare (10,000 square metres in EPSG:3413) per connected
component, along with colour, uniformity, and pixel-count checks. Small
fragments in this standalone per-image preview can fall below that threshold.
The live trial instead classifies the shared reference mosaic before sampling
the requested tile.

## Viewer HTTP sidecar

`Server.start(host="127.0.0.1", port=5180)` explicitly starts the Terrain-owned
viewer compatibility server. It exposes `GET`/`POST /api/tiles` as raw
`binary-v1`, `/api/texture/<tile_id>.jpg` with exact/ancestor provenance, and
`/health`. A repeated start on the same bind is idempotent. The generic MCP
`server.py` is not modified and continues listening independently on port 8025.

`Server.stop()` explicitly shuts down the sidecar and releases its port; a
repeated stop is also idempotent. The existing viewer's default Vite proxy
already targets port 5180, so no proxy override is required for this layout.

`Server.status()` returns the current running state, bind address, URL, thread
state, and startup/runtime error without starting or stopping the sidecar.

## Asset catalog

`Asset.start()`, `Asset.stop()`, `Asset.status()`, and `Asset.list()` manage and
inspect the MCP-owned catalog at `Asset/assets.db`. `Asset.rebuild()` performs a
foreground-only rebuild into a temporary sibling database, validates it, and
atomically replaces the catalog. It does not launch a worker or server.
Start, stop, and successful rebuild calls refresh the Terrain dashboard, whose
`ASSET DB` indicator reflects the shared catalog connection state.

The rebuild reads model/seed metadata from `Asset/metadata.json`, the explicitly
selected Asiaq settlement archives under `Asset/grundkort/`, and the required
per-building terrain measurements in `Asset/building_ground_samples.json`.
Missing archives, metadata, or ground measurements fail the rebuild without
touching the active database. The configured ten-settlement scope reproduces
the migrated catalog exactly; the present `1900ILT` archive is not selected
because it had never been ingested into that catalog and has no migrated ground
measurements.

The viewer sidecar owns the complete runtime contract: `GET /api/assets`,
`GET /api/buildings`, `POST /api/vehicle_state`, and
`PATCH /api/asset/{asset_id}` all read or write this local catalog. There is no
legacy asset-server path or network fallback.
