# Terrain

Greenland terrain acquisition, local storage, and viewer serving. Terrain owns
its SQLite data and HTTP sidecar; the Atlantis MCP host runs independently.

## Setup

1. Create the git-ignored `Terrain/.env` with `DATAFORSYNINGEN_TOKEN`.
   The file must exist even if the token is already in the environment.
   Add `DATAFORSYNINGEN_FTP_USER` and `DATAFORSYNINGEN_FTP_PASS` for GTK50 downloads.
2. Install the EGM2008 geoid grid in the Python environment running Atlantis:
   `projsync --file us_nga_egm08_25.tif`.
3. Call `Terrain/Server/start` through Atlantis. The default bind is
   `127.0.0.1:5180`; `status` and `stop` live in the same folder.

Each start reloads `.env`, overriding matching process environment values,
and requires a nonblank token even when imagery is cached. The sidecar serves
the viewer API; configure the viewer's proxy separately if its target differs.
Check `/health` for HTTP availability and `/api/demand-status` for acquisition
progress. Server logs are in `python-server/runServer.log`.

## Data guarantees

- Persisted DEM elevations are EGM2008 orthometric. ArcticDEM requires the
  real geoid transformation; a missing grid fails that candidate. Copernicus
  can still supply the tile. Provider failures remain visible.
- Ready-data composition does not acquire missing data. Camera demand and
  texture HTTP requests can queue work; partial coverage is normal while
  independent domains converge. Repeated camera requests advance acquisition
  and eligible retries. Transient failures can be reclaimed after cooldown;
  invalid data and configuration errors require intervention.
- Acquisition dates describe source evidence, not cache freshness. Unknown
  dates stay unknown, and older cached rows are not automatically backfilled.
- Ocean-gap repair is derived imagery, enabled beyond the former trial area.
  Serving may persist a separate repair cache; original provider images stay
  intact. Repairs depend on available coastline and imagery evidence and are
  a heuristic, not a provider NoData mask. Corrected `.jpg` responses contain
  PNG bytes: honor `Content-Type`. `X-Tex-Repair` identifies the repair version.

Live terrain and asset databases are untracked. Asset rebuilds require local
source archives, metadata, and measured building-ground samples; an existing
catalog alone is not a complete rebuild source.

## Verification and details

`Terrain/Test/terrain_regression` runs the main offline suite and raises on a
failed check. It includes fixture decoding, persistence, composition, demand,
bathymetry, texture repair, and sidecar lifecycle checks. Some persistence
checks use rollback savepoints on the current database, so use an isolated
test database/process for a full run. Offline does not mean dependency-free:
the geoid grid is still required, and the sidecar test binds a loopback port.

`camera_prefill_offline` and `asset_coordinate_loading_offline` in the same
Test folder are separate checks, not included in that runner.

See [bathymetry notes](Bathymetry/README.md) for regional generation,
[HTTP request parsing](http_adapter.py) for the viewer wire contract, and
[regression gates](Test/regression.py) for the suite's actual coverage.
Use Atlantis tool help for current parameters.
