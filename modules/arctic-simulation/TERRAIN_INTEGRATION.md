# Existing vehicles on Terrain

The reusable simulation package now has a read-only Terrain adapter. It consumes
the existing multi-vehicle asset catalog and EGM2008 DEM tiles. It does not replace
the terrain renderer, create duplicate vehicle models, mint live ownership, or
move module implementation into the Terrain app.

## Host setup

Use the current official MCP checkout with the local extension hooks and install
this package with its `terrain` extra. Keep its dynamic-functions app linked as
documented in the host README. The installed Terrain HTTP app exposes the package
routes through `viewer_extensions`; a smaller host can mount the same routes.

Export these before starting the MCP process (database paths are selected at
module import; changing them requires stopping the existing DB connections):

- `TERRAIN_ASSET_DB_PATH`: the asset database used by Terrain itself.
- `TERRAIN_DB_PATH`: the terrain cache used by Terrain itself.
- `ATLANTIS_TERRAIN_BINDINGS`: the server-owned identity/source mapping below.
- `ATLANTIS_SIM_DB_PATH`: persistent simulation state, separate from source data.
- Existing authenticated identity and bank/world service configuration.

Terrain still uses its normal `Terrain/Server/start`, `.env`, imagery/geoid setup
and viewer proxy. Those requirements are not bypassed. Start the module using
`ArcticSimulation.start`; the existing host core tick is not patched. Its bounded
motion advances in the supervised simulation child, never in a tool call.

Before using historic data with a newer Terrain server, make consistent SQLite
backups and rehearse on copies: normal terrain acquisition/schema maintenance
can write to the configured databases. The module's catalog/DEM adapter itself
always opens its source databases read-only.

## Explicit binding, not automatic reassignment

```json
{
  "version": 1,
  "worlds": {
    "YOUR_APPROVED_WORLD": {
      "assetDatabase": "/absolute/path/to/assets.db",
      "terrainDatabase": "/absolute/path/to/terrain.db",
      "vehicles": {
        "EXISTING_OWNED_BANK_VEHICLE_UUID": {"terrainAssetId": "amv-01"}
      }
    }
  }
}
```

The bank UUID must be authentic, active, owned by the authenticated caller, have
the matching model type, and already be assigned to the simulation state owner.
An email alone is not that identity binding. No tool here issues live bank assets
or changes ownership. One terrain alias cannot map to two canonical UUIDs.

1. Call `terrain_vehicle_plan(asset_id)` to inspect existing pose, DEM provenance
   and height discrepancy without changing anything.
2. Call `terrain_vehicle_attach(asset_id)`. A saved-height difference above 2 m
   is rejected unless explicitly acknowledged with `allow_ground_snap=True`.
   Reattachment does not reset the authoritative pose.
3. Call `terrain_vehicle_viewer_access(asset_id)`. Open the **normal viewer root**
   with `#` followed by its returned `viewerFragment`. Do not log/share that token.
4. Select/enter the vehicle normally. Human controls and MCP `vehicle_claim`,
   `vehicle_drive`, `vehicle_release` use the same server-owned instance.

The normal startup loader requests `/api/simulation/WORLD/terrain-assets` with
the scoped capability. It receives the full existing catalog, replacing only
bound instance IDs/poses with authoritative UUID state. The original alias remains
as provenance, not a second rendered vehicle. The initial snapshot locks the
managed runtime to server authority before viewer-local driving begins. Existing
map selection and model-specific definitions remain in use.

Legacy catalog writes for mapped IDs/aliases are rejected through installed
`atlantis.asset_write_guards`, including the primary-vehicle save path. This
prevents an older connected viewer from competing for that vehicle's pose.

## Scope and verified result

- Adapter control currently commissions AMVs only. The other catalog vehicles
  still retain their existing model/control paths; they are NOT all converted to
  bank-backed server control by this change.
- Grounding uses a 40 m square, 2 m-spaced grid sampled from the deepest available
  cached EGM2008 tiles of depth 12 or finer. Missing/invalid/confidence-zero samples
  fail explicitly. It is not a flat test surface or invented zero height.
- Server pose includes a terrain normal for slope orientation. Motion stops at
  the sampled boundary. Streaming collision surfaces, obstacles, realistic
  suspension/contact physics and unrestricted driving remain unfinished.
- This reader samples canonical DEM, not the renderer's derived coastline/water
  mesh. Agreement with rendered ground, legacy terrain seams and grounding
  discrepancies require separate validation before deployment.
- On the preserved data, `amv-01` saved z is 16.279 m; sampled ground is about
  5.018 m. Do not silently treat those as equivalent or overwrite the saved record.

Verification used the actual Terrain HTTP app, copies of the real Nuuk databases,
the normal viewer index, seven existing models and native Brave WebGL. The AMV
was driven through the server; old pose writes were rejected. Bank ownership used
a clearly isolated fixture UUID, not a live user's account migration.

Reproduce from the workspace with the installed staging environment:

```sh
PYTHONPATH=/path/to/atlantis-mcp-server/python-server \
atlantis-modularization/.venv/bin/python \
atlantis-modularization/tools/check_stack.py --terrain-vehicles --terrain-browser
```

Evidence lives in `atlantis-modularization/evidence/terrain-vehicles/`. The browser
test uses DB backups because the normal Terrain app can acquire/update data.
Temporary services stop after the check. No persistent preview URL is implied.

The screenshot is control-integration evidence, not a terrain-quality acceptance:
historic cache/rendering issues and unavailable acquisition credentials remain.
Real rollout still requires verified user/game/ownership bindings and agreement
on the altitude discrepancy. Redis delivery and the other vehicle classes remain
separate unfinished work.
