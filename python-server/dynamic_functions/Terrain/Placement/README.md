# Scene placement

Open **Place assets** in the connected viewer. Choose an action, fill its parameters,
pick a terrain point or type latitude/longitude, then Run. The viewer invokes the
same owner-authorized dispatcher as MCP. The panel supplies a unique instance/site
key; keep that key when retrying the same model placement.

## Asset names and model locations

`Terrain/Placement/assets` returns every authored model's ID, bounds in metres,
motion metadata and export path. The viewer source is `infrastructure-registry.ts`;
its constructors live under `buildings/`, `defense/`, and `communications/`.
`public/infrastructure-catalog.json` is the exported manifest, with GLBs under
`public/infrastructure-models/`. Runtime scene placement uses the procedural
registry, not the GLB export. An absent export is not an absent procedural model.

These are the original repository names. No model IDs or asset directories are
renamed. `Terrain/Placement/catalog` is retained as a compatibility command for
earlier clients; new commands and documentation use `Terrain/Placement/assets`.

The asset list includes habitat, utility, communications, support vehicle and defense
models. Placement registers a bank-owned instance. Supported ground vehicles attach their
movement controller automatically under that same UUID and immediately expose
Terrain/Vehicles orders. Water vehicles require an explicit water elevation through
Enable movement controller. Read Terrain/Equipment/instructions for supported families
and boat water elevation requirements.

## Two distinct placement actions

`place_model(model_id, instance_key, latitude, longitude, heading_deg=0,
height_offset_m=0)` registers an owned bank UUID and places the model. It samples
verified terrain height and aligns the authored bottom bound with that height.
The optional height offset is explicit; terrain is not flattened or filled.
Identical retries reuse the identity; conflicting terms reject. A placed asset
appears under My infrastructure & habitat. Selecting it exposes implemented
functions, including the model-specific mechanisms reported by
`Terrain/Equipment/inspect`. Server state drives their visible poses.

All six defense display models are available: `defense-upper-tier`,
`defense-middle-tier`, `defense-point-defense`, `defense-laser`,
`defense-microwave`, and `defense-radar`. Placing these models does not activate
an interceptor or sensor.

`deploy_demo_site(site_id, layer_id, latitude, longitude)` creates the existing
fictional simulation package: radar, the selected layer, and support entities.
Choose `upper-tier`, `middle-tier`, `point-defense`, or `directed-energy`.
This uses the game's current default construction duration, layout and tuning.
A new site ID creates another independent package of the same type. Each radar,
launcher, command, resupply and recovery component receives its own bank UUID
and owner. The site ID groups those assets; it is not their asset identity.
Identical retries reuse the same bank UUIDs and site. Changed terms reject.
If bank issuance fails partway, no site is deployed; retry the same terms to
complete issuance without duplicating already issued assets. Bank-linked components
appear in My infrastructure & habitat and Terrain/Infrastructure/fleet.
There is no functional microwave controller. Deployment does not change defense mode or authorize a shot.

## Investor demo recipe

1. Build the visible scene first: place a habitat/airlock, a utility model, and
   support equipment using asset IDs. Keep the authored scale; use the asset
   bounds to review footprints. Check the model from above and ground level for
   intersections and terrain clipping. Placement currently samples one anchor;
   it does not certify a foundation, grade the ground or detect every collision.
2. For an exhibit of the models, place individual defense display models. For a
   working game demonstration, deploy one demo site at a user-chosen clear point.
   Do not count a display model as a second operational system.
3. Read `Terrain/Defense/observe` until the site's radar and layer report ready.
   Use `set_mode("functions")` for an explicit AI/tool demonstration.
4. Launch a synthetic test using `Terrain/Defense/spawn_incoming` with a picked
   destination or coordinates. Poll `alerts`; inspect the incoming type and the
   available game actions. Call `intercept` explicitly, then check the outcome.
5. Show habitat door transitions separately through a nearby authorized subject.
   Utilities, power distribution, resupply and damage are not made functional
   merely by placing their models.

This recipe demonstrates implemented software behavior and visual assembly.
It does not prescribe real-world weapon placement, optimized coverage, spacing,
radar propagation or combat performance. No real-world deployment model is
implemented. The existing site layout is a game template, not an engineering plan.

## MCP examples

Use the exact owner/remote reported by the server; substitute coordinates and IDs:

```text
@/OWNER/REMOTE/Terrain/Placement/assets
@/OWNER/REMOTE/Terrain/Placement/place_model {"model_id":"future-airlock-pedestrian","instance_key":"exhibit-entry-1","latitude":LAT,"longitude":LON,"heading_deg":0}
@/OWNER/REMOTE/Terrain/Placement/deploy_demo_site {"site_id":"exhibit-site-1","layer_id":"point-defense","latitude":LAT,"longitude":LON}
@/OWNER/REMOTE/Terrain/Defense/observe
```

Asset placements can be moved with `Terrain/Infrastructure/move` using
local ENU metres. Operational site relocation/removal is not exposed by this
panel; choose its location before deployment.

## Open the map through a function

`Terrain/Viewer/open_map()` opens the authenticated map in the calling terminal.
`Terrain/Viewer/map_link()` returns the equivalent browser URL. Both accept an
optional owned `asset_id` and `ttl_seconds`; a vehicle is not required for the map.
These are the same world and viewer as 3D mode. Returned URLs contain temporary
access tokens and belong in the session, not source code or shared documentation.

## Multiple copies and identity

Use the same model_id with a new instance_key for each copy. The viewer generates
that request key when opening a new placement form. The request key is not the
asset UUID: only the bank issues the asset identity. Keep the same key for a retry.
For defense packages use a new site_id per package. Never use a model type as a
unique instance identity. Older demo sites created before bank integration remain
legacy simulation entities; they are not silently claimed or reissued by this tool.
