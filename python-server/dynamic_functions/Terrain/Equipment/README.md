# Asset controls

Use the original model IDs returned by `Terrain/Placement/assets`. A model ID
selects the authored model; the bank UUID identifies a placed instance. Multiple
instances keep independent state. No assets or upstream manifests are renamed.

## Discover and run

Before placement, call `Terrain/Equipment/model_controls(model_id)` to read the
original model's mechanism fields and optional movement profile. Enumerate model
IDs with `Terrain/Placement/assets`. This metadata is not permission to run an
action: instance discovery below checks current ownership, proximity and state.

1. Call `Terrain/Objects/functions(asset_id)` for actions available now, parameter
   fields, units, permitted ranges and bound revisions. The connected viewer uses
   the same discovery and invocation path.
2. Call `Terrain/Equipment/inspect(asset_id)` for the complete mechanism contract,
   targets and actual poses. Only send fields listed for that mechanism.
3. Call `set_controls(asset_id, mechanism_id, values, expected_revision)` using the
   returned revision and a dictionary of named values. A stale revision rejects;
   inspect again before deciding whether to retry.
4. Poll inspect. Acceptance sets a target; movement toward that target happens on
   server ticks. Actual pose, target and revision survive simulation persistence.

Examples of authored controls include light enablement, plow lift and angle,
recovery cable extension, landing ramps, fans, rotating joints and nested doors.
The contract gives metres, degrees, metres per second or revolutions per minute
where appropriate. Do not substitute percentages or normalized animation values.
Rates are simulation tuning unless separately documented as manufacturer data.
A moving mechanism does not imply simulated power production, cargo transfer,
recovery loads, decontamination efficacy or weapon operation.

Protected doors require a nearby authorized physical player or vehicle, passed
as `subject_kind` and `subject_id`. The viewer camera grants no access. Discovery
filters actions by proximity; the server rechecks permissions and interlocks
when executing. Close the other airlock door and wait before opening its partner.

## Movement for placed support vehicles

`attach_movement(asset_id, water_level_m=None)` binds an implemented movement
controller to the existing bank UUID. It supports the Sisu GTT, Patria TRACKX,
snowcat, support truck, landing craft and ice coaster models listed by discovery.
Static buildings and site-bound defense package components cannot attach.
For boats, supply the water surface elevation in the terrain vertical datum;
verified water coverage is required. This value positions the model origin at
that elevation; it is not a draft or buoyancy calculation.

Then use `Terrain/Vehicles/drive_to` or `sail_to`, `mission_status` and
`mission_control`. Read `Terrain/instructions("vehicles")` for coordinates,
auto-return and task completion. Routes use the existing server planner and
verified terrain/water data. The controller supplies the visible movement and
track/propeller motion. Direct track/propulsion animation commands are unavailable
once movement attaches, preventing two conflicting motion writers. Other
mechanisms remain controllable. Placement move/remove rejects attached vehicles;
use their movement commands. Detaching a movement controller is not exposed.

The seven original fleet instances retain their existing vehicle controllers.
These mechanism contracts describe the authored infrastructure/support models;
they do not claim that every original fleet model already exposes light or
articulation controls.

## Connected habitat workshop

`Terrain/Viewer/workshop_link(asset_id)` returns an authenticated
`infrastructure-preview.html` link for a placed, owned asset;
`Terrain/Viewer/open_workshop(asset_id)` opens it through MCP. The connected
workshop reads the same world state as Terrain and presents the same function
fields. Its model is a view of that UUID, not a second simulation. Opening the
unconnected model preview still shows local authoring controls.

Keep links private: they contain expiring session credentials. Never put live
UUIDs, links, credentials or saved world state into source documentation.

Map and workshop sessions renew while connected. Renewal preserves their scope
and rechecks authorization; an expired grant cannot be revived. Reopen through
the corresponding Viewer function if the browser was suspended past expiry.

## AI / Lobster sequence

Use the exact command prefix returned by `Terrain/Objects/functions`. Pass real
parameter values; do not use wildcard owner or server names.

1. Enumerate assets and read `model_controls` for the chosen original model ID.
2. Place it through `Terrain/Placement/place_model` with latitude, longitude,
   heading and a unique instance key. Retain the bank UUID returned by placement.
3. Discover actions on that UUID. Attach movement if offered and needed. Discover
   again after attaching; track/propeller animation then follows movement state.
4. Send a coordinate mission with `auto_return=True` and `wait_for_task=True`
   when work is required at the destination. Save the outbound mission ID.
5. Poll until `awaiting_task`. A blocked or paused mission needs investigation;
   neither counts as arrival. Discover the destination object's functions to
   obtain the nearby physical subject binding, permitted actions and revisions.
6. Run the selected action with the discovered parameters. Inspect actual
   mechanism or door positions until they reach the commanded target. Revisions
   detect stale commands; do not blindly retry a conflicting command.
7. Call `Terrain/Vehicles/complete_task` on the waiting vehicle's outbound
   mission. Follow the return leg until it reports `completed`.

Plow angles, recovery cable length, ramps, fans, pointing joints, tower lights
and doors use their model-specific contracts. Recovery cable animation does not
attach a load. Display-only defense models are distinct from operational demo
site components: use `Terrain/Defense/asset_status` for site state and available
synthetic interception actions. Static utility and housing models remain
placeable and inspectable, with no invented power, climate or occupancy state.
