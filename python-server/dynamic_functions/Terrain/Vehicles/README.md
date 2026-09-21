# Coordinate missions through dynamic_functions and Lobster

`Terrain/Vehicles` accepts destinations supplied by the caller. The destination is
not selected by a demo script. The same API supports a one-way journey, an immediate
round trip, or travel → perform a task → return. `Terrain/Demo` contains examples,
not the vehicle controller. Read `Terrain/Vehicles/instructions` from Lobster for
the machine-readable contract.

## What owns state

- The bank owns the canonical vehicle UUID and owner account. Catalog instance IDs
  such as `amv-01` identify the rendered asset; they are not mission IDs.
- The simulation server owns position, velocity, flight phase, rotor state and
  missions. It persists simulation steps and commands to SQLite. Closing the
  viewer does not end a mission.
- `dynamic_functions` checks ownership and sends parameterized commands to that
  server. Saving Python changes does not reload the installed simulation package.
- Lobster orchestrates functions for work at the destination. A function accepting
  a command does not prove that the commanded work has finished.
- The viewer animates authoritative snapshots. It does not decide arrival, task
  completion, or when to launch a return.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `asset_id` | Owned bank vehicle UUID, with a registered and attached controller |
| `latitude`, `longitude` | Destination in geographic degrees |
| `request_id` | Unique caller operation ID; reuse the same value and terms for retries |
| `return_latitude`, `return_longitude` | Optional explicit return destination; supply both or neither |
| `auto_return` | Default `false`. Capture the departure point once on dispatch and return there automatically. Cannot be combined with explicit return coordinates. |
| `wait_for_task` | Default `false`. When `true`, arrival enters `awaiting_task` until acknowledged |
| `altitude_agl_m` | `fly_to` only: destination altitude above verified terrain, 10–300 m, default 60 |
| `land` | `fly_to` only: land at the outbound destination instead of hovering; default `false` |
| `return_altitude_agl_m` | `fly_to` only: return altitude above destination terrain, default 60 |
| `return_land` | `fly_to` only: land on return; default `false` |

There is no hardcoded home location. Set `auto_return: true` to capture and return
to the departure point automatically, or supply explicit return coordinates.
The stored return coordinates are immutable terms of that dispatch, including
across retries and server restarts. The `return:` request-ID prefix is reserved
for the server's return legs.

## States and identity

Every new outbound mission receives an `id` and matching `journeyId`. If a return
is requested, the server creates a separate return mission with its own `id`,
`leg: "return"`, the same `journeyId`, and `parentMissionId` pointing to the outbound
mission. The completed outbound record has `returnMissionId` linking back.
`mission_status` returns the current `mission` and prior missions in `history`.

| State | Meaning and next step |
| --- | --- |
| `queued` | Accepted; waiting for verified terrain and a controller step |
| `running` | Vehicle is executing the current leg |
| `awaiting_task` | Arrived and holding; Lobster must perform/verify the work and call `complete_task` |
| `completed` | This leg finished. An outbound completion can queue a return; inspect the current leg |
| `blocked` | Terrain, clearance or progress prevented movement; inspect `reason`, resolve it, then resume or cancel |
| `paused` | Operator or restart paused movement; resume explicitly |
| `cancelled` | Operator ended this leg; no automatic return is launched |

`remainingM` is horizontal straight-line distance to the destination, not road
length. `travelledM` is measured horizontal travel during this leg. `reason`
explains arrival or interruption; `createdAt` and `updatedAt` are server epoch
milliseconds. Aircraft additionally report flight phase and rotor RPM in `observe`.
Ground arrival requires being within 1.5 m and stopped. Aircraft arrival additionally
checks altitude and low horizontal speed. Task completion is a separate explicit
acknowledgement; it is not inferred from an animation or a timer.

Flows:

```text
One way:         queued → running → completed
Immediate return: outbound completed → return queued → running → completed
Task and return:  queued → running → awaiting_task
                 → complete_task acknowledgement
                 → outbound completed → return queued → running → completed
```

A return is never triggered by `blocked`, `paused` or `cancelled`. A queued/running
leg restores **paused** after server restart. An awaiting task stays waiting;
the server does not re-run an external function. Resume the current mission ID,
not the old outbound ID after a return has begun.

## Lobster commands

Replace `YOUR_USERNAME` below with your authenticated account name. Replace `VEHICLE_UUID`,
`DEST_LAT`, `DEST_LON`, `HOME_LAT`, `HOME_LON`, and mission IDs before running.
Uppercase coordinate placeholders are intentionally not executable JSON numbers.
These are command templates, not a workflow that automatically polls for you.

For a new supported catalog vehicle, register and attach once. Existing attached
vehicles do not need re-registration:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/register {"terrain_asset_id":"CATALOG_INSTANCE_ID"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/attach {"asset_id":"VEHICLE_UUID"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/capabilities {"asset_id":"VEHICLE_UUID"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/observe {"asset_id":"VEHICLE_UUID"}
```

Drive to a supplied location and stop:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/drive_to {"asset_id":"VEHICLE_UUID","latitude":DEST_LAT,"longitude":DEST_LON,"request_id":"delivery-001"}
```

Drive there, perform a task, and return to supplied coordinates:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/drive_to {"asset_id":"VEHICLE_UUID","latitude":DEST_LAT,"longitude":DEST_LON,"request_id":"delivery-002","return_latitude":HOME_LAT,"return_longitude":HOME_LON,"wait_for_task":true}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/mission_status {"asset_id":"VEHICLE_UUID"}
```

Save the returned outbound mission ID. Poll `mission_status` until `awaiting_task`.
If it is blocked, paused or cancelled, handle that state instead of running the task.
Call the desired task function with its own asset UUID and arguments. Inspect that
function's actual state until the task is complete. For example, an airlock command
being accepted is not enough: inspect its component state until the requested door
position is reached. Vehicle arrival does not itself grant access to a habitat.
Then acknowledge the outbound task:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/complete_task {"asset_id":"VEHICLE_UUID","mission_id":"OUTBOUND_MISSION_UUID"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/mission_status {"asset_id":"VEHICLE_UUID"}
```

The current mission now has `leg: "return"`. Continue observing that return mission
until `completed`; do not report the whole round trip complete based on outbound
history. Retrying `complete_task` for the same completed task does not create
another return. `complete_task` is an owner acknowledgement, not proof independently
verified by the controller and not a generic task executor.

For an immediate round trip, use the same return coordinates and omit
`wait_for_task` (or set it to `false`). For work at the destination with no return,
set `wait_for_task: true` and omit both return coordinates.

Fly, wait for a task at the destination, then return and land:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/fly_to {"asset_id":"VEHICLE_UUID","latitude":DEST_LAT,"longitude":DEST_LON,"altitude_agl_m":60,"land":false,"request_id":"flight-001","return_latitude":HOME_LAT,"return_longitude":HOME_LON,"return_altitude_agl_m":60,"return_land":true,"wait_for_task":true}
```

Use the same observe → task → acknowledge → observe-return sequence. Aircraft
hold at their commanded destination altitude while awaiting a task; `land: true`
requests outbound touchdown instead. There is no automatic choice of a safe
landing site: destination clearance may block landing.

Pause/resume/cancel applies to the **current** mission UUID:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/mission_control {"asset_id":"VEHICLE_UUID","mission_id":"CURRENT_MISSION_UUID","action":"pause"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/mission_control {"asset_id":"VEHICLE_UUID","mission_id":"CURRENT_MISSION_UUID","action":"resume"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/mission_control {"asset_id":"VEHICLE_UUID","mission_id":"CURRENT_MISSION_UUID","action":"cancel"}
```

## Routing and current limits

Ground missions project the supplied coordinates into the vehicle's navigation
frame and compute a complete destination route before moving. A* minimizes
weighted travel distance through the dataset: surveyed `VEJMIDTE` roads cost less
than offroad ground, while buildings, water, unknown coverage and excessive grades
are impassable. No mission coordinates or preset routes are stored in the code.
The destination search includes both endpoints and a detour margin. It uses a
2-metre lattice for smaller trips, increasing spacing in even metres to bound
large searches to roughly one million cells. A route is optimal for those grid
costs within that search area, not a guarantee of a globally fastest driving time.
Missing routes fail explicitly instead of sending the vehicle toward a nearby
patch edge on an unconnected peninsula.

The computed `destinationRoute` and its progress cursor persist with the mission.
Detailed 512 × 512 metre patches guide steering along that corridor. Overlapping
patches use the same sampling lattice, so refreshing terrain does not change the
sampled surface underneath the vehicle. Local A* detours around supplied blockages
and rejoins the route; if that fails, `routeNeedsReplan` requests a new destination
route while the vehicle brakes. If the new route is unavailable, the mission
reports `blocked` with its reason. Detection currently uses the supplied terrain,
water and building data; moving-vehicle collision avoidance is not implemented.

The controller predicts steering and stopping against the same surface checks as
the simulation tick. When ordinary tracking cannot make a turn, a bounded search
over position, heading, steering and forward/reverse gear finds a short maneuver
using the AMV turning model. The selected maneuver persists across ticks; it is
invalidated when the local route changes. Terrain refreshes preserve a usable
local route and extend its verified end before the vehicle reaches it, rather
than treating each patch boundary as an arrival. Normal tracking drives forward;
reversing carries a planning cost and is reserved for shorter maneuvering paths. The controller can reverse to maneuver, but does not replace the route
with a straight-line heading to the destination. A valid detour can initially
increase `remainingM`, which is straight-line distance. Return legs compute a new
route to the supplied return coordinates.

A destination within the 20 km ground/boat limit is not guaranteed reachable.
Controllers support Patria AMV, Hrim, Black Hornet, Osprey, patrol boats and RQ-180.
Aircraft legs have a 2 km limit and require verified terrain/building clearance.
Fixed-wing landing remains unimplemented; see Additional fleet controllers below.
Return legs have the same distance limits.

Tasks such as inspection, delivery or habitat interaction must be implemented by
the relevant dynamic_function and orchestrated by Lobster. Naming an arbitrary task
in text does not cause the vehicle controller to execute it. The waiting state and
explicit completion acknowledgement connect those functions to travel without
prewiring a particular task or demo location into the controller.

## Fleet table

The authenticated viewer has a draggable, collapsible **My vehicles & drones**
menu. It lists only vehicles identified by your bank portfolio in the authenticated
catalog response. Other players' map vehicles are excluded. There is no owner
column; ownership labels remain on the map. Rows show state, outbound/return leg
and remaining distance. Select a row to see its UUID, mission ID, flight phase and
reason. Unattached vehicles are marked `not-attached`; stale updates are marked
explicitly instead of being presented as current state.

The corresponding owner-scoped terminal and structured views are:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/fleet_table
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/fleet
```

Each MCP call reads your current bank portfolio and a server snapshot. The terminal
table is a snapshot: run it again to refresh. The viewer updates automatically.

## AMV performance and terrain limits

The `patria-amv` controller reads its performance data from the simulation package's
`runtime/src/vehicle-performance.json`. The AMV 8×8 manufacturer's brochure specifies
maximum speed **over 100 km/h**, climbing grade **70%**, and side slope **40%**:
https://duro-dakovic.com/wp-content/uploads/2023/11/Patria_AMV_8x8.pdf
These are the AMV 8×8 figures, not the different AMV XP specification.

The simulation conservatively caps forward speed at 100 km/h (27.78 m/s). This
replaces the former 4 m/s commissioning cap. It reduces commanded speed for turns,
arrival braking, terrain hazards and the end of verified route coverage. The source
does not specify a universal offroad cruising speed. Reverse speed, acceleration,
braking, steering geometry and comfort settings remain explicit simulation tuning,
separate from the published manufacturer values in that configuration file.

Grades are rise/run percentages, not degrees. The planner and controller share
elevation and traversability checks, including points between grid vertices.
Driving checks longitudinal grade and cross-slope against the vehicle heading.
Predicted steering paths must also pass those checks before movement is accepted.
A mission reports `maneuver: planned-turn` while executing a checked steering
trajectory. If the bounded maneuver search cannot find one, it reports an explicit
blocked reason. A maneuver is not mission completion.
Restored ground missions adopt the current performance profile without changing
mission identity, destination, return destination or ownership.

## Viewer coordinates and ground contact

Vehicle and drone geographic positions use the terrain's EPSG:3413 projection and
active origin/offset. Restoring the camera frame also rebases vehicle positions.
Map-point and object targeting apply the inverse of that same transform. Heading
and slope normals account for grid convergence.

Ground vehicle meshes contact the displayed terrain LOD. This presentation height
can differ from the detailed DEM used by navigation; it does not modify server
elevation, collision checks, mission state or the saved pose. The mesh's
`userData.groundContact` records the displayed tile and both elevations. Aircraft
retain their authoritative flight altitude.

### Automatic return to departure

Add `"auto_return": true` to either `drive_to` or `fly_to`, or enable the
`auto_return` checkbox in the selected object's viewer action form. Do not enter
return latitude/longitude with this option. The departure point is captured by
the authoritative controller when it accepts the request, then persisted with
that mission; retries keep the original point even after the vehicle has moved.

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/drive_to {"asset_id":"VEHICLE_UUID","latitude":DEST_LAT,"longitude":DEST_LON,"request_id":"round-trip-001","auto_return":true}
```

Without `wait_for_task`, arrival starts the return leg immediately. With
`wait_for_task: true`, arrival enters `awaiting_task`; `complete_task` starts the
return after the task actually finishes. Aircraft also capture their departure
altitude and whether they were parked or hovering. Automatic return never queues
another automatic return from its own completion.

Arrival uses a 1.5-metre acceptance radius and requires the vehicle to stop.
`remainingM` retains the measured offset; it is not falsified to zero on arrival.
The viewer labels completed/awaiting-task distances as arrival offsets and shows
the tolerance in mission details.


### Additional fleet controllers

All controls use the same owner-checked MCP gateway and UUIDs as the viewer.
Click a vehicle, choose its coordinate action, fill the fields or select a map
point, then Run. The menu is generated from that vehicle's server capabilities.

- Both AMVs and AT-1 Hrim: `drive_to`. Hrim has its own simulation speed and
  steering profile; its procedural model has no claimed manufacturer specs.
- Black Hornet and Osprey: `fly_to`, with optional landing.
- Patrol boat: `sail_to`, using verified coastal/inland water masks. Land and
  unknown water coverage are impassable. `auto_return` and `wait_for_task` work
  as for ground missions. The configured water level is not a seabed DEM.
- RQ-180: `fly_to` means fly over the coordinate (25 m horizontal capture radius),
  then keep circling. It uses catalog fixed-wing stall/takeoff/climb limits,
  with cruise speed 1.5 times stall speed and a 45 degree bank limit as explicit
  autopilot simulation tuning, not verified manufacturer specifications.
  Ground departures validate a clear level takeoff roll. `takeoff_heading_deg`
  follows the viewer's heading convention: counterclockwise from north.
  Landing is not implemented and is not offered in the menu. `auto_return`
  requires an airborne departure; from the ground, use explicit return
  coordinates and an airborne return altitude instead. Completion means the
  flyover finished, not that the aircraft landed. Pause/restart freeze the
  simulation pose; resume is explicit. Unknown flight terrain blocks movement.

### Flight over water

`fly_to` accepts optional `water_level_m`, an explicit water-surface elevation in
the terrain vertical datum. Only cells positively classified as water use it;
unknown or missing land remains unavailable. Water landing is rejected. The
water reference persists through the outbound and return legs. Flight commands
have a 10 km per-leg simulation limit, separate from manufacturer aircraft range.
Use `wait_for_task=True` to hover at arrival, then `complete_task` to release
the automatic return.

### Boat route following and shoreline recovery

Boats follow a water-checked point ahead on the planned route instead of trying
to hit every grid vertex. The strategic water grid includes the exact departure
point to avoid rounding a shoreline position into a neighboring land cell.
When a forward turn is blocked, the controller searches bounded forward/reverse
trajectories using the model's available reverse speed. Each movement segment
still requires verified water. `maneuvering-in-verified-water` indicates recovery;
`awaiting-water-route-extension` requests the next local segment. A failed
trajectory search reports `no-feasible-water-maneuver` without changing the
boat's saved position.

Boat routes use a conservative circular hull envelope of half the asset's
`realLengthM`. The strategic and local planners reject centres closer to land
or unverified water than this envelope. Their weighted route cost prefers
clearance of twice the turning radius derived from the boat's speed and rudder
profile; hull-safe narrow passages remain usable when needed. Route smoothing
and steering lookahead preserve the available shoreline clearance instead of
cutting a planned open-water bend toward shore. This is a shoreline constraint
from the water mask, not a bathymetric depth or tide model. An unsafe endpoint
is rejected rather than silently moved. Boat route mode is reported as `water`.
