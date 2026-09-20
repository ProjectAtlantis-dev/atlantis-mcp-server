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
| `wait_for_task` | Default `false`. When `true`, arrival enters `awaiting_task` until acknowledged |
| `altitude_agl_m` | `fly_to` only: destination altitude above verified terrain, 10–300 m, default 60 |
| `land` | `fly_to` only: land at the outbound destination instead of hovering; default `false` |
| `return_altitude_agl_m` | `fly_to` only: return altitude above destination terrain, default 60 |
| `return_land` | `fly_to` only: land on return; default `false` |

There is no hardcoded home location. To return to the departure point, call
`observe` before dispatch and pass its `lat` and `lon` as the return coordinates.
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
%YOUR_USERNAME/**/Terrain/Vehicles/register {"terrain_asset_id":"CATALOG_INSTANCE_ID"}
%YOUR_USERNAME/**/Terrain/Vehicles/attach {"asset_id":"VEHICLE_UUID"}
%YOUR_USERNAME/**/Terrain/Vehicles/capabilities {"asset_id":"VEHICLE_UUID"}
%YOUR_USERNAME/**/Terrain/Vehicles/observe {"asset_id":"VEHICLE_UUID"}
```

Drive to a supplied location and stop:

```text
%YOUR_USERNAME/**/Terrain/Vehicles/drive_to {"asset_id":"VEHICLE_UUID","latitude":DEST_LAT,"longitude":DEST_LON,"request_id":"delivery-001"}
```

Drive there, perform a task, and return to supplied coordinates:

```text
%YOUR_USERNAME/**/Terrain/Vehicles/drive_to {"asset_id":"VEHICLE_UUID","latitude":DEST_LAT,"longitude":DEST_LON,"request_id":"delivery-002","return_latitude":HOME_LAT,"return_longitude":HOME_LON,"wait_for_task":true}
%YOUR_USERNAME/**/Terrain/Vehicles/mission_status {"asset_id":"VEHICLE_UUID"}
```

Save the returned outbound mission ID. Poll `mission_status` until `awaiting_task`.
If it is blocked, paused or cancelled, handle that state instead of running the task.
Call the desired task function with its own asset UUID and arguments. Inspect that
function's actual state until the task is complete. For example, an airlock command
being accepted is not enough: inspect its component state until the requested door
position is reached. Vehicle arrival does not itself grant access to a habitat.
Then acknowledge the outbound task:

```text
%YOUR_USERNAME/**/Terrain/Vehicles/complete_task {"asset_id":"VEHICLE_UUID","mission_id":"OUTBOUND_MISSION_UUID"}
%YOUR_USERNAME/**/Terrain/Vehicles/mission_status {"asset_id":"VEHICLE_UUID"}
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
%YOUR_USERNAME/**/Terrain/Vehicles/fly_to {"asset_id":"VEHICLE_UUID","latitude":DEST_LAT,"longitude":DEST_LON,"altitude_agl_m":60,"land":false,"request_id":"flight-001","return_latitude":HOME_LAT,"return_longitude":HOME_LON,"return_altitude_agl_m":60,"return_land":true,"wait_for_task":true}
```

Use the same observe → task → acknowledge → observe-return sequence. Aircraft
hold at their commanded destination altitude while awaiting a task; `land: true`
requests outbound touchdown instead. There is no automatic choice of a safe
landing site: destination clearance may block landing.

Pause/resume/cancel applies to the **current** mission UUID:

```text
%YOUR_USERNAME/**/Terrain/Vehicles/mission_control {"asset_id":"VEHICLE_UUID","mission_id":"CURRENT_MISSION_UUID","action":"pause"}
%YOUR_USERNAME/**/Terrain/Vehicles/mission_control {"asset_id":"VEHICLE_UUID","mission_id":"CURRENT_MISSION_UUID","action":"resume"}
%YOUR_USERNAME/**/Terrain/Vehicles/mission_control {"asset_id":"VEHICLE_UUID","mission_id":"CURRENT_MISSION_UUID","action":"cancel"}
```

## Routing and current limits

Ground missions project the supplied coordinates into the vehicle's navigation
frame, then compute routes from the terrain supplied around its current position.
A local A* search prefers surveyed `VEJMIDTE` road centerlines and can detour over
traversable offroad ground. Ground patches cover 256 × 256 metres at 2-metre
spacing, including roads and building footprints throughout that search area.
Building bounds already include vehicle clearance; additional turning room is
a route preference, not another forbidden zone. The controller can back along
an escape route and measures progress toward waypoints, so a detour may increase
`remainingM` without implying that it is stuck. The terrain worker refreshes patches
as the vehicle moves; the return route is computed again toward its supplied destination, not a
recorded animation played backwards. Building bounds, water masks, unknown water
coverage and excessive slopes can block a route.

This is local navigation, not a complete settlement-wide road graph. A destination
being within the 20 km ground limit does not guarantee a traversable route. Current
server controllers support Patria AMV, Black Hornet and Osprey VTOL. VTOL legs have
a 2 km limit and use verified terrain/building clearance; RQ-180 fixed-wing and boat
controllers remain unfinished. Return legs have the same distance limits.

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
%YOUR_USERNAME/**/Terrain/Vehicles/fleet_table
%YOUR_USERNAME/**/Terrain/Vehicles/fleet
```

Each MCP call reads your current bank portfolio and a server snapshot. The terminal
table is a snapshot: run it again to refresh. The viewer updates automatically.

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
