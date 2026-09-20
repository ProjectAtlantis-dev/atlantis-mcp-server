# Lobster investor demo

The server owns movement, mission status, defense decisions and component positions.
The browser displays those states. Accepted commands are not completed actions.
For parameterized destinations, task waiting, return legs and full state handling,
see [Terrain/Vehicles instructions](../Vehicles/README.md).
All tools remain owner-visible. Replace `YOUR_USERNAME` and UUID placeholders with values from your own account and asset registrations.

Use `/whoami`, then `%YOUR_USERNAME/**/Terrain/Demo/briefing` to see current ownership
and the supported sequence. `%YOUR_USERNAME/**/Terrain/Demo/status` reads live state.
The full paths below work from the root Lobster shell.

## Ground vehicle

AMV UUID: `GROUND_VEHICLE_UUID`.
Use `Terrain/Viewer/open` with that UUID to open the map and mission controls.
Select the AMV, choose its movement action, pick a map destination and press Run, or call:

```text
%YOUR_USERNAME/**/Terrain/Vehicles/drive_to {"asset_id":"GROUND_VEHICLE_UUID","latitude":DESTINATION_LAT,"longitude":DESTINATION_LON,"request_id":"demo-drive-001"}
```

The controller prefers surveyed road centerlines in supplied terrain patches and
can reverse and detour over traversable ground. Water masks, slopes and padded
building footprints constrain the route. Unknown water coverage is blocked.
This is local road-aware navigation, not a complete settlement-wide road graph.
Wheel animation uses signed server travel, including reversing.

## Aircraft

Black Hornet UUID: `DRONE_UUID`.
Osprey UUID: `AIRCRAFT_UUID`.
These have simulated VTOL profiles. RQ-180 fixed-wing runway control is unfinished.

Register the existing catalog instance and attach its controller once:

```text
%YOUR_USERNAME/**/Terrain/Vehicles/register {"terrain_asset_id":"black-hornet-01"}
%YOUR_USERNAME/**/Terrain/Vehicles/attach {"asset_id":"DRONE_UUID"}
%YOUR_USERNAME/**/Terrain/Viewer/open {"asset_id":"DRONE_UUID"}
%YOUR_USERNAME/**/Terrain/Vehicles/takeoff {"asset_id":"DRONE_UUID","request_id":"demo-takeoff-001","altitude_agl_m":30}
```

Wait for `mission_status` to report completed before another mission. Then use
`fly_to` with latitude/longitude and `altitude_agl_m`, or `land` with a new request
ID to descend at the current location. Terrain clearance can block a command.
`mission_control` supports pause/resume/cancel by mission UUID. Aircraft hold their
server pose when paused; viewer extrapolation stops after 250 ms without updates.
The server supplies rotor RPM/angle and flight phase to the animated model.

## Layered defense

`Terrain/Demo/defense_threat` introduces one fictional drone test target with a
unique `request_id`. The existing deployed site must have working sensors and
available inventory. `status` and `ArcticSimulation/events` show the result.
Use `ArcticSimulation/deploy_site` to construct a fresh demo site if needed.

The four layers are game roles: upper-tier ballistic, medium-range mixed-target,
point-defense, and close-range directed energy. Sensors form tracks; engagement
rules select eligible layers; ammunition/cooldown or continuous dwell constrain
the action. Outcomes are simulated, not scripted successes. Numeric parameters
are fictional game tuning. Support radar/launcher/logistics entities are separate
from the seven bank-owned catalog vehicles.

## Habitat

Airlock UUID returned by your registration: `AIRLOCK_UUID`.
Call `Terrain/Infrastructure/inspect` for the current component revision, then:

```text
%YOUR_USERNAME/**/Terrain/Infrastructure/component_command {"asset_id":"AIRLOCK_UUID","action":"airlock_open_outer","expected_revision":CURRENT_REVISION}
```

Observe outer position reach 1. Close using `airlock_close` and the new revision;
wait until both doors reach 0 before `airlock_open_inner`. The server enforces
interlocks and the viewer uses the same door positions. This demonstrates access
components, not pressure/oxygen/power production or environmental control.

## Restart behavior

Active missions restore paused at their saved pose. Resume is explicit. The
viewer never restarts motion or door actions on its own. New request IDs mean new
missions; retries with the same request ID preserve the original mission UUID.
