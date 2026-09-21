# Lobster investor demo

The server owns movement, mission status, detection and component positions. In defense functions mode the AI/Lobster caller explicitly authorizes each engagement.
The browser displays those states. Accepted commands are not completed actions.
For parameterized destinations, task waiting, return legs and full state handling,
see [Terrain/Vehicles instructions](../Vehicles/README.md).
Control guides are available through `Terrain/instructions`; object interactions also enforce their access policy. Replace `YOUR_USERNAME` and UUID placeholders with values from your own account and asset registrations.

Use `/whoami`, then `@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Demo/briefing` to see current ownership
and the supported sequence. `@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Demo/status` reads live state.
The full paths below work from the root Lobster shell.

## Ground vehicle

AMV UUID: `GROUND_VEHICLE_UUID`.
Use `Terrain/Viewer/open` with that UUID to open the map and mission controls.
Select the AMV, choose its movement action, pick a map destination and press Run, or call:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/drive_to {"asset_id":"GROUND_VEHICLE_UUID","latitude":DESTINATION_LAT,"longitude":DESTINATION_LON,"request_id":"demo-drive-001"}
```

The controller prefers surveyed road centerlines in supplied terrain patches and
can reverse and detour over traversable ground. Water masks, slopes and padded
building footprints constrain the route. Unknown water coverage is blocked.
The server plans a complete destination route, then validates local movement and replans for blockage.
Wheel animation uses signed server travel, including reversing.

## Aircraft

Black Hornet UUID: `DRONE_UUID`.
Osprey UUID: `AIRCRAFT_UUID`.
These have simulated VTOL profiles. RQ-180 supports fixed-wing flyover/loiter; landing is not implemented. Boats expose water-only `sail_to`.

Register the existing catalog instance and attach its controller once:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/register {"terrain_asset_id":"black-hornet-01"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/attach {"asset_id":"DRONE_UUID"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Viewer/open {"asset_id":"DRONE_UUID"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Vehicles/takeoff {"asset_id":"DRONE_UUID","request_id":"demo-takeoff-001","altitude_agl_m":30}
```

Wait for `mission_status` to report completed before another mission. Then use
`fly_to` with latitude/longitude and `altitude_agl_m`, or `land` with a new request
ID to descend at the current location. Terrain clearance can block a command.
`mission_control` supports pause/resume/cancel by mission UUID. Aircraft hold their
server pose when paused; viewer extrapolation stops after 250 ms without updates.
The server supplies rotor RPM/angle and flight phase to the animated model.

## Layered defense

Read `Terrain/Defense/instructions` for the AI-readable procedure and display legend.
Set `Terrain/Defense/set_mode` to `functions`, then introduce a synthetic target with
`Terrain/Demo/defense_threat`. Poll `Terrain/Defense/observe`, choose a returned
eligible fictional site/layer option, and call `Terrain/Defense/intercept` with
its exact target/site/layer IDs. Observe events and the outcome; acceptance is not
success. There is no installed background AI: Lobster/the AI caller runs this loop.
The server retains detection, tracking and simulated engagement motion. Credits
are unrelated viewer-local construction currency and are not part of this flow.
See [the full defense guide](../Defense/README.md).

## Habitat

Airlock UUID returned by your registration: `AIRLOCK_UUID`.
Move an authorized player or owned vehicle within the configured interaction range.
Call `Terrain/Objects/functions` for the current runnable action and its bound
revision/subject fields, then:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Infrastructure/component_command {"asset_id":"AIRLOCK_UUID","action":"airlock_open_outer","expected_revision":CURRENT_REVISION,"subject_kind":"vehicle","subject_id":"NEARBY_OWNED_VEHICLE_UUID"}
```

Observe outer position reach 1. Close using `airlock_close` and the new revision;
wait until both doors reach 0 before `airlock_open_inner`. The server enforces
authorization, physical proximity and interlocks; the viewer uses the same door positions. This demonstrates access
components, not pressure/oxygen/power production or environmental control.

## Restart behavior

Active missions restore paused at their saved pose. Resume is explicit. The
viewer never restarts motion or door actions on its own. New request IDs mean new
missions; retries with the same request ID preserve the original mission UUID.
