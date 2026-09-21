# Infrastructure

Owner-visible strategic tools: `catalog`, `list_assets`, `place`, `move`, `remove`.
They call the existing supervised headless service, never schedule a tick.
Placements persist in room checkpoints and appear in the existing snapshot proxy.
The viewer consumes `snapshot.infrastructure` through a separate presentation adapter.

Coordinates are local ENU metres relative to the game origin. Models are authored
Y-up and converted at the viewer boundary. Heading zero maps model +Z forward to
ENU north; heading increases clockwise. Server ports and existing games are unchanged.

`source_vehicle_id` optionally binds a visual to an existing local-ENU logistics
vehicle. Position, heading, fuel, condition and maintenance status then come from
that vehicle, and independent placement moves are rejected. This does not create
a second vehicle, alter vehicle counts, or activate a weapon/network system.
Geodetic player-vehicle binding is rejected until its coordinate adapter exists.

Catalog artifacts are generated from the viewer registry with:
`node tools/audit-infrastructure.mjs` in the terrain checkout's webserver folder.
No renderer is imported by the server. Unverified motion, LOD and collision items
remain explicitly marked. The 5 new support/sensor designs are original game art,
not claimed manufacturer replicas.

Deployment: running Node processes need a supervised restart to load new routes;
use the owner-only `ArcticSimulation/reload_controllers` after package installation.
This preserves placements and viewer grants; active missions restore paused.
The Python dynamic-function files hot-load normally.

For the current live bank-owned placement and interaction flow, read
`Terrain/instructions(topic="infrastructure")`. Component aliases require
`subject_kind` and `subject_id` as well as the current revision; the server checks
bank identity, commissioned access policy and authoritative proximity. Prefer
`Terrain/Objects/functions` to discover currently runnable bound arguments.
