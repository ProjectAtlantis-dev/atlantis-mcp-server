# Infrastructure and habitat access

The viewer renders placed infrastructure from simulation snapshots. Open the fleet panel and select an entry under “My infrastructure & habitat” to inspect it. This table includes only structures in the caller's bank portfolio. World placements can still be visible to other viewers.

Component actions are discovered through `Terrain/Objects/functions` and executed through `Terrain/Objects/call`, using the same gateway as `Terrain/Infrastructure/component_command`. The viewer binds a specific physical player or owned vehicle to each offered action. Its form and copied MCP command include that subject and the current component revision. Use the exact owner and remote server prefix returned by discovery; do not use wildcard routing.

New placements get an owner-only policy with a 5 metre interaction radius, measured from the authored structure bounds. Existing placements must be commissioned through `Terrain/Infrastructure/configure_access`. This owner tool accepts `asset_id`, `interaction_radius_m`, and `allowed_account_ids`; the supplied account list replaces additional grants and the owner remains permitted. Authoring placement/movement is an owner operation, not a proximity-limited in-world action.

Protected actions appear only when an authorized physical subject is nearby. Camera movement is not physical presence. Player presence must have a fresh control lease; vehicles must be owned by the caller and have an authoritative simulation position. Discovery refreshes while a structure is selected. If no actor qualifies, Inspect remains available with an explanation.

The top-level `terrain_access_authorized` predicate authenticates the caller for protected dynamic functions. It does not implement geometry checks: the shared Python gateway validates bank identities and the simulation rechecks the policy and live subject position when executing each command. A stale viewer form or direct MCP call cannot bypass these checks. Revision and door interlocks are also checked at execution.

The simulation tick stops unauthorized walking players and controlled vehicles from crossing a protected structure volume, including a segment that would cross it in one tick. Denial is reported as `access-denied:<structure UUID>`. A subject already inside after access revocation can move toward an exit. These authored bounding volumes protect entry; they are not a full architectural collision mesh or doorway navigation system. Door state and account access are separate checks.

Airlock entry and facility freight animations follow server component positions, not command acceptance. The current controllers simulate doors and their interlocks. Utility supply, habitat climate and occupancy are not simulated and visual-only models expose no operational actions.

`Terrain/Infrastructure/fleet` returns structured owner rows; `fleet_table` returns the terminal table. Both report observed component state, not invented utility measurements.

## Live verification

Verified against the running terrain-sim host and its Greenland viewer on 2026-09-21:

- Existing registered airlock appears in the owner table and as an actual scene rig.
- A direct MCP component command from a distant owned vehicle is rejected for distance.
- An owned Hrim receives a coordinate mission with `wait_for_task` and `auto_return`.
- As Hrim arrives, the already-open object panel adds actions bound to that vehicle without a page reload.
- Run on Close changes the server target, then the observed door position and viewer leaf geometry reach closed. Run on Open Outer restores the initial pose.
- Completing the task dispatches the return leg to the captured departure coordinate. Hrim completed it with a 1.49 metre arrival offset, within its 1.5 metre tolerance. Door actions disappeared again on departure.

Unauthorized walking/vehicle crossings, stale player presence, revoked grants, raw endpoint bypass, and denied aircraft task arrival are covered by simulation regression tests. A live second-account intrusion has not been exercised. The focused viewer and Python gateway tests pass; a broad viewer test run stalled in the existing terrain-buildings-runtime test and was stopped, so it is not reported as a full-suite pass.
