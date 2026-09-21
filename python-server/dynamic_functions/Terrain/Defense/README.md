# Fictional defense demo — dynamic function procedure

This is a virtual game controller. It uses synthetic targets and fictional layer rules, not live sensors, real weapon performance or external weapon control. Model appearance does not establish operational fidelity. The existing four layer IDs are game options; do not infer real-world deployment advice from them.

## Normal scenario: choose the destination, type and direction

In the connected viewer, click **Send test incoming** in the defense panel. Select
`drone`, `cruise` or `ballistic`, enter the approach heading, and either type the
latitude/longitude or use **Pick destination on map** to select terrain/an object.
Run submits the same command as MCP. Picking an object uses its current coordinates;
it does not track that object's later movement.

The AI can launch directly with coordinates, without any viewer interaction:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/spawn_incoming {"incoming_type":"drone","request_id":"scenario-001","latitude":DESTINATION_LAT,"longitude":DESTINATION_LON,"heading_deg":270,"approach_distance_m":5000,"altitude_m":300,"speed_mps":70}
```

Heading is travel direction, clockwise from north: 0 north, 90 east, 180 south,
270 west. A westbound incoming starts east of the selected destination.
`approach_distance_m` controls the horizontal starting offset; `altitude_m` is the
starting height above the destination's verified terrain elevation. The synthetic
trajectory travels to that terrain point. `speed_mps` is a test parameter, not
manufacturer performance. An unknown terrain elevation is rejected, not invented.

This normal tool has no site/layer selector. Radar detection depends on the chosen
path and operational sensor coverage. Poll `alerts`, then call `intercept` using
an observed eligible action. Its `target_id` identifies the incoming, not the
object picked as its destination. Arrival is a simulation outcome; this tool does
not apply damage to registered assets. The separate `spawn_layer_test` tool below
provides controlled test fixtures for each configured game layer.

## Discover and run

Use the exact owner/remote path learned from the current server. Examples use placeholders:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/instructions
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/set_mode {"mode":"functions"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/observe
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/spawn_incoming {"incoming_type":"drone","request_id":"demo-001","latitude":DESTINATION_LAT,"longitude":DESTINATION_LON,"heading_deg":270}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/tracks_table
```

The simulation performs detection and track formation continuously. In `functions` mode it does not launch on its own. The AI/Lobster orchestrator polls `observe`; there is no installed background AI or detection callback. The caller must keep the observation/action loop running.

1. Save `lastEventSequence` from observe before introducing a target.
2. Poll observe until the desired synthetic target has `state=tracked`. Use returned IDs, never coordinates or identities guessed from model labels.
3. `availableActions` lists the existing game's currently eligible site/layer combinations. The caller chooses explicitly among these fictional options; this tool does not choose a real-world interception method. Empty means no runnable action now. Inspect layer readiness, inventory and cooldown; do not invent an eligible choice.
4. Call with the exact observed IDs:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/intercept {"target_id":"OBSERVED_TARGET_ID","site_id":"OBSERVED_SITE_ID","layer_id":"OBSERVED_LAYER_ID"}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/events {"after_sequence":SAVED_EVENT_SEQUENCE}
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/observe
```

5. `accepted=true` means an engagement was created, not that interception succeeded. Observe actual events and target outcome. An engagement can miss and a target can reach its destination. No success is scripted.
6. After each events call, retain the greatest returned sequence. Re-observe after an unavailable-layer or stale-track rejection. Do not repeatedly launch while a target is already engaged. Duplicate active engagement requests are rejected. Untracked requests through this tool are rejected rather than queued.

`set_mode("automatic")` explicitly restores the old game-rule automation. Switching to functions clears queued, unlaunched orders but does not cancel an already active engagement. This changes only defense control mode; it does not reset the world, fleet or habitat.

## What the display means

- Active targets: synthetic targets still travelling.
- Tracked: active targets whose simulated track formation is complete.
- Engagements: actions currently resolving on the server.
- Intercepted / reached destination: cumulative simulated outcomes, not current target counts.
- `upper-tier`, `middle-tier`, `point-defense`: game layer identifiers. The number beside each is remaining simulated shots, not a unit ID, price, vehicle speed or success probability.
- `directed-energy`: game channel readiness/cooldown; its old `1` was a capacity flag, not ammunition.
- Numbers such as 620/440/290/530 on old construction buttons were viewer-only game prices. Credits never funded server-side interceptions. The connected demo no longer displays this unrelated economy.
- Sensor/launcher/logistics entities belonging to defense sites are separate from registered catalog vehicles. Microwave and other visual-only assets do not acquire working controllers merely by being rendered.

## Function contracts

`instructions()` returns this guide to the AI. `set_mode(mode)` persists the control mode. `observe()` returns detected tracks, available game actions, layer state, active engagements, statistics and an event cursor. `tracks_table()` returns rows for Lobster's normal table rendering, following Demo/myTable. `intercept(target_id, site_id, layer_id)` revalidates current state and launches at most one active engagement for that target. `events(after_sequence=0)` reads the event history.

Tools operate on the authenticated caller's current simulation world and are owner-visible. Older `ArcticSimulation/intercept` remains a low-level compatibility tool that can queue authorization before tracking; prefer this Terrain procedure for the demo. Do not reset the live world to change defense mode.

## Test incoming types and radar observation

Call `Terrain/Defense/test_cases` to list the configured layer IDs and compatible
synthetic incoming types. For example, using IDs returned by that tool:

```text
@/YOUR_USERNAME/YOUR_REMOTE/Terrain/Defense/spawn_layer_test {"incoming_type":"drone","request_id":"layer-test-001","site_id":"OBSERVED_SITE_ID","test_layer_id":"OBSERVED_LAYER_ID","heading_deg":270}
```

Supported incoming labels are `drone`, `cruise` and `ballistic`. Choose a compatible
label per returned case to exercise each layer. `test_layer_id` selects a test
fixture placement, not an interceptor or an activation. Placement is derived from
the current configured game bounds, not a hardwired map point. Motion is a simple
120-second synthetic fixture, including the ballistic label; it is deliberately
not a realistic flight/trajectory model. The cruise test currently shares the
drone visualization. Same request ID and terms recover the existing target;
changed terms require a new request ID. A completed target is not respawned by retry.

`observe.sensors` identifies the actual simulation radar entities, their sites,
readiness and configured game coverage. `tracks[].detectedBy` identifies which
radar/site detected a track. An unavailable radar cannot form a track. The event
stream emits `target-detected`, `target-tracked` and `target-track-lost`; these can
be consumed by the AI/Lobster polling loop. The type is the synthetic scenario
label, not a real radar classifier. Once tracked, the AI sees current game choices
in `availableActions` and must call `intercept` separately. The HUD shows radar
status and detected incoming types as well as the overall counts.

### Incoming direction

`spawn_layer_test(..., heading_deg=90)` chooses the direction the synthetic target
travels: 0 north, 90 east, 180 south, 270 west (clockwise from north, in [0,360)).
Thus heading 270 travels west and starts east of the selected test site. The
fixture position is derived from that bearing and current game configuration.
`observe.tracks[].kind` and `headingDeg`, `tracks_table`, and the viewer status
show the requested type/direction. Historical targets without a recorded test
heading report null rather than an invented heading. Changing type or direction
requires a new request ID. Observe available actions, then explicitly call
`intercept(target_id, site_id, layer_id)`; spawning alone does not activate it.

## Radar alerts for the AI

`Terrain/Defense/alerts(after_sequence=0)` returns dedicated radar alert rows:
`incoming_type`, `heading_deg`, `target_id`, `detected_by`, `current_state`,
`ready_for_action` and `available_actions`. Keep `next_sequence` and pass it to the
next poll. Each row identifies a detection, completed track or lost track event;
its runnable choices are re-read from current state, not copied from an old event.
When `ready_for_action` is true, the AI can select a returned fictional game option
and explicitly call `intercept`. Already-engaged or lost tracks are not actionable.
The viewer labels unengaged detections “RADAR ALERT” and shows type, heading and ID.
This is a polling feed for the AI/Lobster caller; it does not install a background
agent or send an unsolicited message to another user. Historical lost-track events
may not contain type/direction and report null rather than guessing.
