# Physical players and resident entrance access

This module reuses `game_player.id` from the existing Greenland world registry.
Host identity bindings resolve the authenticated external user; callers cannot
submit another user's identity or a physical position. No player UUID is minted
here. `player_attach` reconnects to persisted position, not the initial spawn.

## Operator configuration

Set `ATLANTIS_PLAYER_DEPLOYMENTS` before starting the simulation child. The file
is trusted server configuration, not an MCP argument. Example structure (replace
placeholders with existing registry/placed-instance UUIDs):

```json
{
  "version": 1,
  "worlds": {
    "your-world": {
      "players": {
        "EXISTING_PLAYER_UUID": {
          "zoneId": "exterior-walkway",
          "spawn": {"x": 0, "y": 6, "z": 0.3}
        }
      },
      "zones": {
        "exterior-walkway": {
          "minX": -10, "maxX": 10,
          "minY": 5.5, "maxY": 8, "floorZ": 0.3
        }
      },
      "airlocks": {
        "EXISTING_AIRLOCK_UUID": {
          "outerOffset": {"x": 0, "y": 4.5, "z": 0.3},
          "radiusM": 2,
          "allowedPlayerIds": ["EXISTING_PLAYER_UUID"]
        }
      }
    }
  }
}
```

Coordinates are metres in the simulation room's local ENU frame, referenced to
its geodetic origin. Zones are explicitly commissioned flat pedestrian surfaces,
not terrain meshes or automatic navmesh extraction. They must exclude obstacles.
The offset is from the placed airlock origin in ENU at heading zero; clockwise
heading rotates it. The example matches the standalone pedestrian model's outer
door; it is not a universal offset for every model. Maximum entry radius is 3 m.
Policy changes apply to subsequent movement/access checks. Changing a zone does
not teleport a player: incompatible placements require operator reconciliation.

Provision existing players and place the desired airlocks before enabling this
configuration. Protected entrances cannot be moved/deleted or directly operated
through the generic infrastructure tools. Resetting a commissioned player world
is refused. There is no resident-facing ACL-editing or unrestricted pose API.

## MCP operations

1. `player_attach()` then `player_claim()` returns an exclusive lease.
2. `player_walk(lease_id, sequence, east, north, duration_ms)` supplies a unit-length
   or shorter direction vector. Walking is bounded to 2 m/s, for 50–1000 ms.
   A zero vector renews presence without movement. Sequences must increase.
3. `player_observe()` reads authoritative position, revision and presence.
4. `player_request_entry(airlock_id, lease_id, expected_revision)` checks the
   configured player ACL, presence younger than 3 seconds, entrance proximity,
   lease and door interlock. It requests **outer-door opening only**.
5. `player_entry_action()` reports the latest action. `running` is not open;
   `succeeded` means actual outer position reached its target, not completed passage.
6. `player_release()` stops motion and immediately invalidates presence.

The server advances between calls. Read operations never refresh presence.
No control renewal means motion expires; an expired or competing lease is denied.
The residency list is explicitly configured per entrance; automated residency
assignment from housing contracts is not implemented.

## Human viewer

`player_viewer_access()` returns a short-lived capability. Open the existing
remote-authority terrain viewer with this fragment (substitute the returned data):

`#simulation_game=WORLD&simulation_token=TOKEN&simulation_player=PLAYER_UUID`

The fragment is removed immediately and is not saved in localStorage. The
physical-player panel has Enter, Release, held cardinal-direction buttons,
airlock selection, Request entry and Action status. It sends the same server
commands used by MCP. Spectator camera movement does not move the player.
The player is currently shown as a clearly basic capsule marker, not a finished
character model. A separate `player-control-review.html` uses the same panel and
actual authored airlock model for commissioning. It is not a running server by itself.

## Persistence, transport and limits

Player pose and action state are included in the simulation's existing SQLite
room checkpoint. This reuses the position authority; it does not create another
bank or Redis-owned position writer. Snapshots include player UUID, pose and
freshness, without exposing the external user ID or ACL. They currently reach
this viewer through the authenticated snapshot route. Connecting this simulation
stream to the dedicated Redis relay remains separate work; no Postiz Redis is used.

Restart preserves pose but clears held inputs, leases and fresh presence. An
in-flight entry action becomes interrupted. Configured airlock mechanisms pause
at their saved positions until a fresh authorized request reconciles the action.

This is game software, **not a physical access/life-safety controller**. Full
chamber occupancy, obstructions, pressure equalization, emergency egress,
automatic closing/passage cycles, cross-zone navigation, terrain collision,
vehicle boarding and production deployment are not implemented by this change.
Do not interpret a successful outer-door operation as any of those guarantees.

## Reproducible isolated verification

From `/path/to/checkouts`:

```sh
PYTHONPATH=/path/to/atlantis-mcp-server/python-server \
atlantis-modularization/.venv/bin/python \
atlantis-modularization/tools/check_stack.py --players --player-browser
```

Uses temporary bank/world/simulation databases and the real MCP loader. Browser
verification uses native Brave, without software-GPU or bypass flags. Evidence
is saved under `atlantis-modularization/evidence/player-presence/`. This command
does not migrate real identities, start a permanent preview, or cut over live data.
