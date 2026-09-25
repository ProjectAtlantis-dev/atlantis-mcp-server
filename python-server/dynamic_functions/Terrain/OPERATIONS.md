# Start, use, and recover the simulation demo

This runbook describes the existing local demo installation. It does not require
an AI. Run OS commands in a local shell; run `@/...` and `/remote` commands in the
logged-in Atlantis terminal. Do not paste Atlantis commands into zsh/bash.
`python-server/lobster.py` implements the MCP `command` and `chat` tools; it is not
a standalone shell launcher. MCP clients can send the same Atlantis commands
through `command` with a `commandText` argument.

## Services and persistent data

| Service | Local demo port | How to verify |
| --- | --- | --- |
| MCP host | 8002 | Atlantis remote `terrain-sim` connected |
| Terrain/asset HTTP API | 5184 | `http://127.0.0.1:5184/health` |
| Simulation authority | 5190 | `ArcticSimulation/status` |
| Persistent bank | 5192 | `Terrain/Economy/Server/status` |
| Viewer development server | 5273 | Browser or HTTP GET `/` |

5190 and 5192 are service endpoints, not the demo website. An unauthenticated
request may correctly return 401. Verify the bank through its authenticated MCP
status tool. Port numbers here describe this installation, not mandatory defaults.
The terrain tool defaults to 5180; explicitly pass 5184 for this viewer setup.

The repository root contains `runtime-data/simulation.sqlite`, `bank.sqlite`,
`terrain-bindings.json`, and `identity-bindings.json`. Terrain's configured SQLite
files hold terrain and asset data. `Chat/Data/` holds saved chat games. Preserve
these files. Do not reset, enroll replacement accounts, rebuild the asset catalog,
issue replacement assets, or delete SQLite/WAL files to recover a stopped service.

## 1. Start the MCP host

From the host repository root, in a dedicated OS terminal:

```sh
cd python-server
sh ./runServerArcticSim >> runServer.log 2>&1
```

Leave that terminal running. The launcher is a local, git-ignored installation
file and currently is not executable; invoke it with `sh`. It sources the private
`dynamic_functions/Chat/.env` before starting Python, sets database/binding paths,
and selects the `terrain-sim` remote. `OPENROUTER_API_KEY` belongs in that sourced
Chat environment, not Terrain's environment. Keep secret files owner-readable
only, and never copy their contents into logs or source control.

In another OS terminal, monitor startup:

```sh
tail -f python-server/runServer.log
```

For a new installation, provision the local launcher and private configuration
from your deployment settings first. The launcher, databases, credentials, and
installed dependencies are not supplied by this runbook or a clean Git checkout.
Do not invent new bindings for an existing world. See the host setup documentation
and [Terrain setup](README.md) for dependencies and geoid/provider configuration.

## 2. Start the dependent services

Log into your existing Atlantis account and terminal. The examples below use the
current installation's owner/remote, `Billtester/terrain-sim`. If your remote has a
different owner/name, use its actual discovered path; do not use a fixed shell ID.
Run these commands in order and wait for each to return:

```text
@/Billtester/terrain-sim/Terrain/Economy/Server/start
@/Billtester/terrain-sim/Terrain/terrain_start {"host":"127.0.0.1","port":5184}
@/Billtester/terrain-sim/ArcticSimulation/start {"host":"127.0.0.1","port":5190}
```

Verify with:

```text
@/Billtester/terrain-sim/Terrain/Economy/Server/status
@/Billtester/terrain-sim/Terrain/Server/status
@/Billtester/terrain-sim/ArcticSimulation/status
@/Billtester/terrain-sim/Terrain/Vehicles/fleet
@/Billtester/terrain-sim/Terrain/Infrastructure/fleet
```

These start calls restore existing persistent data. They do not automatically
resume paused/blocked vehicle missions. A successful fleet call checks both bank
ownership and the simulation snapshot.

## 3. Start and open the connected viewer

If the viewer is already listening on 5273, keep it running. Otherwise, from the
host repository root in another OS terminal:

```sh
cd viewer
FLASK_PROXY_TARGET=http://127.0.0.1:5184 VITE_PORT=5273 npm run dev -- --host 127.0.0.1 --strictPort
```

This installation runs the `viewer/` checkout. Use the actual viewer checkout on
another machine. Its dependencies must already be installed. `--strictPort`
prevents silently choosing another port and breaking the configured viewer URL.

In Atlantis, generate an authenticated link:

```text
@/Billtester/terrain-sim/Terrain/Viewer/map_link
```

Open the complete returned URL. It includes the fleet grant; the bare 5273 URL
does not. The URL contains an expiring credential: do not commit or publish it.
After an MCP restart, create a new link; refreshing an old grant cannot restore it.
`127.0.0.1` links work only on the machine running the viewer. For remote access,
use your configured authenticated HTTPS/tunnel deployment and its viewer URL.

## 4. Use Arnold or operate directly

Resume the saved chat through the existing Atlantis terminal:

```text
@/Billtester/terrain-sim/Chat/first_menu
```

Choose Resume existing game and the existing command-center game. Do not create
another world to recover chat. Arnold's model and persona live in
`Chat/Game/Bots/commander/`. Read [Chat setup](../Chat/README.md) for initial setup.

For direct operation, read the same dynamic-function instructions the bot uses:

```text
@/Billtester/terrain-sim/Terrain/instructions {"topic":"vehicles"}
@/Billtester/terrain-sim/Terrain/instructions {"topic":"infrastructure"}
@/Billtester/terrain-sim/Terrain/instructions {"topic":"equipment"}
@/Billtester/terrain-sim/Terrain/instructions {"topic":"placement"}
@/Billtester/terrain-sim/Terrain/instructions {"topic":"defense"}
```

Get each instance's bank UUID from fleet/placement. Then use
`Terrain/Objects/functions` with `{"asset_id":"COPY_BANK_UUID_HERE"}` to discover
currently runnable actions, parameter schemas, and bound mission/revision fields.
Use exact IDs from those responses, never display names as UUIDs.

A vehicle order is accepted before it finishes. Read `Terrain/Vehicles/mission_status`
and distinguish queued, running, awaiting_task, blocked, paused, and completed.
`auto_return: true` captures the departure position. `wait_for_task: true` requires
an explicit `complete_task` after the task is actually done. Follow the return leg
to completion. The bot does not run continuously between messages.

## Shutdown and recovery

To shut down deliberately, first inspect the fleet and pause active missions using
the current vehicle and mission UUIDs. Then, in Atlantis:

```text
@/Billtester/terrain-sim/ArcticSimulation/stop
@/Billtester/terrain-sim/Terrain/terrain_stop
@/Billtester/terrain-sim/Terrain/Economy/Server/stop
```

Only after those return, stop the MCP OS process with Ctrl-C in its terminal.
Stop the viewer with Ctrl-C in its own terminal if desired. Explicitly stopping
Terrain closes its HTTP thread before the MCP process exits.

If shutdown hangs, inspect the actual listeners before doing anything else:

```sh
lsof -nP -iTCP:8002 -iTCP:5184 -iTCP:5190 -iTCP:5192 -iTCP:5273 -sTCP:LISTEN
```

Do not launch a duplicate process onto an occupied port. Do not kill every Python
or Node process. Confirm which PID belongs to this host, try graceful termination,
and use force termination only after confirming its bank/simulation children have
stopped and identifying the stuck process. Then repeat the startup steps above.

| Symptom | Check/action |
| --- | --- |
| Fleet not connected | Issue a fresh `Viewer/map_link` and open the full URL. |
| Defense/API 503 | Check bank, simulation, and Terrain status separately; start the stopped service. |
| Viewer opens but API fails | Match `FLASK_PROXY_TARGET` to the running Terrain port. |
| Tool not found | Check remote connection, exact path, and `/search fleet`. After dynamic-function changes use `/remote refresh terrain-sim`. |
| Arnold cannot call a function | Discover it; current tool lists are subsets. Inspect `/which /Billtester/terrain-sim/Terrain/Vehicles/fly_to` for authoritative descriptions/schema. |
| Vehicle blocked | Read the mission reason and pose; resolve the cause before `mission_control` with `action: "resume"`. Do not reset or teleport it. |
| Mission paused after restart | Expected recovery behavior. Resume only the intended current mission explicitly. |

Dynamic-function refresh does not reload imported installed Python packages or a
running Node simulation child. Runtime package changes require a planned restart
and a fresh viewer grant. Configuration/docstring updates alone do not justify
shutting down the stack.

Recovery verification: all service status calls succeed, viewer and Terrain health
respond, fleet identities remain present, and no paused mission resumed without an
explicit command. The last recovery was verified on 2026-09-25 with ten fleet
vehicles; this count is evidence from that session, not a hardcoded expectation.

## Recover a blocked vehicle without editing code

1. Select the vehicle in the connected viewer and read its mission reason. For a
   blocked mission the menu offers **Retry route**. This invokes the existing
   `Terrain/Vehicles/mission_control` function with `action: "resume"`.
2. From Atlantis/MCP, read the fleet and `mission_status`, then use their current
   UUIDs (the placeholders below are not literal IDs):

   ```text
   @/Billtester/terrain-sim/Terrain/Vehicles/mission_control {"asset_id":"VEHICLE_UUID","mission_id":"CURRENT_MISSION_UUID","action":"resume"}
   @/Billtester/terrain-sim/Terrain/Vehicles/mission_status {"asset_id":"VEHICLE_UUID"}
   ```

3. For a blocked mission, resume discards the old route and recovery maneuvers,
   queues fresh planning and terrain validation, and preserves the mission UUID,
   destination and return settings. Queued is not proof of movement: observe
   running state and changing travelled distance, then completion of each leg.
4. If the reason persists, do not loop retries blindly. Missing terrain must be
   acquired, permission problems resolved, or a genuinely unreachable destination
   changed. Cancel the existing mission before issuing a replacement destination.
   Retry never teleports the vehicle or disables collision/water/slope checks.

This recovery operation does not require a service restart or a runtime code edit.
A planner defect itself still requires a software fix; retry only uses the code,
world state and terrain currently available to the running server.
