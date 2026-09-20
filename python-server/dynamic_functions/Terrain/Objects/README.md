# Selected-object functions

The viewer and MCP share `Terrain/Objects/runtime.py` for discovery and invocation.
The browser never invents available actions or moves objects locally. Every call
checks the authenticated principal's bank ownership, world and current object state.

## Viewer

1. Click a vehicle marker, its map label, its fleet row or a registered structure.
2. The **Object functions** panel fetches currently runnable functions from MCP's
   Terrain server. Select one to reveal its parameter fields.
3. For movement, click **Pick destination on map**, then click a map point or another
   object. The selected source vehicle stays selected; the click fills latitude and
   longitude. An object target supplies its location at selection time, not a pursuit
   mission. The coordinates remain editable.
4. Optionally choose **Pick return point on map**, altitude, landing and task waiting.
5. Click **Run** to submit. Picking a point does not execute anything. Escape cancels
   target picking. **Copy MCP command** copies the equivalent terminal command.
6. Observe the returned state and fleet status. Acceptance is not arrival or task
   completion. The task/return contract is documented in `../Vehicles/README.md`.

The selected object's UUID is bound automatically. Mission controls bind the current
mission UUID; component actions bind the observed revision. Stale forms are rejected,
not silently redirected to a newer mission or component revision. The server checks
runnability again on submission because state may change while a form is open.

Unattached vehicles expose inspection only. Unsupported/unregistered scene geometry
has no invented executable functions. Registered habitat components expose only
currently permitted component actions; the server still enforces interlocks.

## MCP

```text
%YOUR_USERNAME/**/Terrain/Objects/functions {"asset_id":"OBJECT_UUID"}
%YOUR_USERNAME/**/Terrain/Objects/inspect {"asset_id":"OBJECT_UUID"}
%YOUR_USERNAME/**/Terrain/Objects/call {"asset_id":"OBJECT_UUID","action_id":"ACTION_ID","parameters":{}}
```

`functions` returns function names, parameter descriptions and bound values. Supply
all required parameters and the returned bound values to `call`. The corresponding
`Terrain/Vehicles` and `Terrain/Infrastructure` functions remain usable directly.
Both paths delegate to the same simulation authority; no private viewer state machine
executes these commands.
