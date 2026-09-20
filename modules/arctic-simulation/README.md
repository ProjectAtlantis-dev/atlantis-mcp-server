# Arctic simulation module — staging

Independent extraction of the tested simulation port, not a production world server.
The original integration checkout is preserved. This repository has no remote and
has not been deployed to the running stack. It is now installed in the separate
staging host; see `../atlantis-modularization/INTEGRATION.md` for current results.

## Host installation contract

Install this Python package into the **target MCP host's environment** using
`python -m pip install -e /absolute/path/to/atlantis-simulation-module`.
Node >=22.5 is also required. Symlink the repository's
`dynamic_functions/ArcticSimulation` folder into the host's `dynamic_functions/`.
Do not replace that entire directory or overwrite an existing application.
The upstream loader follows application symlinks; no loader patch is required.
The wheel packages the runtime; the MCP application folder is installed separately.

Set `ATLANTIS_SIM_DB_PATH` explicitly to a dedicated scenario database outside
source control. Never point it at the legacy bank/world databases. Choose a free
loopback port when calling `ArcticSimulation.start`; startup is explicit, not an
import side effect. Do not expose the child service directly to the network.

## Authority and boundaries

- One supervised Node child owns the fixed-step clock and scenario persistence.
  Dynamic functions issue commands; they do not advance time. No second callback
  is registered against core `game_tick.py` for these same scenario entities.
- This is **scenario state**, not canonical bank ownership or the shared world.
  Scenario IDs and the default game name are not trusted multiplayer identities.
- The infrastructure adapter exposes individual component operations with state
  preconditions. Animation/visual state does not establish engineering safety.
- Reset affects a scenario only. Automatic terrain-catalog synchronization and
  `sync_asset_vehicles` were deliberately excluded: they depended on a patched
  old Terrain API and cannot claim canonical UUID authority. The complete old
  implementation remains in the integration checkout and preservation archive.
- A packaged viewer HTTP adapter and explicit caller/game identity bindings are
  tested in staging. Bank/world UUID continuity is verified, but live catalog and
  scenario IDs are not yet reconciled with that bank. Production lifecycle and
  migration remain gated. Do not activate two writers for the same entities.

## Verification

`node --test python/atlantis_simulation/runtime/test/*.test.mjs`

`PYTHONPATH=python python3 -m unittest discover -s tests -v`

Tests use temporary databases and free ports, never the running stack or Redis.
