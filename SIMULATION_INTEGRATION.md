# Greenland simulation and dynamic functions integration

Paired viewer branch: `artic-simulation` in `ProjectAtlantis-dev/atlantis-3d-viewer`.

## Repository responsibilities

- `modules/arctic-simulation`: authoritative simulation, ground routing, VTOL controllers, mission and return state, persistence, habitat/defense components, and the authenticated viewer bridge.
- `modules/bank-world` and `modules/host-adapters`: bank ownership/UUIDs, economy/world services, and host identity adapters.
- `python-server/dynamic_functions/Terrain/Vehicles`: coordinate-driven movement, mission status/control, return destinations, and owner fleet queries.
- `Terrain/Objects`: shared runnable-action discovery/invocation for MCP and viewer, parameter schemas, and current-state/ownership checks.
- `Terrain/Viewer`: scoped links and session renewal. `Terrain/Demo`, `Economy`, and `Infrastructure` expose the related application workflows.
- Terrain catalog, composition, and HTTP changes connect existing terrain data to these services and normalize legacy elevation datum metadata.

The viewer's rendering, models, menus and animations belong in the paired viewer repository. The local `viewer/` checkout is ignored here. Runtime databases, tokens, caches, virtual environments and local vehicle ownership records are not source artifacts.

## Runtime and instructions

Read the module READMEs for service setup and `python-server/dynamic_functions/Terrain/Vehicles/README.md` for parameters, states, Lobster commands, and limits. Configure your own terrain databases, catalog bindings and owner assets; none are preassigned by this branch. Install the local modules into the host environment; editing packaged controller source requires reinstall/reload. Python dynamic functions hotload through the host.

Ground route search uses verified terrain, road centerlines and padded building bounds, with a 256-metre local window. It supports backing out and waypoint-based progress. It is not a complete settlement-wide road graph. Supported server vehicle controllers are Patria AMV and Black Hornet/Osprey VTOL; fixed-wing RQ-180 and boat controllers remain unfinished.

All 39 simulation runtime tests passed, including the two localhost tests rerun outside the sandbox, with live AMV mission and browser verification. The paired viewer has a separately documented full-suite building-model test failure.
