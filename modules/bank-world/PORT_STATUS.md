# Preserved candidate — installed in isolated staging, not deployed live

Source copied from `grid_visualizer/greenland-game-server` on 2026-09-17.
Original files remain in place. Dependencies, secrets and runtime databases were
not copied into this repository; consistent database snapshots are in the
separate modularization preservation archive.

Do not activate against production accounts or the existing Redis listener.
The MCP apps now use the shared explicit identity binding adapter and are linked
alongside Terrain and Chat in the staging host. Actual UUID continuity/restart and
dedicated Redis world-relay integration tests pass; see INTEGRATION.md below.
This code includes bank/world UUID, accounting and Redis bridge work worth
preserving, but needs integration and persistence auditing before deployment.

See `../atlantis-modularization/INTEGRATION.md` for ownership contracts, baseline,
verified work and remaining blockers. In particular, never silently enable SID
identity fallback, point tests at Postiz Redis, or let both the scenario runtime
and world service write the same entity's position.
