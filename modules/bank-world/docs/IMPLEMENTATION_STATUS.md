# Implementation status and next gates

## Agreed workstreams

| Workstream | Development location | Current state | Next acceptance gate |
| --- | --- | --- | --- |
| Partial lot sale | `packages/bank` | Implemented and tested atomically for 12.5t → 5t + 7.5t | Port to PostgreSQL transaction/row locks and exact numeric adapter |
| Gatho + Packcat | `packages/realtime`, `packages/protocol`, `spikes/gatho` | Transport boundary, sequencing, replay, UUID codec, and spike criteria exist | Pin exact revisions; prove auth, reconnect, backpressure, slow clients, restart recovery |
| Redis | `packages/realtime`, `packages/persistence`, `docs/REDIS.md` | ioredis pinned; test-safe Pub/Sub, Streams replay, presence, leases, and connection factory implemented | Exercise against a dedicated local Redis and then connect the Gatho Redis driver |
| First bank UUID boat in WebGPU | `worktrees/atlantis-terrain-game/webserver/game` | Applied on local branch `greenland-game-webgpu`; build, 62 terrain tests, and Vite→game proxy read model pass | Visually verify the starter boat in the rendered scene |
| Rapier local physics | `packages/physics` and terrain integration bundle | WGS84 local frame and injected server/client adapters exist | Pin Rapier; test player/boat/VTOL bodies and local terrain colliders |
| BVH static collision | terrain integration bundle `collision/` | Per-tile BVH lifecycle module staged | Wire tile add/remove lifecycle; benchmark rocks, placement, LOS, and raycasts |
| Instanced animated actors | terrain integration bundle `actors/` | Three upgraded to 0.185.1; disabled dynamic-import feature module staged | Install package and run a rendered crowd/animation stress test |
| GPUcat-inspired swarms | terrain integration bundle `drones/` | Three-WebGPU/TSL pass design recorded; GPUcat is not installed | Build CPU authority first, then benchmark TSL clear/bin/simulate/cull at 256+ drones |

## Vertical slice already executable

- Atlantis `x_user` identity resolves to a separate game-bank account.
- New players receive one deterministic bank-registered starter boat UUID.
- Nuuk, Sisimiut, and Ilulissat and explicit bidirectional sea routes are seeded.
- Cargo accepts bank-verified resource-lot UUIDs and enforces tonnage capacity.
- Voyages are server-timed, idempotent, fuel-checked, interpolated in WGS84, and dock only at arrival.
- The bank holds depth-12 parcel titles, balances, listings, quotes, custody, service credentials, production claims, transformations, and provenance.
- Full and partial warehouse settlements are bank-atomic.
- Named bank, game, and warehouse dynamic functions are staged as thin adapters.

## Required before any production-like environment

1. Implement PostgreSQL repositories for bank and world instead of changing the tested domain behavior.
2. Run migrations only against explicitly named local databases such as `greenland_bank_test` and `greenland_world_test`.
3. Add row-level locking, exact `numeric` quantities, balanced deferred ledger constraints, and integration tests.
4. Finish trusted caller-context propagation in Atlantis MCP and verify that no public tool accepts a selected player/account ID.
5. Add a restricted same-origin terrain read-model gateway; never expose authority/service tokens to WebGPU.
6. Add outbox/inbox reconciliation between simulation and bank for market, fuel, production, and damage-related economic events.
7. Add rate limits, audit correlation IDs, metrics, backups, and recovery exercises.

## Not yet implemented

- Buying market stock and selling cargo as a complete game command.
- Dock/berth allocation and local maneuver sessions.
- Fuel purchase settlement, repairs, damage, weather, missions, loans, and insurance.
- Automated production, warehouse billing, routes, projects, weapons, and warfare.
- A running Redis deployment and completed Gatho Redis-driver integration.
- Browser-level visual verification against a simultaneously running game API.
