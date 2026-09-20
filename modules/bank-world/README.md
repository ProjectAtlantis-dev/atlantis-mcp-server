# Greenland game server

This is the test-only authoritative backend staging area for the Greenland logistics game. It was created inside `grid_visualizer` because that is the writable workspace; it can become a sibling repository later without changing its internal layout.

Nothing here connects to a production database. The runnable adapter uses local SQLite files whose names must end in `.test.sqlite`, refuses `NODE_ENV=production`, and rejects `DATABASE_URL`, `BANK_DATABASE_URL`, and `WORLD_DATABASE_URL`.

## Runtime ownership

```text
Atlantis login / trusted x_user.id
                  |
      named atlantis-mcp-server instances
       game          bank          warehouses
         |             |                |
         +-------------+----------------+
                       |
       Greenland authoritative backend
       world simulation <-> central bank
              |              |
       realtime rooms    ledger/UUID/land
              \              /
               WebGPU read models
                       |
         atlantis-terrain-server main
```

The bank owns money, ownership, custody, parcel titles, UUID provenance, and settlement. The world owns ports, markets, routes, cargo assignment, fuel, vehicle operation, and voyage timing. The terrain client renders read models and sends authenticated commands; it is never an economic or position authority.

## Layout

```text
apps/api                         local/test HTTP process
packages/bank                    bank service and routes
packages/world                   authoritative world simulation
packages/domain                  authority and UUID policy
packages/protocol                versioned realtime envelopes + 16-byte UUID codec
packages/realtime                transport, Redis streams/fanout/presence/lease adapters
packages/physics                 local geodetic frame + injected Rapier session
packages/persistence             production/test database safety gates
integrations/atlantis-mcp        thin dynamic-function adapters
migrations/bank                  future PostgreSQL bank schema
migrations/world                 future PostgreSQL world schema
spikes/gatho                     Gatho/Packcat acceptance criteria
test                             bank, world, identity, protocol, physics, and safety tests
```

## Local verification

From this directory:

```sh
npm install
npm test
```

To run the API locally after choosing two test-only secrets:

```sh
BANK_AUTHORITY_TOKEN=local-bank-test-secret \
GAME_SERVER_AUTHORITY_TOKEN=local-game-test-secret \
npm start
```

It binds to `127.0.0.1:3010` and creates `.local/greenland_bank.test.sqlite` plus `.local/greenland_world.test.sqlite`. Without the secrets, status reads remain available while mutations are disabled.

## Resource UUID rule

One unique boat, machine, structure, equipment item, warehouse, or land title has one UUID. Fungible goods use one UUID per conserved lot, not per kilogram or individual fish.

A partial sale is atomic. Selling 5 tonnes from a 12.5-tonne fish lot consumes the parent UUID and creates:

- a bank-owned lineage edge to a new 5-tonne buyer lot UUID;
- a bank-owned lineage edge to a new 7.5-tonne seller remainder UUID;
- balanced buyer/seller/warehouse ledger entries;
- one settled quote and idempotency record.

If any validation or ledger operation fails, none of those changes persist.

## Current boundary

The SQLite services are a verified vertical-slice implementation, not the final scale architecture. The Redis bridge is implemented for ephemeral realtime coordination but is never bank authority. Gatho/Packcat, Rapier, BVH, animated instancing, and GPU compute remain adapters or compatibility spikes until their pinned versions pass the gates in `docs/IMPLEMENTATION_STATUS.md`.
