# Redis role and boundaries

Redis is part of the target runtime, but it is not a second bank or world database.

Use Redis for:

- Gatho room drivers and cross-process room coordination;
- realtime Pub/Sub fanout between game-server instances;
- bounded Redis Streams replay for reconnecting clients;
- short-lived player presence and session routing;
- expiring leases so one process owns a local simulation room/tick;
- cached world/read-model data that can always be rebuilt;
- rate-limit counters and short-lived command deduplication.

Do not store the authoritative version of these only in Redis:

- game-credit balances or ledger entries;
- bank UUID assets, owners, custody, or provenance;
- land titles;
- settled warehouse transactions;
- durable voyages, production claims, projects, or market settlement results.

Those remain in PostgreSQL, with SQLite used only by the current isolated prototype tests.

## Test safety

The Redis test configuration defaults to `redis://127.0.0.1:6379/15` and requires the namespace prefix `greenland:test:`. Remote Redis is rejected unless explicitly opted in. No Redis connection is made merely by importing the packages.

`RedisRealtimeBridge` accepts dedicated publisher/subscriber clients, publishes live room messages, keeps a bounded Redis Stream for replay, writes expiring presence, and provides compare-and-delete leases. Gatho should consume the same Redis deployment behind its own driver once its spike passes.

For deployment, use a dedicated Redis database/cluster and credentials per environment. Redis loss must degrade realtime presence/replay, not corrupt bank or persistent world truth.
