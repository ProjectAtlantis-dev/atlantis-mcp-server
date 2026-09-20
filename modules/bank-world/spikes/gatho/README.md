# Gatho + Packcat evaluation spike

Do not couple game/domain code directly to Gatho. Implement it behind `RealtimeTransport` after a focused spike.

Evaluate the exact reviewed revisions first:

- Gatho: pin reviewed commit `775b8e38b4e64c15f7a46072e255cce66cd19c7d` (`0.0.0-alpha-3`).
- Packcat: use typed binary command/snapshot schemas and encode UUIDs as 16 bytes.

The spike passes only if it demonstrates authenticated room admission, reconnect replay, bounded reliable buffering, backpressure, a slow-client policy, process restart recovery, and observability. Use port/local-action rooms such as `port:NUUK` and `combat:<session UUID>`; persistent ownership, money, cargo provenance, and global voyages remain in the bank/world databases.

Until this passes, the in-memory adapter is for tests only. No alpha networking dependency is installed in the authoritative packages.
