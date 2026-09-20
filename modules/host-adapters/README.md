# Atlantis host adapters

Shared contracts used by independent simulation and bank/world applications.
No services start on import. The bank does not depend on the simulation runtime.

`ATLANTIS_IDENTITY_BINDINGS` names an operator-owned JSON file outside source
control. Format:

```json
{"version": 1, "bindings": []}
```

Each approved entry contains `caller` (host-authenticated caller SID),
`user_game_id` (positive host game integer), `external_user_id` (`x_user:` plus
the verified numeric account ID), `scenario` (letters/numbers/underscore/hyphen),
and `permissions` (an explicit subset of `simulation`, `bank`, `world`).

An empty policy denies all calls that need identity. Do not invent a numeric
user ID or derive it from an email/SID. Provision only from verified account
records. This operator binding is not a claim that the current cloud protocol
provides numeric identity; it is an explicit trust adapter until it does.
Do not accept a policy path, account ID or permission list from tool arguments.

Policy is rechecked on each scoped command and viewer request. Duplicate or
malformed bindings fail closed. The host's existing localhost-owner/cloud trust
boundary still applies; this does not authenticate a malicious local process.
