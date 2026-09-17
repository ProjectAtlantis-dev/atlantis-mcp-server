# Security Model

Atlantis is designed around a local MCP server that can optionally connect outward to the Atlantis cloud. The local server is trusted infrastructure; the cloud supplies authenticated user identity for remote calls.

Related docs:
- [Dynamic Functions](./README.dynamic_functions.md)
- [atlantis API](./README.atlantis_api.md)

## Network Boundary

By default, the MCP server binds only to localhost (`HOST` in `state.py`). That means remote clients cannot directly connect to the local WebSocket server. External access is mediated by the server's outbound Socket.IO connection to the Atlantis cloud.

Do not bind to `0.0.0.0` unless you have added another authentication layer. Binding publicly exposes the local MCP surface to the network.

## Cloud Identity

The server connects to the cloud with `email`, `apiKey`, service name, server UUID, and service metadata. After authentication, the cloud sends a `welcome` payload containing username records:

```json
[
  { "username": "alice", "isDefault": true },
  { "username": "bob", "isDefault": false }
]
```

These become the owners returned by `atlantis.get_owner_usernames()`, with the `isDefault` one as `atlantis.get_default_owner()`.

Cloud-forwarded tool calls are trusted to carry the authenticated `user` from the cloud. Localhost connections are treated as owner-equivalent because any local process already has access to the user's machine.

## Access Control

Internal management tools are owner-only:

- `_function*`
- `_server*`
- `_admin*`

The server authorizes these by checking whether the caller is localhost or whether the cloud-authenticated username is in `atlantis.get_owner_usernames()`.

Dynamic functions are hidden unless decorated. The main visibility model is:

- `@public` / `@index`: callable by anyone.
- `@protected`: callable only if the custom protection function allows it.
- `@visible` and other non-public decorators: owner-only.
- No visibility decorator: not remotely callable.

## Function Source Exposure

`_function_get` returns the entire file containing a function, not only that function body. If a file contains imports, helpers, constants, comments, or multiple functions, all of that content can be returned to an authorized caller.

Do not hardcode secrets in dynamic function files. Read them from the environment, and keep sensitive implementation details separate from public or shared functions.

## Trust Assumptions

This model assumes:

- The host machine and local processes are trusted.
- The cloud correctly authenticates users and does not spoof the forwarded `user`.
- The local server remains bound to localhost in normal use.
- Secrets live in environment variables or local ignored files, not dynamic function source.

It does not protect against:

- Malicious local processes.
- A compromised host machine.
- A compromised or malicious cloud server.
- Secret leakage from hardcoded values in files exposed by `_function_get`.
