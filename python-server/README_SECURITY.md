# Security Model

This document describes the security architecture of the Atlantis MCP Server.

> **📖 Related Documentation**:
> - Function-level access control details: [README.dynamic_functions_technical.md](./README.dynamic_functions_technical.md#security-model)
> - Decorator usage (@visible, @public, @protected): [README.dynamic_functions_technical.md](./README.dynamic_functions_technical.md#visible-vs-public-vs-protected)

## Network Architecture

### Localhost-Only Binding

The MCP server listens **only on 127.0.0.1** (localhost) by default:

```python
# state.py:71
HOST = "127.0.0.1"  # Listen on localhost only for security
```

**Why this matters**: Remote clients cannot directly connect to the WebSocket server. Only local processes on the same machine can establish connections.

### Cloud Connection Model

External access is provided through an **outbound connection** to the Atlantis cloud server:

1. The local server initiates a Socket.IO connection to `ws.projectatlantis.ai`
2. The cloud server authenticates users and forwards MCP requests
3. The `user` parameter in requests comes from the cloud server's authentication, not from untrusted clients
4. Owner identity is set by the cloud server via the `welcome` event (server.py:3522)

**Security benefit**: The attack surface is minimized. There are no inbound connections from untrusted sources.

## Access Control

### Internal Functions (_function*, _server*, _admin*)

Functions with these prefixes are **owner-only** (server.py:1797-1812):

```python
if (actual_function_name.startswith('_function') or
    actual_function_name.startswith('_server') or
    actual_function_name.startswith('_admin')):

    caller = user or client_id or "unknown"
    owner = atlantis.get_owner()

    # Treat localhost websocket connections as the owner
    if caller.startswith("ws_127.0.0.1_") and owner:
        caller = owner

    if owner and caller != owner:
        raise ValueError(f"Access denied: Internal functions can only be accessed by owner")
```

**Protected functions include**:
- `_function_set`: Upload/modify function code
- `_function_get`: Retrieve function source code
- `_function_remove`: Delete functions
- `_server_*`: Server management operations
- `_admin_*`: Administrative operations

### Function Code Retrieval Scope

**IMPORTANT**: `_function_get` returns the **entire file** containing the requested function, not just the function itself (DynamicFunctionManager.py:374-375).

**Security implications**:
- If a file contains multiple functions, all of them are returned
- All imports, helper code, comments, and constants in the file are exposed
- **Sensitive data** (API keys, credentials, secrets) should NEVER be hardcoded in function files
- Use environment variables for sensitive configuration (see `set_OPENROUTER` example)

**Example**:
```python
# BAD - Don't do this
API_KEY = "sk-secret-key-12345"  # This will be exposed via _function_get!

@public
async def my_function():
    # ...
```

```python
# GOOD - Use environment variables
import os
API_KEY = os.getenv("API_KEY")  # Safe - only the reference is exposed

@public
async def my_function():
    # ...
```

### Decorator-Based Access Control

User-defined functions must have visibility decorators (server.py:1814-1856):

- **@public / @index**: Anyone can call (including anonymous users)
- **@protected**: Access validated by custom protection function
- **Other decorators** (see `DynamicFunctionManager.py`): Owner-only access

Functions without visibility decorators **cannot be called remotely**.

### Localhost Privilege Escalation

Connections from `127.0.0.1` are treated as the owner (server.py:1805-1806, 1847-1848):

```python
if caller.startswith("ws_127.0.0.1_") and owner:
    caller = owner
```

**Rationale**: Since only the localhost binding is exposed, any process that can connect is already running on the owner's machine with local access. This allows local tools (Claude Code, scripts, etc.) to function as the owner without requiring separate authentication.

## Authentication Flow

1. **Cloud Authentication**: User authenticates with cloud server using email + API key
2. **Owner Assignment**: Cloud server sends `welcome` event with username → sets `atlantis._owner`
3. **Request Context**: Cloud server includes `user` field in forwarded tool calls
4. **Access Check**: Server validates `user` matches `owner` for protected operations

## Security Assumptions

This security model assumes:

1. ✅ The host machine is trusted (localhost binding)
2. ✅ The cloud server properly authenticates users
3. ✅ The cloud server doesn't spoof the `user` parameter
4. ✅ Local processes are authorized to act as owner

## Threat Model

### Protected Against

- ❌ Direct remote connections to MCP server (localhost binding)
- ❌ Unauthorized modification of functions (owner checks)
- ❌ Remote execution of internal tools (prefix-based protection)
- ❌ Execution of functions without visibility decorators

### Not Protected Against

- ⚠️ Malicious local processes (they have full access via localhost)
- ⚠️ Compromised cloud server (trust relationship required)
- ⚠️ Host machine compromise (local attacker = owner privileges)
- ⚠️ Exposure of hardcoded secrets via `_function_get` (use env vars instead)

## Best Practices

1. **Never hardcode secrets** in function files - use environment variables
2. **Use `set_OPENROUTER`-style files** (gitignored) for local configuration
3. **Keep sensitive logic separate** - don't mix public and sensitive code in the same file
4. **Review file contents** before sharing function names with untrusted parties
5. **Bind to 127.0.0.1 only** in production environments

## Configuration

To change the binding address (⚠️ **not recommended for production**):

```bash
python server.py --host 0.0.0.0  # Exposes to network - USE WITH CAUTION
```

**Warning**: Binding to `0.0.0.0` or a public IP exposes the server to remote connections and bypasses the security model. Only do this in controlled environments with additional authentication layers.
