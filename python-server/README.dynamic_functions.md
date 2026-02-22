# Dynamic Functions Documentation

**Quick Links:**
- [User Guide](#user-guide) - How to create and use dynamic functions
- [Technical Architecture](#technical-architecture) - Internal implementation for developers

---

# User Guide

## Overview

Create Python functions that become MCP tools automatically. Put `.py` files in `dynamic_functions/` directory.

**Features:**
- Multiple functions per file supported
- Auto-discovery and registration
- Live reloading on file changes
- Each function becomes its own MCP tool

## File Structure

```
dynamic_functions/
├── chat.py              # Single function (or kitty.py)
├── math_operations.py   # Multiple functions
├── user_management.py   # Related functions grouped
└── OLD/                 # Automatic backups
```

Organize functions however makes sense - one per file or group related functions together.

## Basic Example

```python
import atlantis

@visible
async def add(x: float, y: float):
    """Add two numbers. Use for basic addition operations."""
    result = x + y
    await atlantis.client_log(f"{x} + {y} = {result}")
    return result

# No decorator = hidden by default
async def helper():
    """Helper function - not exposed as tool."""
    return "internal use only"
```

## Requirements

1. Import `atlantis` module
2. Use `async def` (recommended)
3. Add type hints for parameters
4. **CRITICAL**: Docstring becomes AI tool description
5. Use appropriate decorators

## Docstring Guidelines

Write for AI consumption. Be explicit about purpose and when to use.

**Good:**
```python
"""Calculate distance between coordinates using Haversine formula. Use for measuring distances between lat/lng points."""
```

**Bad:**
```python
"""This function does math."""  # Too vague
```

## Decorators

### Visibility (Required)

Functions are **hidden by default** - you must use a visibility decorator to expose them as MCP tools.

- **`@visible`** - Make function visible in tools list (owner-only access)
- **`@public`** - Make function publicly accessible to all users (no authorization)
- **`@protected("func_name")`** - Make function visible with custom authorization
- **No decorator** - Function is hidden by default, not exposed as tool

### Optional Metadata
- **`@copy`** - Allow non-owners to view function source via `_function_get` (based on visibility rules)
- **`@chat`** - Chat functions that get transcript/tools and call LLM
- **`@location(name="location_name")`** - Associate with location
- **`@shared`** - Persist across reloads

### Deprecated
- **`@hidden`** - Obsolete (functions are hidden by default without @visible)

**Combine decorators:**
```python
@location(name="office")
@visible
async def calculate(x: float, y: float):
    """Calculate with app and location context."""
    return x + y
```

## Decorator Behavior

### @visible vs @public vs @protected

Understanding the difference between these decorators is important for access control:

**`@visible`** - Owner-only access:
```python
@visible
async def admin_command(action: str):
    """Execute admin action. Only accessible by function owner."""
    return f"Executing {action}"
```
- Function appears in tools list
- Only the **owner** can call this function
- Use for admin tools, private operations, owner-specific features

**`@public`** - Multi-user access:
```python
@public
async def public_service(query: str):
    """Public API service. Accessible to all users."""
    return f"Result for {query}"
```
- Function appears in tools list (implies `@visible`)
- **Anyone** can call this function (handled in cloud infrastructure)
- Use for shared tools, public APIs, multi-user features
- No need to combine with `@visible` - `@public` includes visibility

**`@protected(name)`** - Group-based access control:
```python
@protected("demo_group")
async def special_function(data: str):
    """Protected function with custom authorization."""
    return f"Processing {data}"

@visible
async def demo_group(user: str):
    """Protection function that authorizes users for demo_group."""
    allowed_users = ["alice", "bob", "charlie"]
    return user in allowed_users
```
- Function appears in tools list (visible to everyone)
- When called, the **protection function** (named by the decorator parameter) is invoked first
- Protection function name must be a valid Python identifier (e.g., `demo_group`, not `demo group`)
- Protection function receives the `user` parameter and returns `True` (allow) or `False` (deny)
- If allowed, the protected function executes; otherwise, raises `PermissionError`
- Use for custom authorization: groups, roles, permissions, API keys, database checks, etc.
- Protection functions must be top-level (not in any app) and decorated with `@visible`

**Access Control Summary:**
- No decorator → Hidden, not callable
- `@visible` → Visible, owner-only
- `@protected("func_name")` → Visible to all, custom authorization via protection function
- `@public` → Visible, accessible to all users (no authorization)

### @copy - Share Your Source Code

The `@copy` decorator allows non-owners to view a function's source code via `_function_get` based on the function's visibility rules.

**`@copy`** - Source code sharing:
```python
@copy
@public
async def open_source_algorithm(data: list):
    """Public algorithm - anyone can view and copy the source code."""
    return sorted(data, reverse=True)

@copy
@protected("premium_users")
async def premium_algorithm(data: list):
    """Premium algorithm - only authorized users can view source."""
    return [x * 2 for x in data]

@copy
@visible
async def private_algorithm(data: list):
    """Private algorithm - only owner can view source (same as without @copy)."""
    return data[::-1]
```

**How `@copy` works:**
- By default, `_function_get` (source code retrieval) is **owner-only** for all functions
- Adding `@copy` allows non-owners to retrieve source code based on visibility:
  - `@copy + @public` → **Anyone** can view source code
  - `@copy + @protected("func")` → **Custom authorization** via protection function
  - `@copy + @visible` → **Owner-only** (same as without `@copy`)
- Without `@copy`, only the owner can ever use `_function_get` on that function

**Use cases:**
- 🌐 **Open source functions** - Share your code publicly
- 📚 **Educational functions** - Let students view example implementations
- 💎 **Premium content** - Grant source access to paying users via `@protected`
- 🔒 **Keep private** - Omit `@copy` to keep source code owner-only

**Security notes:**
- ⚠️ **IMPORTANT:** `_function_get` returns the **entire file** containing the function, not just the function itself
- **Best practice:** Put `@copy` functions in their own dedicated files to avoid exposing other code
- The `@copy` decorator **only affects `_function_get`** (source code viewing). It does not change who can *call* the function - that's controlled by `@visible`/`@public`/`@protected` as usual

**File organization example:**
```
dynamic_functions/
├── my_private_logic.py       # No @copy - contains secrets, private helpers
├── my_public_algorithm.py    # Has @copy - isolated, safe to share
└── my_app/
    ├── internal.py            # No @copy - business logic
    └── examples.py            # Has @copy - educational examples only
```

> **📖 Security Note**: For comprehensive security information including network architecture,
> secrets management, and best practices, see [README_SECURITY.md](./README_SECURITY.md).

## Atlantis Module

The `atlantis` module is automatically injected into every dynamic function's execution context. It provides the bridge between your Python code and the MCP client/cloud infrastructure.

**What it does:**
- **Client Communication**: Send messages, images, video, HTML, markdown, and structured data back to the requesting client
- **Streaming**: Stream responses chunk-by-chunk for real-time output (useful for LLM responses)
- **Awaitable Commands**: Send commands to clients and wait for responses (e.g., get user input, fetch transcript)
- **Context Access**: Get info about who's calling, which request this is, who owns the remote
- **Shared State**: Persist objects (like DB connections) across function reloads

**See `atlantis.py` for the full API** - the docstrings there are authoritative. Key functions include `client_log()`, `client_command()`, `stream_start/stream/stream_end()`, and various `get_*()` context accessors.


## Shared Container

Use `atlantis.shared` for persistent memory objects (connections, not data).

```python
# Initialize database connection once
if not atlantis.shared.get("db"):
    atlantis.shared.set("db", sqlite3.connect("app.db"))

db = atlantis.shared.get("db")
```

**Store:** DB connections, API clients, caches
**Don't store:** User data, application data (use databases)

**Methods:** `shared.get(key)`, `shared.set(key, value)`, `shared.remove(key)`, `shared.keys()`

## Examples

### Multiple Functions Per File
You can put many functions in one file - each becomes its own MCP tool:

```python
# File: user_management.py
import atlantis

@visible
async def create_user(username: str, email: str):
    """Create user account. Use for user registration."""
    return {"user_id": 123, "username": username}

@visible
async def get_user(username: str):
    """Get user by username. Use to retrieve user details."""
    return {"username": username, "email": "user@example.com"}

@visible
async def delete_user(username: str):
    """Delete user account. Use to remove users."""
    return {"success": True}

# No decorator = hidden by default
def _validate_email(email: str):
    """Helper function - not exposed as MCP tool."""
    return "@" in email
```

**Result:** 3 separate MCP tools (`create_user`, `get_user`, `delete_user`) + 1 hidden helper
**Benefits:** Group related functions, share helpers, common imports

### Streaming
```python
@visible
async def stream_data():
    """Stream data to client."""
    stream_id = await atlantis.stream_start("data", "stream_data")
    await atlantis.stream("chunk 1", stream_id)
    await atlantis.stream_end(stream_id)
```

### Client Commands
```python
@visible
async def get_input():
    """Get input from client."""
    name = await atlantis.client_command("\\input", {"prompt": "Name?"})
    return f"Hello {name}"
```

### Helper Functions
Functions without `@visible` are hidden by default - perfect for internal helpers:

```python
@visible
async def process_data(data: str):
    """Process and validate data."""
    # Use internal helper functions
    if not _validate_data(data):
        return "Invalid data"

    cleaned = _clean_data(data)
    return f"Processed: {cleaned}"

# No decorator = hidden by default, not exposed as MCP tool
def _validate_data(data: str):
    """Internal helper - validates data format."""
    return len(data) > 0 and data.strip() != ""

# No decorator = hidden by default, not exposed as MCP tool
def _clean_data(data: str):
    """Internal helper - cleans and formats data."""
    return data.strip().lower()
```

**Patterns:**
- Keep helper/utility functions without decorators (hidden by default)
- Only add `@visible` to functions that should be MCP tools
- Internal functions can still be called by visible functions

### Chat Function
```python
@chat
@visible
async def chat():
    """Chat function that processes conversation and calls LLM."""
    # Get conversation history
    transcript = await atlantis.client_command("\\transcript get")

    # Get available tools
    tools = await atlantis.client_command("\\transcript tools")

    # Call your LLM with transcript and tools
    response = await call_llm(transcript, tools)

    # Stream response back
    stream_id = await atlantis.stream_start("chat", "ai_assistant")
    await atlantis.stream(response, stream_id)
    await atlantis.stream_end(stream_id)
```

## Type Hints

Type hints generate JSON schemas automatically:
```python
def func(text: str, items: List[str], optional: Optional[int] = None):
    """Function with type hints."""
    pass
```

## Best Practices

- Use `async def` for functions
- Add type hints for parameters
- Write clear docstrings for AI
- Use `atlantis.shared` for connections only
- Group related functions in same file
- Use descriptive names

## Troubleshooting

**Function not showing:** Check syntax, decorators, file location
**Execution errors:** Check `.log` files in `dynamic_functions/`
**Context issues:** Use `await` with atlantis methods

Functions automatically become MCP tools when saved to `dynamic_functions/`.

---

# Technical Architecture

This section explains the internal architecture of the dynamic functions system for developers working on the server codebase.

## Architecture Overview

The dynamic functions system has **one source of truth**: the **file mapping**. This mapping controls both what functions can be called and what functions appear in the tools list.

```
┌─────────────────────────────────────────────────────────────┐
│                    File System (*.py files)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         _build_function_file_mapping()                       │
│         (DynamicFunctionManager.py:683)                      │
│                                                              │
│  • Scans all .py files recursively                          │
│  • AST parses each file                                     │
│  • Extracts function metadata                               │
│  • EXCLUDES @hidden functions (line 736)                    │
│  • Builds _function_file_mapping dicts                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              File Mapping (SINGLE SOURCE OF TRUTH)           │
│                                                              │
│  _function_file_mapping:         {func_name: file_path}     │
│  _function_file_mapping_by_app:  {app: {func: file_path}}   │
│  _skipped_hidden_functions:      [{name, app, file}, ...]   │
└─────────────┬─────────────────────────────┬─────────────────┘
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│   _get_tools_list()     │   │   function_call()           │
│   (server.py:621)       │   │   (DynamicFunctionMgr:239)  │
│                         │   │                             │
│  • Uses file mapping    │   │  • Looks up in file mapping │
│  • Creates Tool objects │   │  • If not found → 404       │
│  • Redundant @hidden    │   │  • If found → load & exec   │
│    check (line 564)     │   │                             │
└─────────────────────────┘   └─────────────────────────────┘
```

## Connection Types & Request Routing

The server supports **two distinct connection types** with different entry points but shared core logic:

### 1. Local WebSocket Connection 🏠

**Used by:** `npx atlantis-mcp --port 8000`, Claude Desktop, or any MCP client connecting locally

**Endpoint:** `ws://localhost:PORT/mcp` (defined in server.py:4066)

**Entry Point:** `handle_websocket()` (server.py:3737)

**Architecture:** Acts as a **routing layer** that forwards requests to cloud connections

**Capabilities:**
- ✅ Exposes pseudo tools (defined dynamically by cloud via `welcome` event)
- 🔄 Actual work is routed to cloud connections (not executed locally)
- ✅ Standard MCP JSON-RPC protocol over WebSocket

**Client Registration:**
```python
client_id = f"ws_{websocket.client.host}_{id(websocket)}"
client_connections[client_id] = {"type": "websocket", "connection": websocket}
```

### 2. Cloud Socket.IO Connection ☁️

**Used by:** Cloud-based clients connecting via Socket.IO

**Transport:** Socket.IO with custom namespace

**Entry Point:** `@self.sio.event` handler for `service_message` (server.py:3502)

**Capabilities:**
- ✅ Full access to all dynamic functions
- ✅ Can execute actual Python functions from `dynamic_functions/`
- ✅ JSON-RPC over Socket.IO events
- ✅ Supports awaitable commands with correlation IDs

**Request Flow:**
```
Socket.IO(service_message) → service_message handler (3502)
  → _process_mcp_request() (3558)
  → [method routing]
  → tools/list:  get_filtered_tools_for_response() (2991)
  → tools/call:  _handle_tools_call(for_cloud=True) (2687)
                   → SKIPS pseudo tool intercepts
                   → _execute_tool() (2769)
                   → function_manager.function_call()
                   → Actual dynamic function execution
```

**Client Registration:**
```python
client_id = f"cloud_{self._creation_time}_{id(self)}"
client_connections[client_id] = {"type": "cloud", "connection": self}
```

### Pseudo Tools (Cloud Welcome Event)

Local WebSocket clients see **pseudo tools** that act as a routing layer to cloud execution. These tool definitions are sent dynamically by the cloud as part of the `welcome` event when the server connects:

```typescript
// Welcome message payload from cloud
interface WelcomeMessage {
  usernames: string[];
  genericRequestId: string;   // required - fatal error if missing
  pseudoTools: PseudoTool[];  // tool definitions for local clients
}
```

The pseudo tools are stored on the server and returned to local MCP clients via `get_pseudo_tools_for_response()`. Cloud clients execute dynamic functions directly and do not use pseudo tools.

### Comparison Table

| Aspect | Local (WebSocket) | Cloud (Socket.IO) |
|--------|-------------------|-------------------|
| **Entry Point** | `handle_websocket()` | `service_message()` |
| **Protocol** | MCP JSON-RPC over WebSocket | JSON-RPC over Socket.IO events |
| **Tools Exposed** | Pseudo tools (from cloud welcome) | All dynamic functions |
| **Dynamic Functions** | Routed to cloud via pseudo tools | Executed directly via `_execute_tool()` |
| **Use Case** | Local MCP clients (routing layer) | Cloud execution backend |

## Key Components

- **`DynamicFunctionManager`** (`DynamicFunctionManager.py`) - Manages function lifecycle: file scanning, mapping, loading, and execution. The file mapping is the single source of truth for what functions exist and can be called.
- **`DynamicAdditionServer`** (`server.py`) - MCP protocol handler. Manages tool lists, caching, and request routing between local and cloud connections.

See code comments in each file for implementation details.

## Security Model

> **📖 See Also**: [README_SECURITY.md](./README_SECURITY.md) for network security, authentication, and secrets management.

Functions are hidden by default. Visibility is opt-in via decorators: `@visible` (owner-only), `@public` (anyone, implies visible), or `@protected("auth_func")` (custom auth - calls the named protection function which returns `True`/`False`). See `_build_function_file_mapping()` in `DynamicFunctionManager.py` for how this is enforced. Functions prefixed with `_function`, `_server`, `_admin` are owner-only internals (see `_execute_tool()` in `server.py`).
