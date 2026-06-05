# Dynamic Functions

Dynamic functions are Python files under `dynamic_functions/` that Atlantis exposes as MCP tools. This directory is not part of the server repo — it is gitignored and should be your own separate repository, symlinked into `python-server/dynamic_functions` (see the main README for setup). The runtime scans the tree, reads top-level functions, builds MCP schemas from signatures/docstrings, and loads the target module when a tool is called.

This document is intentionally an orientation guide, not a full API dump. The authoritative implementation lives in `DynamicFunctionManager.py`; the runtime helpers are in `atlantis.py`. The full `atlantis.*` API (including browser callbacks) is documented in `README.atlantis_api.md`, and security details in `README_SECURITY.md`.

## Minimal Tool

```python
import atlantis

@visible
async def add(x: float, y: float) -> float:
    """Add two numbers."""
    await atlantis.client_log(f"{x} + {y}")
    return x + y
```

Functions are hidden unless they have a visibility/tool decorator. Plain helper functions can live in the same file and are ignored by tool discovery.

## Discovery Rules Worth Knowing

Atlantis scans `.py` and `.txt` files recursively under `dynamic_functions/`. Python tools come from top-level `def` or `async def` functions; `.txt` files become static text tools.

Subdirectories are apps. Nested directories become nested app names. Internally the filesystem uses slash paths, while some management APIs accept dot notation.

A folder is visible only if it has a `main.py` containing a visible `index()` function. This is easy to miss: a decorated function inside a folder may still be invisible if the folder itself has no visible index.

Function names must be unique within the same app. If the same function appears in multiple files for one app, all copies are dropped from the tool list until the duplicate is fixed.

Tool schemas are generated from type hints and docstrings. Untyped parameters default to strings. Parameters without defaults are required; parameters with defaults are optional.

## Decorators

### Visibility Decorators
- **`@visible`** - Make function visible in tools list (owner-only access)
- **`@public`** - Make function publicly accessible to all users (no authorization)
- **`@protected("func_name")`** - Make function visible with custom authorization
- **No decorator** - Function is hidden by default, not exposed as tool

### Callback Decorators (auto-visible)
- **`@chat`** - Chat callback that receives transcript/tools and calls LLM
- **`@tick`** - Tick scheduling is managed by the MCP server; this decorator lets a user trigger a tick manually for debugging
- **`@session`** - Session callback, fired when a user resumes or joins an existing game
- **`@game`** - Game callback, fired once when a brand new game is created (zero events, first session)

### Structural Decorators (auto-visible)
- **`@index`** - Marks a directory index tool
- **`@text("content_type")`** - Text content tool (e.g. `@text("markdown")`)
- **`@location(name="location_name")`** - Associate with a location
- **`@price(per_call=X, per_sec=Y)`** - Set pricing per call or per second
- **`@copy`** - Allow non-owners to view function source via `_function_get` (based on visibility rules)

### Modifier Decorators
- **`@exclude`** - Exclude function from fuzzy search results (still visible in tool catalog/listing)
- **`@button`** - Adds button on dashboard card
- **`@dynamic`** - Marks a function as a dynamic folder provider. The function appears as a subfolder; generated child tools are not emitted yet.

### Deprecated
- **`@hidden`** - Obsolete (functions are hidden by default without @visible)
- **`@app(name="app_name")`** - Obsolete (folder name determines app)

## Atlantis Runtime

Dynamic functions import `atlantis` for client communication, request context, callbacks, and shared state.

Common patterns:

```python
await atlantis.client_log("status")
await atlantis.client_markdown("# Markdown")
await atlantis.client_html("<button>OK</button>")
await atlantis.client_image("/path/to/image.png")

caller = atlantis.get_caller()
```

`atlantis.server_shared` persists server-wide objects across dynamic-function reloads. `atlantis.session_shared` stores values scoped to the current session and raises if there is no session context.

For the full helper surface, read `atlantis.py`; the functions there are short and usually self-describing.

Generally however, it is preferable to run cloud commands directly instead of loading up atlantis.py w wrappers that do the same thing.

## Browser Callbacks

Rendered HTML can call back into Python with `sendChatter(window._accessToken, target, data)`. The payload keys become keyword arguments, so JavaScript key names must exactly match the Python callback parameters.

The target is a routing string such as:

```javascript
await sendChatter(window._accessToken, '$**Home**handle_click', {
  message: 'hello'
});
```

See the **Browser → Dynamic Function callbacks** section of `README.atlantis_api.md` for the current callback contract and examples. The old `html_response` pattern is not the current canonical path.

There is also a runtime-registered callback path in `atlantis.py`: `client_onclick(key, callback)` and `client_upload(key, callback)` register Python callables under a key, and the built-in `_public_click` / `_public_upload` tools route browser events back to those callbacks. Use this when the UI only needs to trigger a server-side callable by key instead of addressing a named dynamic function with `sendChatter`.

## Management Tools

Atlantis includes internal MCP tools for editing and inspecting dynamic functions. They are listed by `tools/list` like everything else.

Those tools operate on whole files where appropriate. For example, `_function_set` updates the file containing the target function, so do not assume it patches only one function body.

## Troubleshooting

If a function does not show up, check these first:

- Does it have a visibility/tool decorator?
- Is it in a folder whose `main.py` has a visible `index()`?
- Is there a syntax error in the file?
- Is there a duplicate function name in the same app?
- Did the app name passed to a management tool match the folder path?

Runtime load or execution failures are logged by the server and may also create function-specific logs under `dynamic_functions/`.

## Architecture Notes

The dynamic function system has one practical source of truth: the function-to-file mapping built by `DynamicFunctionManager._build_function_file_mapping()`.

That mapping drives both discovery and execution. `server.py` turns mapping metadata into MCP `Tool` objects for `tools/list`, authorizes calls, and delegates actual Python execution back to `DynamicFunctionManager.function_call()`.

Local Lobster clients may see cloud-provided pseudo tools when the server is connected to the cloud. Cloud-side calls execute the local dynamic functions directly.
