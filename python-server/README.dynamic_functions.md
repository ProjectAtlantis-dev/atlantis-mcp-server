# Dynamic Functions

Dynamic functions are Python files under `dynamic_functions/` that Atlantis exposes as MCP tools. The runtime scans the tree, reads top-level functions, builds MCP schemas from signatures/docstrings, and loads the target module when a tool is called.

This document is intentionally an orientation guide, not a full API dump. The authoritative implementation lives in `DynamicFunctionManager.py`; the runtime helpers are in `atlantis.py`. Browser callback and security details are documented separately in `README.onclick_callbacks.md` and `README_SECURITY.md`.

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

The current decorator set is defined near the top of `DynamicFunctionManager.py` and installed into dynamic-function modules at load time. Use the source as the canonical list.

The important split is:

- `@visible` exposes an owner-facing tool.
- `@public` exposes a tool for public calling.
- `@protected("auth_function")` exposes a tool behind a custom authorization function.
- `@copy` affects whether non-owners can retrieve source through `_function_get`; it does not affect call permissions.

Other decorators provide metadata or specialized behavior. They exist in the runtime; avoid re-documenting their full behavior here unless the code changes to make that useful.

## Atlantis Runtime

Dynamic functions import `atlantis` for client communication, request context, callbacks, and shared state.

Common patterns:

```python
await atlantis.client_log("status")
await atlantis.client_markdown("# Markdown")
await atlantis.client_html("<button>OK</button>")
await atlantis.client_image("/path/to/image.png")

caller = atlantis.get_caller()
session_id = atlantis.get_session_id()
game_id = atlantis.get_game_id()
```

`atlantis.server_shared` persists server-wide objects across dynamic-function reloads. `atlantis.session_shared` stores values scoped to the current session and raises if there is no session context.

For the full helper surface, read `atlantis.py`; the functions there are short and usually self-describing.

## Browser Callbacks

Rendered HTML can call back into Python with `sendChatter(window._accessToken, target, data)`. The payload keys become keyword arguments, so JavaScript key names must exactly match the Python callback parameters.

The target is a routing string such as:

```javascript
await sendChatter(window._accessToken, '$**Home**handle_click', {
  message: 'hello'
});
```

Use `README.onclick_callbacks.md` for the current callback contract and examples. The old `html_response` pattern is not the current canonical path.

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
