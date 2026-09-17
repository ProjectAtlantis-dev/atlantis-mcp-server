# Dynamic Functions

Dynamic functions are Python files under `dynamic_functions/` that Atlantis
exposes as MCP tools. Home, Chat and Terrain ship with the repo, and Demo is
created on first run. Your own apps are ignored by Git, so you can symlink one
in from its own repository (see the main README).

The loader lives in `DynamicFunctionManager.py`, and decorators are defined
there. The runtime API is covered in [README.atlantis_api.md](README.atlantis_api.md),
and trust boundaries in [README_SECURITY.md](README_SECURITY.md).

## Minimal Tool

```python
import atlantis

@visible
async def add(x: float, y: float) -> float:
    """Add two numbers."""
    await atlantis.client_log(f"{x} + {y}")
    return x + y
```

The docstring becomes the tool description, and type hints become the schema.
Untyped parameters are strings, and parameters with defaults are optional.

## Rules That Bite

- **Hidden by default:** undecorated functions are never exposed, so plain
  helpers can share a file with tools.
- **Folders need an index:** a folder is visible only if its `main.py` has a
  visible `index()`. A decorated function in a folder without one stays invisible.
- **Folders are apps:** nested folders become nested app names.
- **Unique names per app:** if one app defines the same function name in two
  files, every copy disappears from the tool list until you fix it.
- **Whole-file edits:** `_function_set` and `_function_get` work on the whole
  file, not just the one function.
- **`.txt` files** become static text tools.

## Decorators

- **Visibility:**
  - `@visible` is owner-only.
  - `@public` is open to anyone.
  - `@protected("fn")` uses a custom check.
- **Lifecycle callbacks:**
  - `@homepage` runs at startup.
  - `@preflight` runs before the other session callbacks.
  - `@game` runs when a new game is created.
  - `@session` runs when a user joins or resumes.
  - `@chat` handles chat.
  - `@tick` triggers a tick manually.
  - `@file` handles file storage.
- **Structure:** `@index`, `@text("md")`, `@location(...)`, `@price(...)`,
  `@copy`, `@dynamic`.
- **Modifiers:** `@exclude` (hidden from fuzzy search), `@button`.

`@app` is obsolete, because the folder name is the app.

## Troubleshooting

If a function doesn't show up, check the decorator, the folder's `main.py`
index, syntax errors, and duplicate names in the app. Load and execution
failures are in `python-server/runServer.log`.
