# Dynamic Functions

Dynamic functions are Python files under `dynamic_functions/` that Atlantis
exposes as MCP tools. Home, Chat, Bot, Setup, Media and Demo ship with the
repo. Your own apps are ignored by Git, so an app can be its
own repository cloned straight into `dynamic_functions/` (see the main README).

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
  - `@upload` stores files dropped on the chat input and returns what to insert there.
  - `@enter` / `@exit` run when a terminal `cd` enters or leaves the folder's
    subtree (the folder needs an `@index`). Only the nearest folder with a hook
    runs; the ones above it don't.
- **Robot folder:** an app keeps the tools its chat drives (`@enter`, `@exit`,
  `get_status`, `record_turn`) in a `Robot/` subfolder, e.g.
  `Setup/Robot/main.py`. Hooks there belong to the app folder above it, so
  `cd Setup` runs `Setup/Robot/on_enter`. `Robot/` needs its own `main.py`
  with a `@visible` `index()`, like any folder. Bots are never offered Robot
  tools. `Chat/<game>/init.py` makes the game a zone by naming the folder,
  `ROBOT = "$Setup/Robot"`. Chat then:
  - runs `get_status(command_prefix, current_folder)` before every bot round,
    so it must only read state. Chat adds who the user is, whether the bot
    has met them in this game, and how they type.
  - runs `record_turn`, if the Robot has one, once after a bot's reply goes
    out. Chat records the visit itself.
  - loads the zone's own tools into every bot round. If the Robot has a
    `Tools/` folder (Demo), the bot gets only those, a curated set on top
    of what humans run. If it has none (Setup), the bot gets the app
    folder's own tools, the same ones humans see there. Either way, Chat
    adds the folders the Robot's `tool_folders` returns, if it has one.
  - Setup is the package installer: `Setup/Robot` shows each installed
    package's own `<Package>/Robot/get_status` after its own steps, and its
    `tool_folders` names each package's `Robot/Tools`. Package Robots
    (`Media/Robot`, `Terrain/Robot`) have no hooks; a Robot can't install
    itself, so cloning a package stays in Setup.
  - The hooks call `Chat.zone`'s `zone_enter(background)` and `zone_exit()`.
- **Structure:** `@index`, `@text("md")`, `@location(...)`, `@price(...)`,
  `@copy`, `@dynamic`.
- **Modifiers:** `@exclude` (hidden from fuzzy search), `@button`.
- **Keywords:** `@keyword("name")` and `@keywords(["a", "b"])` tag a tool
  with search keywords. They don't make it visible. All keywords on a function
  go to the cloud as one comma-delimited `keyword` annotation (`"a,b"`). Bad
  arguments skip the function.

`@app` is obsolete, because the folder name is the app.

## Troubleshooting

If a function doesn't show up, check the decorator, the folder's `main.py`
index, syntax errors, and duplicate names in the app. Load and execution
failures are in `python-server/runServer.log`.
