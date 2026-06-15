#!/usr/bin/env python3
import logging
import os
import sys
import uuid
from ColoredFormatter import ColoredFormatter, ContextFilter

# --- REMOVED basicConfig ---

# Determine log level from environment variable if set, otherwise default to INFO
default_level = os.environ.get('LOG_LEVEL', 'INFO')
log_level = getattr(logging, default_level, logging.INFO)

# Get our app logger
logger = logging.getLogger("mcp_server")
logger.setLevel(log_level)

# --- ADDED Handler setup ---
# Create console handler
ch = logging.StreamHandler(sys.stdout) # Use stdout
ch.setLevel(log_level) # Process all messages from logger

# Add context filter (injects [reqId-shell] into every log line)
ch.addFilter(ContextFilter())

# Set the custom formatter
ch.setFormatter(ColoredFormatter())

# Add handler to the logger
logger.addHandler(ch)

# Configure the root logger to also use this handler and level
root_logger = logging.getLogger()
# Set root logger level (e.g., INFO to see messages from message_db.py)
root_logger.setLevel(logging.INFO)

# Add the same handler to the root logger if it doesn't have any
# This ensures messages from 'logging.info()' etc. in other modules are also seen
if not root_logger.hasHandlers():
    root_logger.addHandler(ch)

# Prevent logging from propagating to the root logger
# (important if basicConfig was ever called or might be by libraries)
logger.propagate = False
# --- End Handler setup ---

def update_log_level(level_name):
    """Update the logging level of both loggers and handlers.

    Args:
        level_name: String name of logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    level = getattr(logging, level_name)  # Convert string to logging level constant
    logger.setLevel(level)
    ch.setLevel(level)

    # Also update root logger if it's using our handler
    if level_name != "DEBUG":  # Keep root at INFO or higher
        root_logger.setLevel(level)

    logger.info(f"Log level updated to {level_name}")

# Directory to store dynamic function files
FUNCTIONS_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_functions"))

# Create functions directory if it doesn't exist, and scaffold starter files
# for new users who haven't set up their own functions repo yet.
os.makedirs(FUNCTIONS_DIR, exist_ok=True)

def _write_starter_file_if_missing(target_dir: str, filename: str, contents: str) -> bool:
    path = os.path.join(target_dir, filename)
    if os.path.exists(path):
        return False
    with open(path, "w") as f:
        f.write(contents)
    return True

def _copy_starter_asset_if_missing(target_dir: str, filename: str) -> bool:
    source_path = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename))
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path) or not os.path.exists(source_path):
        return False
    with open(source_path, "rb") as source, open(target_path, "wb") as target:
        target.write(source.read())
    return True

# Canonical source for the Home app. Home is NOT tracked in git — it is scaffolded
# here so users can edit it freely without ever hitting merge conflicts on pull.
_HOME_MAIN_PY = '''\
import atlantis
import logging
from pathlib import Path

logger = logging.getLogger("dynamic_function")

# % whoami


@index
@visible
async def index(session_key: str):
    """Docs n stuff"""
    pass


@text("md")
@visible
async def README():
    """Show MULTIX instructions"""

    await atlantis.client_log("README running")

    md_path = Path(__file__).parent / "MULTIX.md"
    return md_path.read_text()


@text("md")
@visible
async def README_LOBSTER():
    """Show Lobster MCP tool instructions."""

    await atlantis.client_log("README_LOBSTER running")

    return """# Atlantis Lobster MCP Tools

Lobster exposes a small local MCP surface that forwards work to the connected Atlantis cloud session.

Tools:
- `readme`: show the Multix help text.
- `command`: send an Atlantis command. If the command has no prefix, `/` is added automatically.
- `chat`: send a plain chat message.

Common command prefixes:
- `/`: Atlantis slash command.
- `@`: tool/function call.
- `~`: routed tool/function call.
"""
'''

_HOME_MULTIX_MD = '''\
# Atlantis MCP tools

## Overview

Each Atlantis MCP acts as a filesystem node for Multix, our nix-like 'operating system' for future Greenland. However, since bots rely on tools, each folder in Multix contains functions instead of files. We feel this approach is closer to the original 1960s vision for UNIX, namely Multics.

- **readme** - this file
- **command** - use this to enter an Atlantis command
- **chat** - use this to just talk into the chat

## Commands

`command` lets humans or bots send commands to Multix. All commands should start with `/`. They kinda follow a Linux style shell approach. In fact, you can enable terminal mode to enter the Multix terminal directly and avoid having to prefix everything with slashes. The main difference is that each MCP exposes a virtual filesystem of sorts but of functions instead of files. The file containers are essentially unwrapped and then hotloaded so they are call ready. Note that the default container is usually main.py and more than one function can be in the same file. Generally, you should not have to care about the containing file except for versioning.

Some interesting commands to get you started:

- `help` - shows all the keywords (**warning:** calling with no arguments dumps a LOT of output; prefer `help <topic>` instead)
- `help <topic>` - does fuzzy search of both keywords and tools
- `ls` - list contents of current folder
- `dir` - list contents from root (ignores current folder)
- `tree` - list all contents from current location
- `pwd` - show current directory (if reconnecting)
- `search` - search functions by description
- `history` - show shell history
- `whoami` - show your username
- `cd` - change into a folder, some special notes:
  - `cd /` - go to root, root is arranged by user
  - `cd ~` - go to your home, arranged by connected servers
  - `cd ..` - go back up one folder
  - `cd H*me` - globs work within a segment (`*` never crosses `/`)
- `add` - add a function in the current location
- `edit` - DO NOT USE, use `codeset` instead (see below)
- `rls` - list the connected remotes

## Tools

You will start at the root of the Atlantis virtual filesystem arranged by usernames, and then you go into a user's home folder, and then connected MCP server for that user. You cannot go into disconnected servers.

To run a tool in the current folder, you can simply use `@name` plus any params, much like a JavaScript function e.g. `@foo` or `@foo(3,100)`

- `cat` - retrieves the text of a function
- `codeset` - sets the content of a function

## Paths and Globs

Paths follow Linux conventions. `/` is the ONLY path separator; dots are ordinary name characters.

- `*` and `?` glob WITHIN a single path segment — `*` never crosses a `/`
- `**` as a whole segment spans any number of folders (globstar)
- `.` and `..` resolve against your current directory
- Matching prefers the exact case first, then falls back to case-insensitive

Anchors (where a path starts):

- no prefix - relative to your current folder
- `/` or `%` - global root (root is arranged by user, so the first segment is a username)
- `~` - your home; `~name` is user *name*'s home (like Linux)
- `$` - root of the remote you are currently inside

## Tool Prefixes

- `@foo` - run `foo` from the current folder (searches PATH folders, nearest match wins)
- `@App/foo` - run `foo` in the App subfolder of the current folder
- `%user/remote/App/foo` or `/user/remote/App/foo` - fully qualified call
- `~/**/coffee` - your own `coffee`, anywhere under your home
- `%**/coffee` - the first `coffee` across all users (anywhere in the tree)
- `%*foo` - note: a single `*` is one segment, so this matches *users* ending in `foo`, not functions
- `$Tools/coffee` - `coffee` inside the `Tools` folder at the current remote's root

## Search Terms

Search terms are just glob paths. You can call a deep function without cd-ing to it first, as long as it resolves uniquely, e.g. `/brickhouse/terrain/InWork/foo` or `**/InWork/foo`.

If you just say `foo` from the top level it could be ambiguous which one you mean; the shell will show the candidates so you can pick a fuller path.

## Named Function Parameters

While purely positional parameters usually work, it is better to use explicitly named JSON arguments (parentheses are optional) to avoid escaping issues:

- `foo { x: 3, name: "chicago" }`
- `codeset { searchTerm: "bar", contents: "async def bar(): ... rest of code here ..." }`

The `help` command also provides parameter info.

## How the Description Field Is Populated

The first comment in a Python function is the description displayed in `search` and other various commands.

## SQL Select

When a command or tool returns tabular data, that is saved into a pseudo-table called `prior` and you can run `/select <cols> from prior where ...` if you want to narrow down prior results.

## MCP Servers

Be aware you can list, start, and stop classic MCP servers as well (which are more like plug-ins).

## Cursors

In practical programming terms, a cursor does what a call stack what normally do - it's simply a place to hold function parameters so you don't have to constantly type them in.  That is if you have a function `foo(x,y)` and the cursor already hold `x=3` then you only need to supply `y` on the command line.

Why cursors? Well although tools are functions, we don't have an interactive way to compose functions in a categories (think category theory) since there's no functional programming (FP) here and everyone hates FP anyway. So cursors play the role of monads ie the 'glue' btw functions. If you are familiar with Scala ZIO, think ZLayers; if you are familiar with Unison, think ability stack. Setting x=3 and then x=4 pushes two structures onto the Multix cursor stack and you can pop later.
'''


def _scaffold_starter_functions():
    """Populate the Home and Demo apps once so new users have something to play with."""
    _write_starter_file_if_missing(
        FUNCTIONS_DIR,
        "README.md",
        (
            "# Dynamic Functions\n\n"
            "This is your personal tool code directory. Everything here is YOUR code,\n"
            "separate from the Atlantis server platform.\n\n"
            "We recommend keeping this directory as its own git repo and symlinking it in:\n\n"
            "    cd python-server\n"
            "    rm -rf dynamic_functions\n"
            "    ln -s ~/my-atlantis-functions dynamic_functions\n\n"
            "This makes it clear to both you and AI coding agents (Claude, Codex, etc.)\n"
            "where Atlantis platform code ends and your tool code begins.\n\n"
            "See README.dynamic_functions.md in the server directory for authoring details.\n"
        ),
    )

    # One-time marker. Despite the legacy name it gates BOTH the Home and Demo
    # scaffolds; existing installs that already have it keep their on-disk apps.
    marker_path = os.path.join(FUNCTIONS_DIR, ".demo_scaffolded")
    if os.path.exists(marker_path):
        return

    created = []

    # Home app — landing page plus the Multix/lobster help text.
    home_dir = os.path.join(FUNCTIONS_DIR, "Home")
    os.makedirs(home_dir, exist_ok=True)
    if _write_starter_file_if_missing(home_dir, "main.py", _HOME_MAIN_PY):
        created.append("Home/main.py")
    if _write_starter_file_if_missing(home_dir, "MULTIX.md", _HOME_MULTIX_MD):
        created.append("Home/MULTIX.md")

    demo_dir = os.path.join(FUNCTIONS_DIR, "Demo")
    os.makedirs(demo_dir, exist_ok=True)

    if _write_starter_file_if_missing(
        demo_dir,
        "main.py",
        (
            "import atlantis\n"
            "import logging\n\n"
            'logger = logging.getLogger("dynamic_function")\n\n\n'
            "@index\n"
            "@visible\n"
            "async def index():\n"
            '    """Demo app"""\n'
            '    return "Welcome to the Atlantis demo app!"\n'
        ),
    ):
        created.append("Demo/main.py")

    if _write_starter_file_if_missing(
        demo_dir,
        "bar.py",
        (
            "import atlantis\n"
            "import logging\n\n"
            'logger = logging.getLogger("dynamic_function")\n\n\n'
            "@visible\n"
            "async def bar():\n"
            '    """Print caller context info"""\n'
            "    lines = [\n"
            '        f"user_game_id: {atlantis.get_user_game_id()}",\n'
            '        f"caller_sid:   {atlantis.get_caller()}",\n'
            '        f"caller_shell: {atlantis.get_caller_shell_path()}",\n'
            '        f"exec_shell:   {atlantis.get_exec_shell_path()}",\n'
            '        f"request_id:   {atlantis.get_request_id()}",\n'
            '        f"session_key:  {atlantis.get_session_key()}",\n'
            '        f"entry_point:  {atlantis.get_entry_point_name()}",\n'
            "    ]\n"
            '    return "\\n".join(lines)\n'
        ),
    ):
        created.append("Demo/bar.py")

    if _write_starter_file_if_missing(
        demo_dir,
        "myTable.py",
        (
            "import atlantis\n"
            "import logging\n\n"
            'logger = logging.getLogger("dynamic_function")\n\n\n'
            "@visible\n"
            "async def myTable():\n"
            '    """Return a table of Disney characters"""\n'
            "    # this does not call client_data to display a nicely formatted table, it simply returns an array and lets Atlantis render w default formatting\n"
            "    return [\n"
            '        {"id": 1, "name": "Goofy", "favorite_food": "pizza"},\n'
            '        {"id": 2, "name": "Donald", "favorite_food": "corn"},\n'
            '        {"id": 3, "name": "Simba", "favorite_food": "gazelle"},\n'
            '        {"id": 4, "name": "Stitch", "favorite_food": "coconut cake"},\n'
            '        {"id": 5, "name": "Ratatouille", "favorite_food": "ratatouille"},\n'
            "    ]\n"
        ),
    ):
        created.append("Demo/myTable.py")

    if _write_starter_file_if_missing(
        demo_dir,
        "myImage.py",
        (
            "import os\n"
            "import atlantis\n"
            "import logging\n\n"
            'logger = logging.getLogger("dynamic_function")\n\n\n'
            "@visible\n"
            "async def myImage():\n"
            '    """Display the happy.png image"""\n'
            "    img_path = os.path.join(os.path.dirname(__file__), \"..\", \"..\", \"..\", \"happy.png\")\n"
            "    await atlantis.client_image(img_path)\n"
        ),
    ):
        created.append("Demo/myImage.py")

    if _write_starter_file_if_missing(
        demo_dir,
        "myVideo.py",
        (
            "import os\n"
            "import atlantis\n"
            "import logging\n\n"
            'logger = logging.getLogger("dynamic_function")\n\n\n'
            "@visible\n"
            "async def myVideo():\n"
            '    """Display the TaffyWide.mp4 video"""\n'
            "    video_path = os.path.join(os.path.dirname(__file__), \"..\", \"..\", \"..\", \"TaffyWide.mp4\")\n"
            "    await atlantis.client_video(video_path)\n"
        ),
    ):
        created.append("Demo/myVideo.py")

    if _write_starter_file_if_missing(
        demo_dir,
        "win.py",
        (
            "import base64\n"
            "import json\n"
            "import mimetypes\n"
            "import os\n\n"
            "import atlantis\n\n\n"
            "def _audio_data_url(audio_path: str) -> str:\n"
            "    mime_type, _ = mimetypes.guess_type(audio_path)\n"
            "    if not mime_type or not mime_type.startswith(\"audio/\"):\n"
            "        mime_type = \"audio/mpeg\"\n"
            "    with open(audio_path, \"rb\") as audio:\n"
            "        encoded = base64.b64encode(audio.read()).decode(\"ascii\")\n"
            "    return f\"data:{mime_type};base64,{encoded}\"\n\n\n"
            "@visible\n"
            "async def win_background() -> None:\n"
            "    \"\"\"Test the Windows 95 forest tile as a repeated terminal background.\"\"\"\n"
            "    forest_path = os.path.join(os.path.dirname(__file__), \"win_forest.jpg\")\n"
            "    await atlantis.set_background(\n"
            "        forest_path,\n"
            "        vertical_align=\"top\",\n"
            "        horizontal_align=\"left\",\n"
            "        background_repeat=\"repeat\",\n"
            "        background_size=\"auto\",\n"
            "    )\n"
            "    chime_path = os.path.join(os.path.dirname(__file__), \"win95chime.mp3\")\n"
            "    chime_url = _audio_data_url(chime_path)\n"
            "    await atlantis.client_terminal_script(f\"\"\"\n"
            "(function(){{\n"
            "  var audio = new Audio({json.dumps(chime_url)});\n"
            "  audio.preload = \"auto\";\n"
            "  audio.play().catch(function(err) {{\n"
            "    console.warn(\"Unable to play Windows 95 chime\", err);\n"
            "  }});\n"
            "}})();\n"
            "\"\"\")\n"
        ),
    ):
        created.append("Demo/win.py")
    if _copy_starter_asset_if_missing(demo_dir, "win_forest.jpg"):
        created.append("Demo/win_forest.jpg")
    if _copy_starter_asset_if_missing(demo_dir, "win95chime.mp3"):
        created.append("Demo/win95chime.mp3")

    if _write_starter_file_if_missing(
        demo_dir,
        "foo.py",
        (
            "import atlantis\n"
            "import logging\n\n"
            'logger = logging.getLogger("dynamic_function")\n\n\n'
            "@visible\n"
            "async def foo(x: int, y: int):\n"
            '    """Add two integers"""\n'
            "    return x + y\n"
        ),
    ):
        created.append("Demo/foo.py")

    if _write_starter_file_if_missing(
        demo_dir,
        "hello.py",
        (
            "import atlantis\n"
            "import logging\n\n"
            'logger = logging.getLogger("dynamic_function")\n\n\n'
            "@visible\n"
            "async def hello():\n"
            '    """Say hello to the caller"""\n'
            "    caller = atlantis.get_caller() or \"stranger\"\n"
            "    await atlantis.client_log(f\"Hello, {caller}!\")\n"
            '    return f"Hello, {caller}!"\n'
        ),
    ):
        created.append("Demo/hello.py")

    if created:
        logger.info(f"📦 Scaffolded starter dynamic functions: {', '.join(created)}")
    with open(marker_path, "a"):
        pass

_scaffold_starter_functions()

# Directory to store dynamic server configs
SERVERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_servers")
# Create servers directory if it doesn't exist
os.makedirs(SERVERS_DIR, exist_ok=True)

# Server configuration
HOST = "127.0.0.1"  # Listen on localhost only for security
PORT = 8000

SERVER_REQUEST_TIMEOUT = 3600.0 # Seconds to wait for proxied server requests and awaitable client commands (1 hour for long-running cloud jobs)

SERVER_UUID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_uuid")


def load_or_create_server_uuid():
    """Return the stable UUID for this server, creating it on first startup."""
    if os.path.exists(SERVER_UUID_PATH):
        with open(SERVER_UUID_PATH, 'r') as f:
            server_uuid = f.read().strip()

        try:
            uuid.UUID(server_uuid)
            return server_uuid
        except ValueError:
            logger.warning(f"⚠️ Invalid server UUID in {SERVER_UUID_PATH}; generating a replacement")

    server_uuid = str(uuid.uuid4())
    temp_path = f"{SERVER_UUID_PATH}.tmp"
    with open(temp_path, 'w') as f:
        f.write(f"{server_uuid}\n")
    os.replace(temp_path, SERVER_UUID_PATH)
    return server_uuid

# Flags to track server state
is_shutting_down = False
