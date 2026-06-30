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

def _scaffold_starter_functions():
    """Populate the Demo app once so new users have something to play with."""
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

    # One-time marker. Despite the legacy name it gates the Demo scaffold;
    # existing installs that already have it keep their on-disk apps.
    marker_path = os.path.join(FUNCTIONS_DIR, ".demo_scaffolded")
    if os.path.exists(marker_path):
        return

    created = []

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
            "async def bar() -> dict[str, object]:\n"
            '    """Return caller context info"""\n'
            "    return {\n"
            '        "user_game_id": atlantis.get_user_game_id(),\n'
            '        "caller_sid": atlantis.get_caller(),\n'
            '        "caller_shell": atlantis.get_caller_shell_path(),\n'
            '        "exec_shell": atlantis.get_exec_shell_path(),\n'
            '        "request_id": atlantis.get_request_id(),\n'
            '        "session_key": atlantis.get_session_key(),\n'
            '        "entry_point": atlantis.get_entry_point_name(),\n'
            "    }\n"
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
            "async def myTable() -> list[dict[str, object]]:\n"
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
            "    # Use one-shot script so re-engaging the terminal does not replay old chimes.\n"
            "    await atlantis.client_script(f\"\"\"\n"
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
            "async def foo(x: int, y: int) -> int:\n"
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
            "async def hello() -> None:\n"
            '    """Say hello to the caller"""\n'
            "    caller = atlantis.get_caller() or \"stranger\"\n"
            "    await atlantis.client_log(f\"Hello, {caller}!\")\n"
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
