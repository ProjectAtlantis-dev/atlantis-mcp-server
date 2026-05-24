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

def _scaffold_starter_functions():
    """Populate an empty dynamic_functions dir with a minimal Home app so new users have something to play with."""
    if os.listdir(FUNCTIONS_DIR):
        return

    home_dir = os.path.join(FUNCTIONS_DIR, "Home")
    os.makedirs(home_dir, exist_ok=True)

    with open(os.path.join(home_dir, "README.md"), "w") as f:
        f.write(
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
        )

    with open(os.path.join(home_dir, "README.py"), "w") as f:
        f.write(
            "from pathlib import Path\n\n\n"
            "@text(\"md\")\n"
            "@visible\n"
            "async def README():\n"
            '    """Simple README"""\n'
            "    return Path(__file__).with_name(\"README.md\").read_text(encoding=\"utf-8\")\n"
        )

    with open(os.path.join(home_dir, "main.py"), "w") as f:
        f.write(
            "import atlantis\n\n\n"
            "@index\n"
            "@visible\n"
            "async def index():\n"
            '    """Demo Home folder"""\n'
            '    return "Welcome to Atlantis!"\n'
        )

    with open(os.path.join(home_dir, "bar.py"), "w") as f:
        f.write(
            "import atlantis\n\n\n"
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
        )

    with open(os.path.join(home_dir, "myTable.py"), "w") as f:
        f.write(
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
        )

    with open(os.path.join(home_dir, "myImage.py"), "w") as f:
        f.write(
            "import os\n"
            "import atlantis\n\n\n"
            "@visible\n"
            "async def myImage():\n"
            '    """Display the happy.png image"""\n'
            "    img_path = os.path.join(os.path.dirname(__file__), \"..\", \"..\", \"..\", \"happy.png\")\n"
            "    await atlantis.client_image(img_path)\n"
        )

    with open(os.path.join(home_dir, "foo.py"), "w") as f:
        f.write(
            "@visible\n"
            "async def foo(a: int, b: int):\n"
            '    """Add two integers"""\n'
            "    return a + b\n"
        )

    with open(os.path.join(home_dir, "hello.py"), "w") as f:
        f.write(
            "import atlantis\n\n\n"
            "@visible\n"
            "async def hello():\n"
            '    """Say hello to the caller"""\n'
            "    caller = atlantis.get_caller() or \"stranger\"\n"
            "    await atlantis.client_log(f\"Hello, {caller}!\")\n"
            '    return f"Hello, {caller}!"\n'
        )

    logger.info("📦 Scaffolded starter dynamic functions in Home/")

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
