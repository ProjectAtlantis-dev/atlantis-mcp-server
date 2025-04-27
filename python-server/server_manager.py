"""
Manages the lifecycle of dynamic MCP servers (configs) by launching them
as background tasks using the MCP Python SDK's stdio transport.
Stores each server config as a JSON file under SERVERS_DIR.
"""
import os
import json
import logging
import asyncio
from typing import Any, Dict, Optional, List

# Assuming the MCP SDK is correctly installed/available in the Python path
# If it's directly from a path, sys.path manipulation might be needed elsewhere
try:
    from mcp import ClientSession, StdioServerParameters, stdio_client
    from mcp.types import TextContent
except ImportError as e:
    # Handle case where SDK might not be installed or path isn't set up
    # This is a basic placeholder; more robust handling might be needed
    logging.error(f"MCP SDK components not found. Please ensure the SDK is installed and accessible: {e}")
    # Define dummy types/classes to allow the rest of the module to load without errors,
    # although functionality will be broken.
    class TextContent: pass
    class ClientSession: pass
    class StdioServerParameters: pass
    async def stdio_client(*args, **kwargs): raise NotImplementedError("MCP SDK not loaded")

from state import SERVERS_DIR, logger

# Ensure SERVERS_DIR exists
os.makedirs(SERVERS_DIR, exist_ok=True)

# --- Tracking for Active Server Tasks ---
# Stores {'server_name': {'task': asyncio.Task, 'params': StdioServerParameters}}
ACTIVE_SERVER_TASKS: Dict[str, Dict[str, Any]] = {}

# --- 1. File Save/Load ---
def _fs_save_server(name: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Saves the provided JSON config dict to {name}.json in SERVERS_DIR.
    Returns the full path if successful, None otherwise.
    """
    safe_name = f"{name}.json"
    file_path = os.path.join(SERVERS_DIR, safe_name)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"💾 Saved server config for '{name}' to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ _fs_save_server: Failed to write {file_path}: {e}")
        return None


def _fs_load_server(name: str) -> Optional[Dict[str, Any]]:
    """
    Loads and returns the JSON config dict from {name}.json in SERVERS_DIR.
    Returns None if not found or error.
    """
    safe_name = f"{name}.json"
    file_path = os.path.join(SERVERS_DIR, safe_name)
    if not os.path.exists(file_path):
        logger.info(f"⚠️ _fs_load_server: Existing config not found for '{name}' at {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ _fs_load_server: Failed to read {file_path}: {e}")
        return None

# --- 2. Config CRUD Operations ---
def server_add(name: str, config: Dict[str, Any]) -> bool:
    """
    Adds a new server config. Returns False if it already exists.
    """
    if _fs_load_server(name) is not None:
        logger.warning(f"Add failed: Server '{name}' already exists.")
        return False
    return _fs_save_server(name, config) is not None


def server_remove(name: str) -> bool:
    """
    Removes the server config by deleting its JSON file. Returns False if missing.
    Also stops the server if it's running.
    """
    # Attempt to stop the server first if it's running
    if name in ACTIVE_SERVER_TASKS:
        logger.info(f"Server '{name}' is running, attempting to stop before removing config.")
        # Create a dummy 'server' object if needed by server_stop signature
        # This might need adjustment based on how server_stop is registered
        loop = asyncio.get_event_loop()
        loop.create_task(server_stop({'name': name}, None)) # Fire and forget stop

    file_path = os.path.join(SERVERS_DIR, f"{name}.json")
    if not os.path.exists(file_path):
        logger.warning(f"Remove failed: Server config for '{name}' does not exist.")
        return False # Config doesn't exist
    try:
        os.remove(file_path)
        logger.info(f"🗑️ Removed server config '{name}' at {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ server_remove: Failed to delete {file_path}: {e}")
        return False


def server_get(name: str) -> Optional[Dict[str, Any]]:
    """
    Returns the server config dict or None if not found.
    """
    return _fs_load_server(name)


def server_list() -> List[str]:
    """
    Lists all server config names available in SERVERS_DIR.
    """
    try:
        files = os.listdir(SERVERS_DIR)
        return [os.path.splitext(f)[0] for f in files if f.endswith('.json')]
    except FileNotFoundError:
        logger.warning(f"Server directory {SERVERS_DIR} not found during list.")
        return []


async def server_set(args: Dict[str, Any], server) -> List[TextContent]:
    """
    MCP handler to add/update a server config.
    Expects args['config']: dict containing {"mcpServers": {"server_name": {...}}}
    The server name is derived from the key within 'mcpServers'.
    If the server is running and the config is updated, it might need restarting.
    """
    logger.debug(f"Received server_set request with args: {args}")

    config_input = args.get('config')

    if isinstance(config_input, str):
        try:
            config = json.loads(config_input)
            logger.debug("Parsed 'config' from JSON string to dictionary.")
        except json.JSONDecodeError as e:
            msg = f"Invalid parameter: 'config' was a string but failed JSON parsing: {e}"
            logger.error(f" server_set: {msg}")
            return [TextContent(type='text', text=msg)]
    elif isinstance(config_input, dict):
        config = config_input # It's already a dictionary

    if not config or not isinstance(config, dict):
        msg = "Missing or invalid parameter: 'config' must be a dictionary."
        logger.error(f"❌ server_set: {msg}")
        return [TextContent(type='text', text=msg)]

    mcp_servers_config = config.get('mcpServers')
    if not mcp_servers_config or not isinstance(mcp_servers_config, dict):
        msg = "Invalid config: Missing or invalid 'mcpServers' dictionary within 'config'."
        logger.error(f"❌ server_set: {msg}")
        return [TextContent(type='text', text=msg)]

    if not mcp_servers_config:
        msg = "Invalid config: 'mcpServers' dictionary cannot be empty."
        logger.error(f"❌ server_set: {msg}")
        return [TextContent(type='text', text=msg)]

    # Extract the server name from the first key in mcpServers
    name = next(iter(mcp_servers_config))
    server_specific_config = mcp_servers_config[name] # The actual config for this server

    # Validate the extracted server-specific config minimally
    if not isinstance(server_specific_config, dict) or 'command' not in server_specific_config:
        msg = f"Invalid config structure for server '{name}' under 'mcpServers'. Must be a dictionary with at least a 'command' key."
        logger.error(f"❌ server_set: {msg}")
        return [TextContent(type='text', text=msg)]

    logger.info(f"Processing server_set for derived name: '{name}'")

    # Check if the server is currently running - potential restart needed notification?
    # Note: We don't automatically restart here, just save the config.
    # Manual restart via server_stop/server_start might be required.
    is_running = name in ACTIVE_SERVER_TASKS
    if is_running:
        logger.warning(f"⚠️ Server '{name}' is currently running. Updating config may require a manual restart.")

    # Save the full original config structure (including 'mcpServers' wrapper)
    # This ensures _fs_load_server retrieves the same structure later
    saved_path = _fs_save_server(name, config) # Save the whole config blob under the derived name

    if saved_path:
        validation = server_validate(name) # Validate the *saved* config
        valid_msg = f"Config for server '{name}' saved successfully."
        if not validation.get('valid'):
            valid_msg += f" WARNING: Config validation failed: {validation.get('error')}"
        else:
            valid_msg += " Config structure appears valid."

        if is_running:
            valid_msg += " Server is running; manual restart might be needed to apply changes."

        return [TextContent(type='text', text=valid_msg)]
    else:
        return [TextContent(type='text', text=f"Failed to save config for server '{name}'.")]


def server_validate(name: str) -> Dict[str, Any]:
    """
    Validates that the *saved* server config JSON has required keys.
    Returns a dict with 'valid':bool and 'error':Optional[str].
    """
    config = _fs_load_server(name)
    if config is None:
        return {'valid': False, 'error': f"Server '{name}' config not found."}
    # Basic required keys
    # 'args' and 'env' are optional in StdioServerParameters technically
    req = ['mcpServers']
    missing = [k for k in req if k not in config]
    if missing:
        return {'valid': False, 'error': f"Missing keys: {missing}"}
    return {'valid': True, 'error': None}


# --- 3. Background Task Runner ---
async def _run_mcp_client_session(name: str, params: StdioServerParameters):
    """Runs the MCP client session in the background using stdio_client."""
    logger.info(f"Background task starting for MCP server '{name}'...")
    session = None # Define session here to potentially use in finally if needed
    try:
        # stdio_client handles process start, stream creation, and cleanup
        async with stdio_client(params) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            # Store session if needed for direct interaction (optional)
            if name in ACTIVE_SERVER_TASKS: # Check if task wasn't cancelled before context entered
                 ACTIVE_SERVER_TASKS[name]['session'] = session
            logger.info(f"✅ Successfully connected to MCP server '{name}' via stdio.")
            # Keep the task alive while the context manager is active
            # The session communication happens within the context
            await asyncio.sleep(float('inf')) # Rely on context exit or cancellation
    except asyncio.CancelledError:
        logger.info(f"🛑 Task for server '{name}' cancelled.")
        # Process termination should be handled by stdio_client's finally block
    except Exception as e:
        logger.error(f"❌ Error in client session task for '{name}': {e}", exc_info=True)
        # Process termination should be handled by stdio_client's finally block
    finally:
        logger.info(f"Background task finished for MCP server '{name}'.")
        # Clean up tracking entry regardless of how the task ended
        if name in ACTIVE_SERVER_TASKS:
             del ACTIVE_SERVER_TASKS[name]
             logger.info(f"Removed '{name}' from active server task tracking.")

# --- 4. Server Start/Stop Operations ---
async def server_start(args: Dict[str, Any], server) -> List[TextContent]:
    """
    MCP handler to start a managed MCP server process as a background task.
    Expects args['name']: str.
    """
    name = args.get('name')
    if not name or not isinstance(name, str):
        msg = "Missing or invalid parameter: 'name' must be str."
        logger.error(f"❌ server_start: {msg}")
        return [TextContent(type='text', text=msg)]

    if name in ACTIVE_SERVER_TASKS:
        logger.warning(f"⚠️ server_start: Server '{name}' is already running or starting.")
        return [TextContent(type='text', text=f"Server '{name}' is already running.")]

    config = _fs_load_server(name)
    if config is None:
        msg = f"Server '{name}' config not found."
        logger.error(f"❌ server_start: {msg}")
        return [TextContent(type='text', text=msg)]

    validation = server_validate(name)
    if not validation.get('valid', False):
        error = validation.get('error', 'Unknown error')
        msg = f"Invalid config for '{name}': {error}"
        logger.error(f"❌ server_start: {msg}")
        return [TextContent(type='text', text=msg)]

    try:
        # Prepare parameters for stdio_client
        params = StdioServerParameters(
            command=config['command'],
            args=config.get('args', []),
            env=config.get('env', None), # Pass None if not present, SDK handles default
            cwd=config.get('cwd', None)  # Add cwd support to config
        )
    except Exception as e: # Catch potential Pydantic validation errors
        logger.error(f"❌ server_start: Failed to create StdioServerParameters for '{name}': {e}", exc_info=True)
        return [TextContent(type='text', text=f"Failed to prepare start parameters for '{name}': {e}")]

    # Start the background task
    logger.info(f"Attempting to start background task for server '{name}'...")
    task = asyncio.create_task(_run_mcp_client_session(name, params))

    # Store task info immediately
    ACTIVE_SERVER_TASKS[name] = {'task': task, 'params': params}

    # Return success - PID is not available synchronously here
    return [TextContent(type='text', text=f"Background task started to connect to server '{name}'.")]


async def server_stop(args: Dict[str, Any], server) -> List[TextContent]:
    """
    MCP handler to stop a managed MCP server process task.
    Expects args['name']: str.
    """
    name = args.get('name')
    if not name or not isinstance(name, str):
        msg = "Missing or invalid parameter: 'name' must be str."
        logger.error(f"❌ server_stop: {msg}")
        return [TextContent(type='text', text=msg)]

    if name not in ACTIVE_SERVER_TASKS:
        logger.warning(f"⚠️ server_stop: Server '{name}' not found in active tasks.")
        return [TextContent(type='text', text=f"Server '{name}' is not running or not managed.")]

    task_info = ACTIVE_SERVER_TASKS.get(name)
    if not task_info or 'task' not in task_info:
        logger.error(f"❌ server_stop: Inconsistent state for server '{name}'. No task found.")
        # Clean up if entry exists but is broken
        if name in ACTIVE_SERVER_TASKS:
            del ACTIVE_SERVER_TASKS[name]
        return [TextContent(type='text', text=f"Error stopping server '{name}': Inconsistent state.")]

    task = task_info['task']
    if task.done():
        logger.info(f"🧹 Task for server '{name}' was already finished. Cleaning up entry.")
        # Cleanup potentially missed by the finally block
        if name in ACTIVE_SERVER_TASKS:
            del ACTIVE_SERVER_TASKS[name]
        return [TextContent(type='text', text=f"Server '{name}' task was already finished.")]
    else:
        logger.info(f"Attempting to cancel task for server '{name}'...")
        task.cancel()
        # Give cancellation a moment to potentially propagate, though the
        # finally block in the task handles the actual cleanup.
        try:
            # Wait briefly to see if cancellation completes quickly
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            logger.info(f"✅ Cancellation successful for server '{name}' task.")
        except asyncio.TimeoutError:
            logger.info(f"⏳ Task for server '{name}' cancellation initiated, may take time to fully stop.")
        except Exception as e:
            logger.error(f"❓ Unexpected error while waiting for cancellation of '{name}': {e}")

        # The finally block in _run_mcp_client_session should remove the entry.
        # If immediate feedback is needed, could remove here, but safer to let task clean itself up.
        # if name in ACTIVE_SERVER_TASKS:
        #     del ACTIVE_SERVER_TASKS[name]

        return [TextContent(type='text', text=f"Stop request sent to server '{name}'.")]
