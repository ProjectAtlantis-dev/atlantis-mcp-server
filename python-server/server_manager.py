"""
Manages the lifecycle of dynamic MCP servers (configs) by launching them
as background tasks using the MCP Python SDK's stdio transport.
Stores each server config as a JSON file under SERVERS_DIR.
"""
import os
import json
import logging
import asyncio
import shutil
import datetime
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
# Define and ensure OLD_DIR exists for backups
OLD_DIR = os.path.join(SERVERS_DIR, 'OLD')
os.makedirs(OLD_DIR, exist_ok=True)

# --- Tracking for Active Server Tasks ---
# Stores {'server_name': {'task': asyncio.Task, 'params': StdioServerParameters}}
ACTIVE_SERVER_TASKS: Dict[str, Dict[str, Any]] = {}
SERVER_START_TIMES: Dict[str, datetime.datetime] = {} # New dictionary for start times
_server_load_errors: Dict[str, str] = {} # Cache for server config load errors

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
        _server_load_errors.pop(name, None) # Clear potential old error if file is gone
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        # Success: clear any cached error for this server
        _server_load_errors.pop(name, None)
        return config_data
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {e}"
        logger.error(f"❌ _fs_load_server: {error_msg}")
        _server_load_errors[name] = error_msg # Cache the error
        _write_server_error_log(name, error_msg)
        return None
    except IOError as e:
        error_msg = f"IO error: {e}"
        logger.error(f"❌ _fs_load_server: {error_msg}")
        _server_load_errors[name] = error_msg # Cache the error
        _write_server_error_log(name, error_msg)
        return None
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(f"❌ _fs_load_server: {error_msg}", exc_info=True)
        _server_load_errors[name] = error_msg # Cache the error
        _write_server_error_log(name, error_msg)
        return None


def _write_server_error_log(name: str, error_message: str) -> None:
    '''
    Write an error message to a server-specific log file in the SERVERS_DIR.
    Overwrites any existing log to only keep the latest error.
    Creates a log file named {name}.log with timestamp.
    '''
    try:
        # Use the original name directly as it's used for the .json file
        log_path = os.path.join(SERVERS_DIR, f"{name}.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Open in write mode to overwrite previous content
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"{timestamp} [ERROR] {error_message}\n")

        logger.debug(f"Wrote error log for server '{name}' at {log_path}")
    except Exception as e:
        # Don't let logging errors disrupt the main flow
        logger.error(f"Failed to write server error log for '{name}': {e}")


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
        # This might be more robust to move than delete, similar to functions
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            backup_filename = f"{name}_{timestamp}.json.removed"
            backup_path = os.path.join(OLD_DIR, backup_filename)
            shutil.move(file_path, backup_path)
            logger.info(f"🗑️ Moved server config '{name}' to '{backup_path}'")
            # Clear any cached load error after successful removal
            _server_load_errors.pop(name, None)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to move server config file '{file_path}': {e}")
        os.remove(file_path)
        logger.info(f"🗑️ Removed server config '{name}' at {file_path}")
        # Clear any cached load error after successful removal
        _server_load_errors.pop(name, None)
        return True
    except Exception as e:
        logger.error(f"❌ server_remove: Failed to delete {file_path}: {e}")
        return False


def server_get(name: str) -> Optional[Dict[str, Any]]:
    """
    Returns the server config dict or None if not found.
    """
    return _fs_load_server(name)


def server_list() -> List[TextContent]:
    """
    Lists all server config names available in SERVERS_DIR and their status.
    Returns a list of TextContent objects.
    """
    results = []
    try:
        for filename in os.listdir(SERVERS_DIR):
            if filename.endswith('.json'):
                name = filename[:-5] # Remove .json
                status = "Running" if name in ACTIVE_SERVER_TASKS else "Stopped"
                results.append(TextContent(type='text', text=f"{name} (Status: {status})"))
    except Exception as e:
        logger.error(f"❌ server_list: Failed to list servers in {SERVERS_DIR}: {e}")
        # Optionally return an error message as TextContent
        # return [TextContent(text=f"Error listing servers: {e}")]
    return results

async def server_set(args: Dict[str, Any], server) -> List[TextContent]:
    """
    MCP handler to add/update a server config.
    Expects args['config']: dict or JSON string containing {"mcpServers": {"server_name": {...}}}
    The server name is derived from the key within 'mcpServers'.
    If the server is running and the config is updated, it might need restarting.
    """
    logger.debug(f"Received server_set request with args: {args}")
    config_input = args.get('config')
    config_container = None # Will hold the final dictionary

    # Handle if input is a JSON string
    if isinstance(config_input, str):
        try:
            config_container = json.loads(config_input)
            logger.debug("Parsed 'config' from JSON string to dictionary.")
        except json.JSONDecodeError as e:
            msg = f"Invalid parameter: 'config' was a string but failed JSON parsing: {e}"
            logger.error(f"❌ server_set: {msg}")
            return [TextContent(type='text', text=msg)]
    elif isinstance(config_input, dict):
        config_container = config_input # It's already a dictionary
    else:
         msg = "Missing or invalid parameter type: 'config' must be a dictionary or JSON string."
         logger.error(f"❌ server_set: {msg}")
         return [TextContent(type='text', text=msg)]

    # Now validate the container
    if not config_container:
         msg = "Invalid parameter: 'config' could not be processed."
         logger.error(f"❌ server_set: {msg}")
         return [TextContent(type='text', text=msg)]

    mcp_servers = config_container.get("mcpServers")
    if not mcp_servers or not isinstance(mcp_servers, dict) or len(mcp_servers) != 1:
        msg = "Invalid 'config': Expected a single key under 'mcpServers'."
        logger.error(f"❌ server_set: {msg}")
        return [TextContent(type='text', text=msg)]

    name = list(mcp_servers.keys())[0]
    server_specific_config = mcp_servers[name] # The inner config

    if not name or not isinstance(name, str):
        msg = "Invalid server name derived from 'config'."
        logger.error(f"❌ server_set: {msg}")
        return [TextContent(type='text', text=msg)]

    # Validate the extracted server-specific config minimally (e.g., has command)
    if not isinstance(server_specific_config, dict) or 'command' not in server_specific_config:
        msg = f"Invalid config structure for server '{name}' under 'mcpServers'. Must be a dictionary with at least a 'command' key."
        logger.error(f"❌ server_set: {msg}")
        return [TextContent(type='text', text=msg)]

    logger.info(f"Processing set request for server: '{name}'")
    logger.debug(f"Config details for '{name}': {json.dumps(server_specific_config, indent=2)}") # Log inner config for detail

    # --- Backup Logic Start ---
    file_path = os.path.join(SERVERS_DIR, f"{name}.json") # Construct the target file path
    if os.path.exists(file_path):
            logger.info(f"💾 Found existing file for '{name}', attempting backup...")
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                # Using .json.bak for clarity, matching the original file type
                backup_filename = f"{name}_{timestamp}.json.bak"
                backup_path = os.path.join(OLD_DIR, backup_filename)
                shutil.copy2(file_path, backup_path) # copy2 preserves metadata
                logger.info(f"🛡️ Successfully backed up '{name}' to '{backup_path}'")
            except Exception as e:
                logger.error(f"❌ Failed to backup existing file '{file_path}' to OLD folder: {e}")
                # Log error but continue
    else:
        logger.info(f"ⓘ No existing file found for '{name}', creating new file.")
    # --- Backup Logic End ---

    # Save the *entire original config container* structure
    saved_path = _fs_save_server(name, config_container)

    if saved_path:
        validation = server_validate(name) # Validate the *saved* config
        valid_msg = f"Server config '{name}' saved successfully."
        if not validation.get('valid', False):
            error = validation.get('error', 'Unknown validation error')
            valid_msg += f" WARNING: Validation failed: {error}"
        else:
            valid_msg += " Config structure appears valid."

        # Add running status warning if needed
        if name in ACTIVE_SERVER_TASKS:
            logger.warning(f"🔔 Server '{name}' is currently running. Configuration updated. Restart may be required for changes to take effect.")
            valid_msg += " Server is running; manual restart might be needed to apply changes."

        # Clear any cached load error now that it's updated
        _server_load_errors.pop(name, None)

        return [TextContent(type='text', text=valid_msg)]
    else:
        return [TextContent(type='text', text=f"Failed to save server config '{name}'.")]


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
    logger.info(f"MCP server '{name}' started")
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
        raise ValueError(msg)

    # Load the config from file
    full_config = _fs_load_server(name)
    if not full_config:
        msg = f"Config file not found for server '{name}'."
        logger.error(f"❌ server_start: {msg}")
        raise FileNotFoundError(msg)

    # Extract the specific server's config from the 'mcpServers' structure
    try:
        server_config = full_config['mcpServers'][name]
    except KeyError as e:
        msg = f"Could not find server '{name}' within the 'mcpServers' key in the config file."
        logger.error(f"❌ server_start: {msg} - KeyError: {e}")
        raise KeyError(msg)
    except TypeError:
        msg = f"Invalid config structure for server '{name}'. Expected dict with 'mcpServers'."
        logger.error(f"❌ server_start: {msg} - Loaded config: {full_config!r}")
        raise TypeError(msg)

    if name in ACTIVE_SERVER_TASKS:
        msg = f"Server '{name}' is already running."
        logger.warning(f"⚠️ server_start: {msg}")
        raise ValueError(msg)

    validation = server_validate(name)
    if not validation.get('valid', False):
        error = validation.get('error', 'Unknown error')
        msg = f"Invalid config for '{name}': {error}"
        logger.error(f"❌ server_start: {msg}")
        raise ValueError(msg)

    try:
        # Prepare parameters for stdio_client
        params = StdioServerParameters(
            command=server_config['command'], # Use extracted server_config
            args=server_config.get('args', []),
            env=server_config.get('env', None),
            cwd=server_config.get('cwd', None)
        )
    except Exception as e: # Catch potential Pydantic validation errors or KeyErrors
        msg = f"Failed to prepare start parameters for '{name}': {e}"
        logger.error(f"❌ server_start: {msg}", exc_info=True)
        raise ValueError(msg)

    # Start the background task
    logger.info(f"Attempting to start background task for server '{name}'...")
    task = asyncio.create_task(_run_mcp_client_session(name, params))

    # Store task info immediately
    ACTIVE_SERVER_TASKS[name] = {'task': task, 'params': params}
    SERVER_START_TIMES[name] = datetime.datetime.now() # Record start time

    # Return success - PID is not available synchronously here
    return [TextContent(type='text', text=f"MCP service '{name}' started")]


async def server_stop(args: Dict[str, Any], server) -> List[TextContent]:
    """
    MCP handler to stop a managed MCP server process task.
    Expects args['name']: str.
    """
    name = args.get('name')
    if not name or not isinstance(name, str):
        msg = "Missing or invalid parameter: 'name' must be str."
        logger.error(f"❌ server_stop: {msg}")
        raise ValueError(msg)

    if name not in ACTIVE_SERVER_TASKS:
        msg = f"Server '{name}' is not running or not managed."
        logger.warning(f"⚠️ server_stop: {msg}")
        raise ValueError(msg)

    task_info = ACTIVE_SERVER_TASKS.get(name)
    if not task_info or 'task' not in task_info:
        msg = f"Error stopping server '{name}': Inconsistent state."
        logger.error(f"❌ server_stop: {msg}")
        # Clean up if entry exists but is broken
        if name in ACTIVE_SERVER_TASKS:
            del ACTIVE_SERVER_TASKS[name]
        raise RuntimeError(msg)

    task = task_info['task']
    if task.done():
        logger.info(f"🧹 Task for server '{name}' was already finished. Cleaning up entry.")
        # Cleanup potentially missed by the finally block
        if name in ACTIVE_SERVER_TASKS:
            del ACTIVE_SERVER_TASKS[name]
        SERVER_START_TIMES.pop(name, None) # Remove start time
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
        SERVER_START_TIMES.pop(name, None) # Remove start time

        return [TextContent(type='text', text=f"MCP server '{name}' stopped")]
