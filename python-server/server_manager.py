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
from typing import Any, Dict, Optional, List, Tuple, Union

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import TextContent, Tool, ListToolsResult, ListToolsRequest

from state import SERVERS_DIR, logger

# Directory to store error logs for server configs
SERVER_ERRORS_DIR = "." # Store error logs in the current directory

# Ensure SERVERS_DIR exists
os.makedirs(SERVERS_DIR, exist_ok=True)
# Define and ensure OLD_DIR exists for backups
OLD_DIR = os.path.join(SERVERS_DIR, 'OLD')
os.makedirs(OLD_DIR, exist_ok=True)

# --- Tracking for Active Server Tasks ---
# Stores {'server_name': {'task': asyncio.Task, 'params': StdioServerParameters, 'shutdown_event': asyncio.Event, 'session': Optional[ClientSession], 'ready_event': asyncio.Event}}
ACTIVE_SERVER_TASKS: Dict[str, Dict] = {}
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
    logger.debug(f"---> _fs_save_server: Attempting to save '{name}' to path: {file_path}") # Added log
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"💾 Saved server config for '{name}' to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ _fs_save_server: Failed to write {file_path}: {e}")
        return None


def _fs_load_server(name: str) -> Optional[Union[Dict[str, Any], str]]:
    """
    Loads the config for server '{name}' from {name}.json in SERVERS_DIR.
    Returns the parsed JSON dict on success.
    Returns the raw file content (str) if JSON parsing fails.
    Returns None if the file doesn't exist or an IO error occurs.
    """
    safe_name = f"{name}.json"
    file_path = os.path.join(SERVERS_DIR, safe_name)
    if not os.path.exists(file_path):
        logger.info(f"⚠️ _fs_load_server: Existing config not found for '{name}' at {file_path}")
        _server_load_errors.pop(name, None) # Clear potential old error if file is gone
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        # Attempt to parse the JSON
        config_data = json.loads(raw_content)
        # Success: clear any cached error for this server
        _server_load_errors.pop(name, None)
        return config_data
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {e}"
        logger.error(f"❌ _fs_load_server: {error_msg}")
        _server_load_errors[name] = error_msg # Cache the error
        _write_server_error_log(name, error_msg, raw_content) # Pass raw_content to error log func
        return raw_content # Return raw content on JSON error
    except IOError as e:
        error_msg = f"IO error reading {file_path}: {e}"
        logger.error(f"❌ _fs_load_server: {error_msg}")
        _server_load_errors[name] = error_msg # Cache the error
        _write_server_error_log(name, error_msg) # Don't have raw_content here necessarily
        return None # Return None on IO error
    except Exception as e:
        error_msg = f"Unexpected error loading {file_path}: {e}"
        logger.error(f"❌ _fs_load_server: {error_msg}", exc_info=True)
        _server_load_errors[name] = error_msg # Cache the error
        _write_server_error_log(name, error_msg)
        return None # Return None on unexpected errors


def _write_server_error_log(name: str, error_msg: str, raw_content: Optional[str] = None) -> None:
    '''
    Write an error message to a server-specific log file in the SERVER_ERRORS_DIR.
    Overwrites any existing log to only keep the latest error.
    Includes raw content if provided (e.g., for JSON decode errors).
    '''
    try:
        if not os.path.exists(SERVER_ERRORS_DIR):
            os.makedirs(SERVER_ERRORS_DIR)

        log_path = os.path.join(SERVER_ERRORS_DIR, f"{name}.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Open in write mode to overwrite previous content
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"{timestamp} [ERROR] {error_msg}\n")
            if raw_content:
                log_file.write("\n--- Raw Config Content ---\n")
                log_file.write(raw_content)
                log_file.write("\n--------------------------\n")

        logger.debug(f"📝 Wrote error log for server '{name}' at {log_path}")
    except Exception as e:
        # Avoid nested errors, just log if writing the error log fails
        logger.error(f"❌ Failed to write server error log for '{name}': {e}", exc_info=True)


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
            logger.error(f"❌ Failed to move server config file '{file_path}' to OLD folder: {e}")
        os.remove(file_path)
        logger.info(f"🗑️ Removed server config '{name}' at {file_path}")
        # Clear any cached load error after successful removal
        _server_load_errors.pop(name, None)
        return True
    except Exception as e:
        logger.error(f"❌ server_remove: Failed to delete {file_path}: {e}")
        return False


def server_get(name: str) -> Optional[Union[Dict[str, Any], str]]:
    """
    Returns the server config dict or None if not found.
    """
    return _fs_load_server(name)


import asyncio
import logging
from typing import List, Dict, Any, Optional
from state import SERVERS_DIR, logger # Assuming state.py provides logger
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import Tool, ListToolsResult # <-- Remove ListToolsRequest import
from server_manager import ACTIVE_SERVER_TASKS # Import necessary items from server_manager

# Define Timeout for Requests (adjust as needed)
SERVER_REQUEST_TIMEOUT = 10.0 # seconds




async def get_server_tools(name: str) -> list[dict]:
    """
    Connects to a specific running managed server using its stored start
    parameters and fetches its tool list via a temporary connection.

    Args:
        name: The name of the managed server.

    Returns:
        A list of dictionaries representing Tool objects
        from the target server.

    Raises:
        ValueError: If the server is not found, not running, or config is invalid.
        TimeoutError: If the connection or request to the server times out.
        Exception: For other communication or MCP errors.
    """
    logger.info(f"Attempting to get tools for server '{name}'...")
    task_info = ACTIVE_SERVER_TASKS.get(name)

    if not task_info:
        msg = f"Server '{name}' not found in active tasks (may not be running)."
        logger.warning(f"⚠️ get_server_tools: {msg}")
        raise ValueError(msg)

    # --- Wait for session using asyncio.Event --- #
    session: Optional[ClientSession] = None
    ready_event = task_info.get('ready_event')
    session_ready_timeout = 2.0 # How long to wait for the session to become ready

    if not ready_event:
        msg = f"Server '{name}' task info is missing the 'ready_event'. Cannot wait for session."
        logger.error(f"❌ get_server_tools: {msg}")
        raise ValueError(msg)

    try:
        logger.debug(f"Waiting up to {session_ready_timeout}s for session '{name}' to become ready...")
        await asyncio.wait_for(ready_event.wait(), timeout=session_ready_timeout)
        logger.debug(f"Session ready event received for '{name}'.")
        # Event received, session should be available now
        session = task_info.get('session')
        if not session:
             # This case *shouldn't* happen if the event logic is correct, but check anyway
            msg = f"Server '{name}' ready_event was set, but session object is still missing."
            logger.error(f"❌ get_server_tools: {msg}")
            raise ValueError(msg)

    except asyncio.TimeoutError:
        msg = f"Timeout ({session_ready_timeout}s) waiting for session for server '{name}' to become ready."
        logger.error(f"❌ get_server_tools: {msg}")
        raise ValueError(msg) from None
    except Exception as e:
        msg = f"Error while waiting for session event for server '{name}': {e}"
        logger.error(f"❌ get_server_tools: {msg}", exc_info=True)
        raise ValueError(msg) from e
    # --- End wait logic --- #

    # If we get here, session must be valid and ready
    logger.info(f"Found session object for '{name}'. Requesting tools/list...")

    try:
        # Use the existing session to list tools with a timeout
        response = await asyncio.wait_for(
            session.list_tools(),
            timeout=SERVER_REQUEST_TIMEOUT
        )

        # Process the response
        if isinstance(response, ListToolsResult) and isinstance(response.tools, list):
            tools_as_dicts: list[dict] = [] # Changed type hint and variable name
            for i, item in enumerate(response.tools):
                tool_dict = {}
                if isinstance(item, Tool):
                    # Convert Tool Pydantic model to dict
                    try:
                        tool_dict = item.model_dump(mode='json') # Use model_dump for Pydantic v2+
                    except AttributeError:
                        tool_dict = item.dict() # Fallback for Pydantic v1
                    except Exception as dump_err:
                        logger.warning(f"⚠️ Could not dump tool item #{i} ({item.name}) from '{name}': {dump_err}")
                        continue # Skip this tool
                elif isinstance(item, dict):
                    tool_dict = item # Assume it's already a dict
                else:
                    logger.warning(f"⚠️ Unexpected item type #{i} in tools/list response from '{name}': {type(item)}")
                    continue # Skip this tool

                tools_as_dicts.append(tool_dict)

            logger.info(f"✅ Successfully retrieved and processed {len(tools_as_dicts)} tools from '{name}' via existing session.")
            return tools_as_dicts # Return list of dicts
        else:
            error_msg = f"Unexpected response format from '{name}' for tools/list via existing session: {response}"
            logger.error(f"❌ get_server_tools: {error_msg}")
            raise Exception(ErrorData(code=-32002, message=error_msg)) # Use MCP specific error

    except asyncio.TimeoutError:
        logger.error(f"❌ Timeout requesting tools from running server '{name}' via existing session ({SERVER_REQUEST_TIMEOUT}s)." )
        # Wrap timeout in McpError
        raise Exception(ErrorData(code=-32001, message=f"Timeout getting tools from '{name}' ({SERVER_REQUEST_TIMEOUT}s).")) from None
    except Exception as e: # Catch other unexpected errors
        logger.error(f"❌ Unexpected error getting tools from server '{name}' via existing session: {e}", exc_info=True)
        raise # Re-raise other unexpected errors
    finally:
        pass

def server_list() -> List[TextContent]:
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

    logger.debug(f"---> server_set: Preparing to save config for '{name}': {json.dumps(config_container, indent=2)}") # Added log
    # Save the *entire original config container* structure
    saved_path = _fs_save_server(name, config_container)
    logger.debug(f"<--- server_set: _fs_save_server returned path: {saved_path}") # Added log

    if saved_path:
        # Clear any cached load error now that it's updated
        _server_load_errors.pop(name, None)

        # Attempt to remove the error log file now that config is saved successfully
        error_log_path = os.path.join(SERVER_ERRORS_DIR, f"{name}.log")
        if os.path.exists(error_log_path):
            try:
                os.remove(error_log_path)
                logger.debug(f"🧹 Removed stale error log file: {error_log_path}")
            except OSError as e:
                logger.warning(f"⚠️ Failed to remove stale error log file {error_log_path}: {e}")

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
async def _run_mcp_client_session(name: str, params: StdioServerParameters, shutdown_event: asyncio.Event) -> None:
    """Runs the MCP client session in the background using stdio_client."""
    logger.info(f"Starting MCP server '{name}'")
    try:
        # Use stdio_client as an async context manager
        async with stdio_client(params) as (reader, writer):
            logger.debug(f"▶️ _run_mcp_client_session: stdio_client context entered for '{name}'. Creating ClientSession.")

            # Use ClientSession as an async context manager AND initialize
            async with ClientSession(reader, writer) as session:
                logger.debug(f"▶️ _run_mcp_client_session: ClientSession context entered for '{name}'. Initializing session...")
                try:
                    await asyncio.wait_for(session.initialize(), timeout=10.0) # Add timeout to initialization
                    logger.info(f"✅ Successfully initialized session with MCP server '{name}'.")

                    # Store the *active and initialized* session object
                    if name in ACTIVE_SERVER_TASKS:
                        ACTIVE_SERVER_TASKS[name]['session'] = session
                        ready_event = ACTIVE_SERVER_TASKS[name]['ready_event']
                        ready_event.set() # Signal that the session is ready!
                        logger.debug(f"▶️ _run_mcp_client_session: Initialized session stored for '{name}'.")
                    else:
                         logger.warning(f"❓ Task entry for '{name}' disappeared before session could be stored.")
                         return # Exit if task entry is gone

                    # Keep the task alive by waiting for the shutdown event
                    logger.debug(f"▶️ _run_mcp_client_session: Entering wait loop for shutdown event for '{name}'.")
                    await shutdown_event.wait() # Wait until signaled to shut down
                    logger.info(f"🛑 Shutdown event received for '{name}'. Exiting session context.")

                except asyncio.TimeoutError:
                     logger.error(f"❌ Timeout initializing session with '{name}' after 10s.")
                     # Ensure session is marked as None or invalid if init fails
                     if name in ACTIVE_SERVER_TASKS:
                         ACTIVE_SERVER_TASKS[name]['session'] = None
                except Exception as init_err:
                     logger.error(f"❌ Error initializing session with '{name}': {init_err}")
                     if name in ACTIVE_SERVER_TASKS:
                         ACTIVE_SERVER_TASKS[name]['session'] = None

            logger.debug(f"▶️ _run_mcp_client_session: Exited ClientSession context for '{name}'.")

    except asyncio.CancelledError:
        logger.info(f"🛑 MCP client session task for '{name}' was cancelled.")
    finally:
        logger.info(f"MCP server task for '{name}' finished.")
        if name in ACTIVE_SERVER_TASKS:
            ACTIVE_SERVER_TASKS[name]['session'] = None # Ensure session is cleared on any exit
            ACTIVE_SERVER_TASKS[name]['task'].cancel() # Attempt to cancel if somehow still running
            # Optionally remove the entry entirely or mark as stopped
            # del ACTIVE_SERVER_TASKS[name]
            logger.debug(f"▶️ Cleaned up session/task entry for '{name}' in finally block.")


# --- 4. Server Start/Stop Operations ---
async def server_start(args: Dict[str, Any], server) -> List[TextContent]:
    """
    MCP handler to start a managed MCP server process as a background task.
    Expects args['name']: str.
    """
    name = args.get('name')
    logger.debug(f"▶️ server_start: Entered with args: {args}") # DEBUG ADDED
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
        raise ValueError(msg) # Or return a message indicating it's already running

    # Load the server configuration first
    server_config_full = _fs_load_server(name)
    if not server_config_full:
        load_error = _server_load_errors.get(name, f"Configuration for '{name}' not found or failed to load.")
        logger.error(f"❌ server_start: Failed to load config for '{name}': {load_error}")
        raise ValueError(f"Failed to load config for server '{name}': {load_error}")

    logger.debug(f"▶️ server_start: Loaded full config for '{name}': {server_config_full}") # DEBUG ADDED

    # Extract the specific server's config from within 'mcpServers'
    # Assuming the structure is { "mcpServers": { "server_name": { ... } } }
    if 'mcpServers' not in server_config_full or name not in server_config_full['mcpServers']:
        msg = f"Invalid config structure for '{name}'. Missing 'mcpServers' key or entry for '{name}'."
        logger.error(f"❌ server_start: {msg}")
        raise ValueError(msg)

    server_config = server_config_full['mcpServers'][name]
    logger.debug(f"▶️ server_start: Extracted specific config for '{name}': {server_config}") # DEBUG ADDED

    # Perform basic validation on the extracted config
    if not isinstance(server_config, dict) or 'command' not in server_config:
        msg = f"Invalid server config for '{name}': must be a dictionary with at least a 'command' key."
        logger.warning(f"⚠️ server_start: {msg}")
        raise ValueError(msg)

    validation = server_validate(name)
    logger.debug(f"▶️ server_start: Validation result for '{name}': {validation}") # DEBUG ADDED
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
        logger.debug(f"▶️ server_start: Prepared StdioServerParameters for '{name}': {params}") # DEBUG ADDED
    except Exception as e: # Catch potential Pydantic validation errors or KeyErrors
        msg = f"Failed to prepare start parameters for '{name}': {e}"
        logger.error(f"❌ server_start: {msg}", exc_info=True)
        raise ValueError(msg)

    # Start the background task
    logger.info(f"Attempting to start background task for server '{name}'...")
    logger.debug(f"▶️ server_start: Creating asyncio task for _run_mcp_client_session('{name}')...") # DEBUG ADDED
    shutdown_event = asyncio.Event()
    ready_event = asyncio.Event() # Create the ready event

    # Create the task first
    task = asyncio.create_task(_run_mcp_client_session(name, params, shutdown_event))

    # Now store the complete task info in one step
    ACTIVE_SERVER_TASKS[name] = {
        'task': task, 
        'config': server_config_full, 
        'shutdown_event': shutdown_event, 
        'session': None, # Initialize session as None
        'ready_event': ready_event 
    }

    logger.info(f"MCP server '{name}' started")

    SERVER_START_TIMES[name] = datetime.datetime.now() # Record start time
    logger.debug(f"▶️ server_start: Task and start time recorded for '{name}'.") # DEBUG ADDED


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
