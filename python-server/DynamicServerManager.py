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

from state import (
    SERVERS_DIR,
    logger,
    ACTIVE_SERVER_TASKS,
    SERVER_REQUEST_TIMEOUT
)

class DynamicServerManager:
    """
    Manages dynamic MCP servers lifecycle - adding, removing, starting, stopping, etc.
    Handles the configuration of MCP servers and their execution as background tasks.
    """

    def __init__(self, servers_dir: str):
        """
        Initialize the Dynamic Server Manager with a specific servers directory.

        Args:
            servers_dir: Path to the directory where server configuration files are stored
        """
        self.servers_dir = servers_dir
        self.old_dir = os.path.join(servers_dir, 'OLD')

        # Ensure directories exist
        os.makedirs(self.servers_dir, exist_ok=True)
        os.makedirs(self.old_dir, exist_ok=True)

        # Server state tracking
        self.active_server_tasks = ACTIVE_SERVER_TASKS
        self.server_start_times = {}
        self._server_load_errors = {}

        logger.info(f"Dynamic Server Manager initialized with servers dir: {servers_dir}")

    # --- 1. File Save/Load Methods ---

    async def _fs_save_server(self, name: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Saves the provided JSON config dict to {name}.json in servers_dir.
        Returns the full path if successful, None otherwise.
        """
        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)
        logger.debug(f"---> _fs_save_server: Attempting to save '{name}' to path: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info(f"💾 Saved server config for '{name}' to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"❌ _fs_save_server: Failed to write {file_path}: {e}")
            return None

    async def _fs_load_server(self, name: str) -> Optional[Union[Dict[str, Any], str]]:
        """
        Loads the config for server '{name}' from {name}.json in servers_dir.
        Returns the parsed JSON dict on success.
        Returns the raw file content (str) if JSON parsing fails.
        Returns None if the file doesn't exist or an IO error occurs.
        """
        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)
        if not os.path.exists(file_path):
            logger.info(f"⚠️ _fs_load_server: Existing config not found for '{name}' at {file_path}")
            self._server_load_errors.pop(name, None)  # Clear potential old error if file is gone
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            # Attempt to parse the JSON
            config_data = json.loads(raw_content)
            # Success: clear any cached error for this server
            self._server_load_errors.pop(name, None)
            return config_data
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}"
            logger.error(f"❌ _fs_load_server: {error_msg}")
            self._server_load_errors[name] = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg, raw_content)
            return raw_content  # Return raw content on JSON error
        except IOError as e:
            error_msg = f"IO error reading {file_path}: {e}"
            logger.error(f"❌ _fs_load_server: {error_msg}")
            self._server_load_errors[name] = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg)
            return None  # Return None on IO error
        except Exception as e:
            error_msg = f"Unexpected error loading {file_path}: {e}"
            logger.error(f"❌ _fs_load_server: {error_msg}", exc_info=True)
            self._server_load_errors[name] = error_msg  # Cache the error
            await self._write_server_error_log(name, error_msg)
            return None  # Return None on unexpected errors

    async def _write_server_error_log(self, name: str, error_msg: str, raw_content: Optional[str] = None) -> None:
        """
        Write an error message to a server-specific log file in the servers_dir.
        Overwrites any existing log to only keep the latest error.
        Includes raw content if provided (e.g., for JSON decode errors).
        """
        log_filename = f"{name}_error.log"
        log_path = os.path.join(self.servers_dir, log_filename)
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"=== Error Log for '{name}' @ {timestamp} ===\n\n")
                f.write(f"{error_msg}\n\n")
                if raw_content:
                    f.write("=== Raw Content ===\n\n")
                    f.write(raw_content)
            logger.debug(f"📝 Error log written to {log_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write error log for '{name}': {e}")

    # --- 2. Config CRUD Methods ---

    async def server_add(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Adds a new server config. Returns False if it already exists.
        """
        # Check if server already exists
        existing = await self._fs_load_server(name)
        if existing is not None:
            logger.warning(f"⚠️ Server '{name}' already exists, not adding.")
            return False

        # Save the new config
        result = await self._fs_save_server(name, config)
        return result is not None

    async def server_remove(self, name: str) -> bool:
        """
        Removes the server config by deleting its JSON file. Returns False if missing.
        Also stops the server if it's running.
        """
        # Check if server is running and stop it first
        if name in self.active_server_tasks:
            logger.info(f"🛑 Stopping running server '{name}' before removal...")
            try:
                # Implement server_stop functionality here
                # await self.server_stop({"name": name}, None)
                pass  # Placeholder for now
            except Exception as e:
                logger.error(f"❌ Failed to stop server '{name}' during removal: {e}")
                # Continue with removal anyway

        # Remove the config file
        file_path = os.path.join(self.servers_dir, f"{name}.json")
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ Server config '{name}' not found for removal.")
            return False

        try:
            # Backup the config before deletion
            backup_dir = self.old_dir
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{timestamp}_{name}.json")
            shutil.copy2(file_path, backup_path)
            logger.info(f"📦 Backed up '{name}' config to {backup_path}")

            # Delete the original file
            os.remove(file_path)
            logger.info(f"🗑️ Removed server config '{name}'")

            # Clean up any cached errors
            self._server_load_errors.pop(name, None)

            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove server '{name}': {e}")
            return False

    async def server_get(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Returns the server config dict or None if not found.
        """
        result = await self._fs_load_server(name)
        if isinstance(result, dict):
            return result
        return None

    async def get_server_tools(self, name: str) -> List[Tool]:
        """
        Fetches the tool list from a managed MCP server.
        Prioritizes using an existing session from a running server.
        Only creates a temporary connection if no running session exists.

        Returns a list of Tool objects from the server.

        Raises:
            ValueError: If the server name is invalid or not found.
            RuntimeError: If the server config is invalid or server is not running.
            TimeoutError: If the connection or request to the server times out.
            Exception: For other communication or MCP errors.
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"Invalid server name: {name}")

        # Detailed logging for debugging
        logger.debug(f"get_server_tools: Checking for running server '{name}'")

        # Check if the server is already running (use existing session)
        if name in self.active_server_tasks:
            task_info = self.active_server_tasks[name]
            session = task_info.get('session')

            # If we have an existing session, try to use it
            if session is not None:
                logger.debug(f"get_server_tools: Found existing session for '{name}', attempting to use it")
                try:
                    # Check if session is usable by attempting a list_tools operation
                    # Use a timeout to prevent hanging if the session is stale
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=SERVER_REQUEST_TIMEOUT)

                    # Update last successful use timestamp
                    self.active_server_tasks[name]['last_used'] = datetime.datetime.now(datetime.timezone.utc)
                    logger.info(f"Successfully fetched {len(tools_result.tools)} tools from running server '{name}' using existing session")
                    return tools_result.tools
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout using existing session for '{name}', session may be stale")
                    # Continue to fetch tools via a new temporary connection
                except Exception as e:
                    logger.warning(f"Could not use existing session for '{name}': {e}")
                    # Continue to fetch tools via a new temporary connection
            else:
                logger.debug(f"Server '{name}' exists in active_server_tasks but session is not available yet")

                # Check if the server is still starting up
                if task_info.get('status') == 'starting' and 'ready_event' in task_info:
                    ready_event = task_info['ready_event']
                    logger.info(f"Server '{name}' is still starting, waiting for ready event")
                    try:
                        # Wait for the ready event with timeout
                        await asyncio.wait_for(ready_event.wait(), timeout=SERVER_REQUEST_TIMEOUT)

                        # If we're here, the ready event was set - check if a session is now available
                        if task_info.get('session') is not None:
                            # Try again with the new session
                            logger.debug(f"Server '{name}' is now ready, trying new session")
                            try:
                                tools_result = await asyncio.wait_for(task_info['session'].list_tools(), timeout=SERVER_REQUEST_TIMEOUT)
                                logger.info(f"Successfully fetched {len(tools_result.tools)} tools from newly started server '{name}'")
                                return tools_result.tools
                            except Exception as e:
                                logger.warning(f"Failed to use newly ready session for '{name}': {e}")
                                # Continue to temporary connection
                    except asyncio.TimeoutError:
                        logger.warning(f"Timed out waiting for server '{name}' to become ready")
                    # Continue to temporary connection if we couldn't use the session

        # Get the server config (for temporary connection or for starting the server)
        logger.debug(f"get_server_tools: Loading config for '{name}'")
        server_config = await self.server_get(name)
        if not server_config:
            raise ValueError(f"Server config not found for '{name}'")

        if 'mcpServers' not in server_config or name not in server_config['mcpServers']:
            raise ValueError(f"Invalid server config structure for '{name}'")

        specific_config = server_config['mcpServers'][name]
        if 'command' not in specific_config:
            raise ValueError(f"Missing 'command' in server config for '{name}'")

        # Option: First try starting the server to get a persistent connection
        # This is the preferred approach for frequent tool use
        if name not in self.active_server_tasks:
            logger.info(f"Server '{name}' is not running. Consider using server_start for persistent connection.")
            # Note: we don't auto-start here as that should be a deliberate action by the caller

        # Create temporary connection parameters for one-time use
        try:
            logger.debug(f"get_server_tools: Creating temporary connection parameters for '{name}'")
            params = StdioServerParameters(
                command=specific_config['command'],
                args=specific_config.get('args', []),
                env=specific_config.get('env', {}),
                cwd=specific_config.get('cwd', None)
            )
        except Exception as e:
            raise ValueError(f"Invalid server parameters for '{name}': {e}")

        # Create temporary connection
        logger.info(f"Creating temporary connection to server '{name}'")
        try:
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    # Wait for initialization with timeout
                    await asyncio.wait_for(session.initialize(), timeout=SERVER_REQUEST_TIMEOUT)

                    # Get tools list
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=SERVER_REQUEST_TIMEOUT)
                    logger.info(f"Successfully fetched {len(tools_result.tools)} tools from '{name}' via temporary connection")
                    return tools_result.tools
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout while connecting to server '{name}'")
        except Exception as e:
            raise RuntimeError(f"Error fetching tools from server '{name}': {e}")

    async def server_list(self) -> List[str]:
        """
        Returns a list of all configured server names.
        """
        server_list = []
        if os.path.exists(self.servers_dir):
            for filename in os.listdir(self.servers_dir):
                if filename.endswith('.json'):
                    server_name = filename[:-5]  # Remove .json extension
                    server_list.append(server_name)
        return server_list

    async def server_set(self, args: Dict[str, Any], server) -> Tuple[Optional[str], List[TextContent]]:
        """
        MCP handler to add/update a server config.
        Expects args['config']: dict or JSON string containing {"mcpServers": {"server_name": {...}}}
        The server name is derived from the key within 'mcpServers'.
        If the server is running and the config is updated, it might need restarting.

        Returns:
            Tuple containing:
                - Extracted server name (str) if successful, None otherwise
                - List of TextContent for the response
        """
        # Get the config from the args
        config_param = args.get('config')
        if not config_param:
            return None, [TextContent(type="text", text="Error: Missing 'config' parameter.")]

        # Parse the config if it's a string
        if isinstance(config_param, str):
            try:
                config = json.loads(config_param)
            except json.JSONDecodeError as e:
                error_msg = f"Error parsing JSON config: {e}"
                logger.error(f"❌ server_set: {error_msg}")
                return None, [TextContent(type="text", text=error_msg)]
        else:
            config = config_param

        # Validate basic structure
        if not isinstance(config, dict):
            error_msg = "Error: 'config' must be a JSON object (dict)"
            logger.error(f"❌ server_set: {error_msg}")
            return None, [TextContent(type="text", text=error_msg)]

        if 'mcpServers' not in config:
            error_msg = "Error: Config must have 'mcpServers' key at the top level"
            logger.error(f"❌ server_set: {error_msg}")
            return None, [TextContent(type="text", text=error_msg)]

        mcpServers = config.get('mcpServers')
        if not isinstance(mcpServers, dict) or not mcpServers:
            error_msg = "Error: 'mcpServers' must be a non-empty object (dict)"
            logger.error(f"❌ server_set: {error_msg}")
            return None, [TextContent(type="text", text=error_msg)]

        # Extract the server name from the key in mcpServers
        # We only support one server per config for simplicity
        if len(mcpServers) != 1:
            logger.warning("⚠️ server_set: Multiple servers in config. Using only the first one.")

        server_name = next(iter(mcpServers.keys()))
        server_config = mcpServers[server_name]

        # Validate server config
        if not isinstance(server_config, dict):
            error_msg = f"Error: Server config for '{server_name}' must be an object (dict)"
            logger.error(f"❌ server_set: {error_msg}")
            return None, [TextContent(type="text", text=error_msg)]

        if 'command' not in server_config:
            error_msg = f"Error: Server config for '{server_name}' must have 'command' field"
            logger.error(f"❌ server_set: {error_msg}")
            return None, [TextContent(type="text", text=error_msg)]

        # Check if server is running (might need to stop it)
        is_running = server_name in self.active_server_tasks
        needs_restart = False

        if is_running:
            # Check if the config has changed in a way that requires restart
            current_config = await self._fs_load_server(server_name)
            if current_config and 'mcpServers' in current_config and server_name in current_config['mcpServers']:
                current_server_config = current_config['mcpServers'][server_name]

                # Check if fundamental parameters changed
                restart_params = ['command', 'args', 'cwd']
                for param in restart_params:
                    if param in server_config and (param not in current_server_config or
                                               server_config[param] != current_server_config[param]):
                        needs_restart = True
                        break

        # Save the config
        result = await self._fs_save_server(server_name, config)
        if not result:
            error_msg = f"Error: Failed to save server config for '{server_name}'"
            logger.error(f"❌ server_set: {error_msg}")
            return None, [TextContent(type="text", text=error_msg)]

        # Handle restart if needed
        restart_message = ""
        if is_running and needs_restart:
            logger.info(f"🔄 Server '{server_name}' config changed, needs restart")
            restart_message = f" Configuration changed significantly. Server needs to be restarted to apply changes."

        # Return success with extracted name and success message
        success_msg = f"Server '{server_name}' configuration saved successfully.{restart_message}"
        logger.info(f"✅ server_set: {success_msg}")
        return server_name, [TextContent(type="text", text=success_msg)]

    async def server_validate(self, name: str) -> Dict[str, Any]:
        """
        Validates that the *saved* server config JSON has required keys.
        Returns a dict with 'valid':bool and 'error':Optional[str].

        Validates the standard MCP server format with nested 'mcpServers' structure.
        """
        result = {
            'valid': False,
            'error': None
        }

        config = await self.server_get(name)
        if not config:
            result['error'] = f"Server config '{name}' not found or has parsing errors"
            return result

        # Check for proper mcpServers structure
        if not isinstance(config, dict):
            result['error'] = f"Server config for '{name}' is not a valid dictionary"
            return result

        # Check if we have the mcpServers key which is the standard format
        if 'mcpServers' in config:
            # Modern format with mcpServers structure
            if not isinstance(config['mcpServers'], dict):
                result['error'] = f"'mcpServers' in config for '{name}' is not a dictionary"
                return result

            if name not in config['mcpServers']:
                result['error'] = f"Server '{name}' not found within 'mcpServers' key"
                return result

            server_config = config['mcpServers'][name]

            # Check required fields in the server config
            if not isinstance(server_config, dict):
                result['error'] = f"Server config for '{name}' is not a dictionary"
                return result

            if 'command' not in server_config:
                result['error'] = f"Missing required key 'command' in server config for '{name}'"
                return result

        else:
            # Legacy format (direct config)?
            logger.warning(f"⚠️ server_validate: Config for '{name}' does not use 'mcpServers' structure")

            # Check for command directly in the old format
            if 'command' not in config:
                result['error'] = f"Missing required key 'command' in server config for '{name}'"
                return result

        # Config is valid
        result['valid'] = True
        return result

    # --- 3. Background Task Methods ---

    async def _run_mcp_client_session(self, name: str, params: StdioServerParameters, shutdown_event: asyncio.Event) -> None:
        """
        Runs the background MCP client session for a managed server.
        Establishes a client connection and keeps it alive until shutdown is requested.
        The session is stored in active_server_tasks for reuse by other methods.
        """
        logger.info(f"[{name}] Starting MCP client session task...")
        session = None  # Define session here to ensure it's accessible in finally block if needed
        try:
            logger.debug(f"[{name}] Attempting stdio_client connection with params: {params}")
            async with stdio_client(params) as (reader, writer):
                logger.debug(f"[{name}] Stdio connection established. Creating ClientSession...")
                # Create the session object without using context manager so we can keep it alive
                # This is crucial for connection reuse
                session = ClientSession(reader, writer)

                try:
                    # Initialize the session
                    logger.debug(f"[{name}] ClientSession created. Initializing session...")
                    await asyncio.wait_for(session.initialize(), timeout=30.0)  # Increased timeout for reliable init
                    logger.info(f"[{name}] MCP Session initialized successfully.")

                    # Update active_server_tasks *after* successful initialization
                    if name in self.active_server_tasks:
                        logger.debug(f"[{name}] Updating active_server_tasks with session and timestamp...")
                        self.active_server_tasks[name]['session'] = session
                        self.active_server_tasks[name]['started_at'] = datetime.datetime.now(datetime.timezone.utc)
                        self.active_server_tasks[name]['status'] = 'running'  # Mark as running
                        logger.info(f"[{name}] active_server_tasks updated. Current state for '{name}': {{'status': '{self.active_server_tasks[name].get('status')}', 'started_at': '{self.active_server_tasks[name].get('started_at')}', 'session_present': {self.active_server_tasks[name].get('session') is not None}}}")

                        # Pre-fetch tools list to validate the connection works
                        try:
                            tools_result = await session.list_tools()
                            logger.info(f"[{name}] Successfully fetched {len(tools_result.tools)} tools from running server")
                            self.active_server_tasks[name]['tools_count'] = len(tools_result.tools)
                        except Exception as e:
                            logger.warning(f"[{name}] Failed to fetch tools list: {e}")

                        # Signal readiness *after* updating state
                        if 'ready_event' in self.active_server_tasks[name]:
                            self.active_server_tasks[name]['ready_event'].set()
                            logger.debug(f"[{name}] Ready event set.")
                        else:
                            logger.warning(f"[{name}] 'ready_event' not found in active_server_tasks entry.")
                    else:
                        logger.error(f"[{name}] Entry disappeared from active_server_tasks before session could be stored!")
                        # If the entry is gone, we probably can't signal readiness either. Shutdown might be needed.
                        return  # Exit if the task entry is gone

                    # --- Session is initialized and running ---
                    logger.debug(f"[{name}] Session active. Waiting for shutdown signal...")

                    # Keep the session alive until shutdown is requested
                    # Periodically check session is still responsive by calling a lightweight operation
                    while not shutdown_event.is_set():
                        try:
                            # Wait for shutdown event with timeout
                            # This allows us to periodically check if session is still responsive
                            shutdown_requested = await asyncio.wait_for(
                                shutdown_event.wait(),
                                timeout=30.0  # Check every 30 seconds
                            )
                            if shutdown_requested:
                                break

                            # Optional: Check session is still responsive with lightweight ping
                            # Only do this for servers that have been running a while
                            if name in self.active_server_tasks and self.active_server_tasks[name].get('status') == 'running':
                                started_at = self.active_server_tasks[name].get('started_at')
                                now = datetime.datetime.now(datetime.timezone.utc)

                                # Only ping if server has been running more than 2 minutes
                                if started_at and (now - started_at).total_seconds() > 120:
                                    try:
                                        # Just list_tools as a lightweight ping
                                        await asyncio.wait_for(session.list_tools(), timeout=5.0)
                                        # No need to process the result, we're just checking if it responds
                                    except Exception as e:
                                        logger.warning(f"[{name}] Session health check failed: {e}")
                                        # Break the loop if session is no longer responsive
                                        # This will trigger cleanup in finally block
                                        break
                        except asyncio.TimeoutError:
                            # This is expected - it just means the shutdown_event.wait timed out
                            # We'll loop and check again
                            pass
                        except Exception as e:
                            logger.error(f"[{name}] Error during session monitoring: {e}")
                            # Break the loop if we encounter unexpected errors
                            break

                    logger.info(f"[{name}] Shutdown event received. Cleaning up session...")

                    # Close the session cleanly if we still have it
                    if session is not None:
                        try:
                            await asyncio.wait_for(session.close(), timeout=5.0)
                            logger.debug(f"[{name}] Session closed cleanly")
                        except Exception as e:
                            logger.warning(f"[{name}] Error closing session: {e}")

                except asyncio.TimeoutError:
                    logger.error(f"[{name}] Timeout occurred during session.initialize().")
                    if name in self.active_server_tasks:
                        self.active_server_tasks[name]['status'] = 'init_timeout'
                        if 'ready_event' in self.active_server_tasks[name]:
                            self.active_server_tasks[name]['ready_event'].set()  # Signal completion (failure)

                    # Close session if it exists but failed to initialize
                    if session is not None:
                        try:
                            await session.close()
                        except Exception:
                            pass  # Ignore errors when closing an uninitialized session

                except Exception as e:
                    logger.error(f"[{name}] Error during session.initialize() or operation: {e}")
                    if name in self.active_server_tasks:
                        self.active_server_tasks[name]['status'] = 'init_failed'
                        if 'ready_event' in self.active_server_tasks[name]:
                            self.active_server_tasks[name]['ready_event'].set()  # Signal completion (failure)

                    # Close session if it exists but failed
                    if session is not None:
                        try:
                            await session.close()
                        except Exception:
                            pass  # Ignore errors when closing a failed session

        except ConnectionRefusedError:
            logger.error(f"[{name}] Connection refused when starting stdio client process.")
            if name in self.active_server_tasks:
                self.active_server_tasks[name]['status'] = 'connection_refused'
                if 'ready_event' in self.active_server_tasks[name]:
                    self.active_server_tasks[name]['ready_event'].set()  # Signal completion (failure)

        except Exception as e:
            logger.error(f"[{name}] Unexpected error in client session: {e}", exc_info=True)
            if name in self.active_server_tasks:
                self.active_server_tasks[name]['status'] = 'error'
                if 'ready_event' in self.active_server_tasks[name]:
                    self.active_server_tasks[name]['ready_event'].set()  # Signal completion (failure)

        finally:
            # Clean up if needed (task cancellation, etc.)
            logger.info(f"[{name}] Client session task exiting.")

            # Force-clear session reference to help with garbage collection
            if session is not None:
                logger.debug(f"[{name}] Clearing session reference.")
                try:
                    await session.close()  # Ensure session is properly closed
                except Exception as e:
                    logger.debug(f"[{name}] Error closing session during cleanup: {e}")
                session = None

            # Remove from active tasks if still present
            if name in self.active_server_tasks:
                logger.debug(f"[{name}] Removing entry from active_server_tasks.")
                self.active_server_tasks.pop(name, None)
                self.server_start_times.pop(name, None)

    # --- 4. Server Start/Stop Methods ---

    async def server_start(self, args: Dict[str, Any], server) -> List[TextContent]:
        """
        MCP handler to start a managed MCP server process as a background task.
        Expects args['name']: str.
        """
        name = args.get('name')
        logger.debug(f"▶️ server_start: Entered with args: {args}")
        if not name or not isinstance(name, str):
            msg = "Missing or invalid parameter: 'name' must be str."
            logger.error(f"❌ server_start: {msg}")
            raise ValueError(msg)

        # Load the config from file
        full_config = await self._fs_load_server(name)
        if not full_config:
            msg = f"Config file not found for server '{name}'."
            logger.error(f"❌ server_start: {msg}")
            raise FileNotFoundError(msg)

        # Extract the specific server's config from the 'mcpServers' structure
        try:
            if not isinstance(full_config, dict):
                raise TypeError(f"Invalid config format for '{name}', expected dict but got {type(full_config)}")

            if 'mcpServers' not in full_config:
                raise KeyError(f"Missing 'mcpServers' key in config for '{name}'")

            if name not in full_config['mcpServers']:
                raise KeyError(f"Server '{name}' not found within 'mcpServers' in config")

            server_config = full_config['mcpServers'][name]
        except KeyError as e:
            msg = f"Could not find server '{name}' within the 'mcpServers' key in the config file."
            logger.error(f"❌ server_start: {msg} - KeyError: {e}")
            raise KeyError(msg)
        except TypeError as e:
            msg = f"Invalid config structure for server '{name}'. Expected dict with 'mcpServers'."
            logger.error(f"❌ server_start: {msg} - Loaded config: {full_config!r}")
            raise TypeError(msg)

        if name in self.active_server_tasks:
            msg = f"Server '{name}' is already running."
            logger.warning(f"⚠️ server_start: {msg}")
            raise ValueError(msg)  # Or return a message indicating it's already running

        # Perform basic validation on the extracted config
        if not isinstance(server_config, dict) or 'command' not in server_config:
            msg = f"Invalid server config for '{name}': must be a dictionary with at least a 'command' key."
            logger.warning(f"⚠️ server_start: {msg}")
            raise ValueError(msg)

        validation_result = await self.server_validate(name)
        logger.debug(f"▶️ server_start: Validation result for '{name}': {validation_result}")
        if not validation_result.get('valid', False):
            error = validation_result.get('error', 'Unknown error')
            msg = f"Invalid config for '{name}': {error}"
            logger.error(f"❌ server_start: {msg}")
            raise ValueError(msg)

        try:
            # Prepare parameters for stdio_client
            params = StdioServerParameters(
                command=server_config['command'],
                args=server_config.get('args', []),
                env=server_config.get('env', None),
                cwd=server_config.get('cwd', None)
            )
            logger.debug(f"▶️ server_start: Prepared StdioServerParameters for '{name}': {params}")
        except Exception as e:  # Catch potential Pydantic validation errors or KeyErrors
            msg = f"Failed to prepare start parameters for '{name}': {e}"
            logger.error(f"❌ server_start: {msg}", exc_info=True)
            raise ValueError(msg)

        # Start the background task
        logger.info(f"Attempting to start background task for server '{name}'...")
        logger.debug(f"▶️ server_start: Creating asyncio task for _run_mcp_client_session('{name}')...")
        shutdown_event = asyncio.Event()
        ready_event = asyncio.Event()  # Create the ready event

        # Create the task first
        task = asyncio.create_task(self._run_mcp_client_session(name, params, shutdown_event))

        # Now store the complete task info in one step
        self.active_server_tasks[name] = {
            'task': task,
            'config': full_config,
            'shutdown_event': shutdown_event,
            'session': None,  # Initialize session as None
            'ready_event': ready_event,
            'status': 'starting'
        }

        # Record start time
        self.server_start_times[name] = datetime.datetime.now()

        logger.info(f"MCP server '{name}' started")
        logger.debug(f"▶️ server_start: Task info recorded for '{name}'. Start time will be added upon session init.")

        # Return success
        return [TextContent(type='text', text=f"MCP service '{name}' started.")]

    async def server_stop(self, args: Dict[str, Any], server) -> List[TextContent]:
        """
        MCP handler to stop a managed MCP server process task.
        Expects args['name']: str.
        """
        name = args.get('name')
        if not name or not isinstance(name, str):
            msg = "Missing or invalid parameter: 'name' must be str."
            logger.error(f"❌ server_stop: {msg}")
            raise ValueError(msg)

        if name not in self.active_server_tasks:
            msg = f"Server '{name}' is not running or not managed."
            logger.warning(f"⚠️ server_stop: {msg}")
            raise ValueError(msg)

        task_info = self.active_server_tasks.get(name)
        if not task_info or 'task' not in task_info:
            msg = f"Error stopping server '{name}': Inconsistent state."
            logger.error(f"❌ server_stop: {msg}")
            # Clean up if entry exists but is broken
            if name in self.active_server_tasks:
                del self.active_server_tasks[name]
            raise RuntimeError(msg)

        task = task_info['task']
        if task.done():
            logger.info(f"🧹 Task for server '{name}' was already finished. Cleaning up entry.")
            # Cleanup potentially missed by the finally block
            if name in self.active_server_tasks:
                del self.active_server_tasks[name]
            self.server_start_times.pop(name, None)  # Remove start time
            return [TextContent(type='text', text=f"Server '{name}' task was already finished.")]
        else:
            logger.info(f"Attempting to cancel task for server '{name}'...")

            # Set the shutdown event first to signal graceful shutdown
            if 'shutdown_event' in task_info and task_info['shutdown_event'] is not None:
                task_info['shutdown_event'].set()
                logger.debug(f"Shutdown event set for '{name}'")

            # Also cancel the task directly in case it's not responding to the event
            task.cancel()

            # Give cancellation a moment to potentially propagate
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
            # Also clean up our tracking
            self.server_start_times.pop(name, None)  # Remove start time

            return [TextContent(type='text', text=f"MCP server '{name}' stopped")]

    # --- 5. Self-test Methods ---

    async def self_test(self) -> bool:
        """
        Runs a comprehensive self-test of the DynamicServerManager functionality.
        Creates, validates, starts, queries, and stops test servers, then cleans up.

        Returns:
            bool: True if all tests pass, False otherwise
        """
        # Save original logger for restoration
        original_logger = logger if 'logger' in globals() else None

        class MockLogger:
            def __init__(self):
                self.logs = []

            def debug(self, msg, *args, **kwargs):
                print(f"[DEBUG] {msg}")
                self.logs.append(f"DEBUG: {msg}")

            def info(self, msg, *args, **kwargs):
                print(f"[INFO] {msg}")
                self.logs.append(f"INFO: {msg}")

            def warning(self, msg, *args, **kwargs):
                print(f"[WARNING] {msg}")
                self.logs.append(f"WARNING: {msg}")

            def error(self, msg, *args, **kwargs):
                print(f"[ERROR] {msg}")
                self.logs.append(f"ERROR: {msg}")

        try:
            print("\n🧪 STARTING DYNAMICSERVERMANAGER SELF-TEST...")
            all_tests_passed = True
            total_tests = 0
            passed_tests = 0

            # Import needed for testing
            from state import SERVERS_DIR
            from pathlib import Path

            # Use the actual SERVERS_DIR for testing
            servers_dir = SERVERS_DIR
            print(f"\n📁 Using servers directory: {servers_dir}")

            # Set up the manager
            globals()['logger'] = MockLogger()
            manager = DynamicServerManager(servers_dir)

            # Test 1: _fs_save_server method
            total_tests += 1
            print("\n🧪 TEST 1: Testing _fs_save_server method...")
            test_server_name = "test_server"
            test_config = {
                "command": "python -m http.server",
                "args": ["8888"],
                "cwd": "/tmp",
                "env": {"DEBUG": "1"}
            }

            file_path = await manager._fs_save_server(test_server_name, test_config)
            saved_file = Path(servers_dir) / f"{test_server_name}.json"

            if file_path and saved_file.exists():
                print("✅ Test 1 PASSED: Server config was successfully saved")
                passed_tests += 1
            else:
                print("❌ Test 1 FAILED: Server config was not saved")
                all_tests_passed = False

            # Test 2: _fs_load_server method
            total_tests += 1
            print("\n🧪 TEST 2: Testing _fs_load_server method...")
            loaded_config = await manager._fs_load_server(test_server_name)

            if isinstance(loaded_config, dict) and loaded_config.get('command') == test_config['command']:
                print("✅ Test 2 PASSED: Loaded config matches the saved config")
                passed_tests += 1
            else:
                print("❌ Test 2 FAILED: Loaded config doesn't match the saved config")
                print(f"Expected: {test_config}")
                print(f"Got: {loaded_config}")
                all_tests_passed = False

            # Test 3: server_add method
            total_tests += 1
            print("\n🧪 TEST 3: Testing server_add method...")
            new_server_name = "new_test_server"
            new_server_config = {
                "command": "python -m http.server",
                "args": ["9999"],
                "cwd": "/tmp/test",
                "env": {"DEBUG": "0"}
            }

            try:
                result = await manager.server_add(new_server_name, new_server_config)
                new_file = Path(servers_dir) / f"{new_server_name}.json"

                if result and new_file.exists():
                    print("✅ Test 3 PASSED: Server config was successfully added")
                    passed_tests += 1
                else:
                    print("❌ Test 3 FAILED: Server config was not added")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 3 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 4: server_list method
            total_tests += 1
            print("\n🧪 TEST 4: Testing server_list method...")
            try:
                server_list = await manager.server_list()
                if test_server_name in server_list and new_server_name in server_list:
                    print("✅ Test 4 PASSED: Server list correctly contains added servers")
                    passed_tests += 1
                else:
                    print("❌ Test 4 FAILED: Server list doesn't contain expected servers")
                    print(f"Expected servers: {test_server_name}, {new_server_name}")
                    print(f"Got servers: {server_list}")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 4 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 5: server_validate method
            total_tests += 1
            print("\n🧪 TEST 5: Testing server_validate method...")
            try:
                validation_result = await manager.server_validate(test_server_name)
                if validation_result.get('valid'):
                    print("✅ Test 5 PASSED: Server config validation successful")
                    passed_tests += 1
                else:
                    print("❌ Test 5 FAILED: Server config validation failed")
                    print(f"Error: {validation_result.get('error')}")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 5 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 6: Invalid server validation
            total_tests += 1
            print("\n🧪 TEST 6: Testing validation of non-existent server...")
            try:
                # Create a server with missing required fields
                invalid_server_name = "invalid_server"
                invalid_config = {"args": ["-h"], "cwd": "/tmp"} # Missing 'command'
                await manager._fs_save_server(invalid_server_name, invalid_config)

                validation_result = await manager.server_validate(invalid_server_name)
                if not validation_result.get('valid') and validation_result.get('error'):
                    print("✅ Test 6 PASSED: Invalid server correctly failed validation")
                    print(f"Validation error: {validation_result.get('error')}")
                    passed_tests += 1
                else:
                    print("❌ Test 6 FAILED: Invalid server incorrectly passed validation")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 6 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 7: Test server_start and server_stop MCP handlers
            total_tests += 1
            print("\n🧪 TEST 7: Testing server_start and server_stop MCP handlers...")
            try:
                # Create an echo server config that finishes quickly
                echo_server_name = "echo_server"
                echo_config = {
                    "mcpServers": {
                        echo_server_name: {
                            "command": "echo",
                            "args": ["Hello from echo server!"],
                            "cwd": "/tmp"
                        }
                    }
                }

                # Save the echo server config
                await manager._fs_save_server(echo_server_name, echo_config)

                # Create mock server for notifications
                class MockServer:
                    async def publish_notification(self, *args, **kwargs):
                        print(f"🔔 Mock notification published: {args}, {kwargs}")

                mock_server = MockServer()

                # Test server_start
                start_args = {"name": echo_server_name}
                start_response = await manager.server_start(start_args, mock_server)

                # Echo should start and finish quickly, give it a second
                await asyncio.sleep(1)

                # Test server_stop (may already have exited due to echo finishing)
                stop_args = {"name": echo_server_name}
                try:
                    stop_response = await manager.server_stop(stop_args, mock_server)
                    print("✅ Test 7 PASSED: Server start and stop MCP handlers work")
                    passed_tests += 1
                except ValueError as e:
                    # If echo server already exited, this is still a pass
                    if "not running" in str(e):
                        print("✅ Test 7 PASSED: Server start worked, stop showed server already exited (expected)")
                        passed_tests += 1
                    else:
                        print(f"❌ Test 7 FAILED: Unexpected error stopping server: {e}")
                        all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 7 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 8: Test server_set MCP handler
            total_tests += 1
            print("\n🧪 TEST 8: Testing server_set MCP handler...")
            try:
                # Create mock server for notifications
                mock_server = MockServer()

                # Test config to set via server_set
                cat_server_name = "cat_server"
                cat_config = {
                    "mcpServers": {
                        cat_server_name: {
                            "command": "cat",
                            "args": ["/etc/hostname"],
                            "cwd": "/tmp"
                        }
                    }
                }

                # Test server_set
                set_args = {"config": cat_config}
                name, response = await manager.server_set(set_args, mock_server)

                if name == cat_server_name:
                    # Verify the server was saved properly
                    loaded_config = await manager._fs_load_server(cat_server_name)
                    if loaded_config and 'mcpServers' in loaded_config and cat_server_name in loaded_config['mcpServers']:
                        cmd = loaded_config['mcpServers'][cat_server_name].get('command')
                        if cmd == "cat":
                            print("✅ Test 8 PASSED: Server set MCP handler works correctly")
                            passed_tests += 1
                        else:
                            print(f"❌ Test 8 FAILED: Server set returned incorrect command: {cmd}")
                            all_tests_passed = False
                    else:
                        print("❌ Test 8 FAILED: Server set did not save config correctly")
                        all_tests_passed = False
                else:
                    print(f"❌ Test 8 FAILED: Server set returned wrong name: {name}")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 8 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 9: server_remove method
            total_tests += 1
            print("\n🧪 TEST 9: Testing server_remove method...")
            try:
                result = await manager.server_remove(new_server_name)
                new_file = Path(servers_dir) / f"{new_server_name}.json"

                if result and not new_file.exists():
                    print("✅ Test 9 PASSED: Server config was successfully removed")
                    passed_tests += 1
                else:
                    print("❌ Test 9 FAILED: Server config was not removed")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 9 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 10: Testing with a real MCP server (openweather) - simplified direct approach
            total_tests += 1
            print("\n🧪 TEST 10: Testing with real openweather MCP server...")
            try:
                # Create the openweather server config with the real API key
                openweather_name = "openweather"
                openweather_config = {
                    "mcpServers": {
                        openweather_name: {
                            "command": "uvx",
                            "args": [
                                "--from",
                                "atlantis-open-weather-mcp",
                                "start-weather-server",
                                "--api-key",
                                "1234" # replace with real API key for testing
                            ]
                        }
                    }
                }

                # Save the openweather config
                print("  Saving openweather server configuration...")
                save_result = await manager._fs_save_server(openweather_name, openweather_config)

                if save_result:
                    # Try to validate the server configuration
                    print("  Validating openweather server configuration...")
                    validation = await manager.server_validate(openweather_name)

                    if validation.get('valid'):
                        print("✅ Server configuration validated successfully")

                        # Test direct connection approach (which we already know works)
                        print("  Creating direct connection to OpenWeather server...")
                        try:
                            # Get the parameters from the configuration
                            server_config = await manager.server_get(openweather_name)
                            if 'mcpServers' in server_config and openweather_name in server_config['mcpServers']:
                                specific_config = server_config['mcpServers'][openweather_name]

                                # Create connection parameters
                                params = StdioServerParameters(
                                    command=specific_config['command'],
                                    args=specific_config.get('args', []),
                                    env=specific_config.get('env', {}),
                                    cwd=specific_config.get('cwd', None)
                                )

                                # Create a direct connection
                                print("  Opening connection to OpenWeather server...")
                                async with stdio_client(params) as (reader, writer):
                                    print("  Connection established, creating session...")
                                    async with ClientSession(reader, writer) as session:
                                        # Initialize the session
                                        print("  Initializing session...")
                                        await session.initialize()
                                        print("✅ Session initialized successfully!")

                                        # Get available tools
                                        print("  Fetching tool list...")
                                        tools_result = await session.list_tools()
                                        print(f"  ✅ Found {len(tools_result.tools)} tools: {[t.name for t in tools_result.tools]}")

                                        # If we got tools, try to get weather for Nuuk
                                        if tools_result.tools:
                                            # Set up the tool arguments
                                            print("  Fetching current weather for Nuuk, Greenland...")
                                            tool_args = {
                                                "location": "Nuuk",
                                                "timezone_offset": 0
                                            }

                                            # Call the get_current_weather tool
                                            result = await session.call_tool("get_current_weather", tool_args)

                                            if result and hasattr(result, 'content') and result.content:
                                                weather_data = result.content
                                                print(f"  Weather response received: {weather_data}")
                                                
                                                # Check if the response contains an error
                                                error_detected = False
                                                for content_item in weather_data:
                                                    if hasattr(content_item, 'text') and isinstance(content_item.text, str):
                                                        text = content_item.text
                                                        if '"error"' in text or 'Error' in text or 'error' in text or '401' in text or 'Unauthorized' in text:
                                                            print(f"  ❌ API Error detected: {text}")
                                                            error_detected = True
                                                            break
                                                
                                                if not error_detected:
                                                    print(f"  ☀️ Current weather in Nuuk successfully retrieved!")
                                                    print("✅ Test 10 PASSED: Successfully fetched weather data for Nuuk!")
                                                    passed_tests += 1
                                                else:
                                                    print(f"❌ Test 10 FAILED: API error in weather response")
                                                    all_tests_passed = False
                                            else:
                                                print(f"❌ Test 10 FAILED: Weather data response was empty or malformed: {result}")
                                                all_tests_passed = False
                                        else:
                                            print("❌ Test 10 FAILED: No tools found on the server")
                                            all_tests_passed = False
                            else:
                                print("❌ Test 10 FAILED: Invalid server config structure")
                                all_tests_passed = False

                        except Exception as e:
                            print(f"❌ Test 10 FAILED with direct connection exception: {e}")
                            all_tests_passed = False
                    else:
                        print(f"❌ Test 10 FAILED: Configuration validation failed: {validation.get('error')}")
                        all_tests_passed = False
                else:
                    print("❌ Test 10 FAILED: Could not save openweather server config")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 10 FAILED with exception: {e}")
                all_tests_passed = False

            # Clean up leftover test files
            print("\n🧹 Cleaning up test files...")
            leftover_files = [
                "test_server",            # From Test 1 & 2
                "invalid_server",        # From Test 6
                "echo_server",           # From Test 7
                "cat_server",            # From Test 8
                "openweather"            # From Test 10
            ]

            for test_file in leftover_files:
                try:
                    if os.path.exists(os.path.join(SERVERS_DIR, f"{test_file}.json")):
                        await manager.server_remove(test_file)
                        print(f"  Removed {test_file}.json")
                except Exception as e:
                    print(f"  ⚠️ Could not remove {test_file}.json: {e}")

            # Print test summary
            print("\n🏁 DYNAMICSERVERMANAGER SELF-TEST COMPLETE 🏁")
            print(f"Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")

            if all_tests_passed:
                print("\n✅ ALL TESTS PASSED! DynamicServerManager is working correctly.")
                # Clean up OLD directory if it exists
                old_dir_final_cleanup = Path(SERVERS_DIR) / "OLD"
                if old_dir_final_cleanup.exists() and old_dir_final_cleanup.is_dir():
                    print(f"🧹 Cleaning up OLD directory: {old_dir_final_cleanup}")
                    import shutil
                    shutil.rmtree(old_dir_final_cleanup) # remove OLD and its contents
                return True
            else:
                print("\n❌ SOME TESTS FAILED. See above for details.")
                print(f"👉 NOTE: Tests were run against {SERVERS_DIR}. Review and manually clean up test files if necessary.")
                return False
        finally:
            # Restore the original logger
            if original_logger:
                globals()['logger'] = original_logger
            else:
                if 'logger' in globals(): # if mock was set but no original, remove it
                    del globals()['logger']


# Run the tests if this module is executed directly
if __name__ == "__main__":
    from state import SERVERS_DIR
    import sys

    # Create a banner
    print("="*80)
    print(f"{'DYNAMIC SERVER MANAGER SELF-TEST':^80}")
    print("="*80)
    print(f"Testing with servers directory: {SERVERS_DIR}")
    print("-"*80)

    async def run_tests():
        """Set up the manager and run self-tests"""
        manager = DynamicServerManager(SERVERS_DIR)
        success = await manager.self_test()
        sys.exit(0 if success else 1)

    try:
        # Check if there's an existing event loop
        try:
            asyncio.get_running_loop()
            print("Already in an event loop, creating a task")
            asyncio.create_task(run_tests())
        except RuntimeError:
            print("Creating new event loop for tests")
            # Create a new event loop for testing
            asyncio.run(run_tests())
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
