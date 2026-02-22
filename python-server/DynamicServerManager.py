"""
Manages the lifecycle of dynamic MCP servers (configs) by launching them
as background tasks using the MCP Python SDK's stdio transport.
Stores each server config as a JSON file under SERVERS_DIR.
"""
import os
import asyncio
import json
import logging
import pathlib
import shutil
import datetime
import traceback
from typing import Any, Dict, Optional, List, Tuple, Union

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import TextContent, Tool, ListToolsResult, ListToolsRequest

from state import (
    SERVERS_DIR,
    logger,
    SERVER_REQUEST_TIMEOUT
)
from utils import format_json_log

# --- Tracking for Server Tasks (Moved from server_manager) ---
SERVER_TASKS: dict[str, dict] = {}


class DynamicServerManager:

    def __init__(self, servers_dir: str):
        self.servers_dir = servers_dir
        self.old_dir = os.path.join(servers_dir, 'OLD')

        # Ensure directories exist
        os.makedirs(self.servers_dir, exist_ok=True)
        os.makedirs(self.old_dir, exist_ok=True)




        self.server_tasks = SERVER_TASKS
        self._server_load_errors = {}

        logger.info(f"Dynamic Server Manager initialized with servers dir: {servers_dir}")

    # --- 1. File Save/Load Methods ---

    def extract_server_name(self, server_json_content: str) -> Optional[str]:
        # Try to parse the JSON content
        config = json.loads(server_json_content)

        # Check if it's in the modern format with mcpServers
        if isinstance(config, dict) and 'mcpServers' in config:
            # Get the first key under mcpServers
            if isinstance(config['mcpServers'], dict) and len(config['mcpServers']) > 0:
                return next(iter(config['mcpServers'].keys()))

        # Check if it's in the legacy format (top-level key)
        elif isinstance(config, dict) and len(config) > 0:
            # Return the first top-level key
            return next(iter(config.keys()))

        return None

    async def _fs_save_server(self, name: str, content: str) -> Optional[str]:

        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)
        logger.debug(f"---> _fs_save_server: Attempting to save '{name}' to path: {file_path}")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"💾 Saved server config for '{name}' to {file_path}")
        return file_path

    async def _fs_load_server(self, name: str) -> str:
        """Load server configuration from file.

        Args:
            name: Name of the server configuration to load

        Returns:
            The raw content of the configuration file

        Raises:
            FileNotFoundError: When the config file doesn't exist
        """
        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)

        if not os.path.exists(file_path):
            error_msg = f"MCP server '{name}' not found (is it installed or spelled correctly?)"
            logger.error(f"❌ _fs_load_server: {error_msg}")
            self._server_load_errors.pop(name, None)  # Clear potential old error if file is gone
            raise FileNotFoundError(error_msg)

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        self._server_load_errors.pop(name, None) # Clear any previous load error for this name if read succeeds
        return raw_content

    async def _write_server_error_log(self, name: str, error_msg: str, raw_content: Optional[str] = None) -> None:
        log_filename = f"{name}_error.log"
        log_path = os.path.join(self.servers_dir, log_filename)

        with open(log_path, 'w', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"=== Error Log for '{name}' @ {timestamp} ===\n\n")
            f.write(f"{error_msg}\n\n")
            if raw_content:
                f.write("=== Raw Content ===\n\n")
                f.write(raw_content)
        logger.debug(f"📝 MCP error log written to {log_path}")

    # --- 2. Config CRUD Methods ---

    async def server_add(self, name: str) -> bool:
        # Check if server already exists directly by checking the file
        # Don't use _fs_load_server which now raises exceptions for missing files
        safe_name = f"{name}.json"
        file_path = os.path.join(self.servers_dir, safe_name)

        if os.path.exists(file_path):
            raise ValueError(f"⚠️ MCP server '{name}' already exists, not adding")

        # Create a template config based on openweather example
        template_obj = {
            "mcpServers": {
                name: {
                    "command": "uvx",  # Default command, can be edited later
                    "args": [
                        "--from",
                        "atlantis-open-weather-mcp",
                        f"start-weather-server",  # Create standard start command based on name
                        "--api-key",
                        "<your api key here>"  # Placeholder for API key
                    ]
                }
            }
        }

        # Save the new template config
        logger.info(f"🆕 Creating new server config template for '{name}'")
        template_config = json.dumps(template_obj, indent=4) # indent for pretty printing, optional
        result = await self._fs_save_server(name, template_config)
        return result is not None

    async def server_remove(self, name: str) -> bool:
        # Check if server is running and stop it first
        if name in self.server_tasks:
            # suppress any errors during stop
            logger.info(f"🛑 Stopping running server '{name}' before removal...")
            try:
                # Call server_stop with proper arguments
                await self.server_stop({"name": name}, None)
                logger.info(f"🔔 Successfully stopped server '{name}' before removal")
            except Exception as e:
                logger.error(f"❌ Failed to stop server '{name}' during removal: {e}")
                # Continue with removal anyway

        # Remove the config file
        file_path = os.path.join(self.servers_dir, f"{name}.json")
        if not os.path.exists(file_path):
            raise ValueError(f"⚠️ MCP server config '{name}' not found for removal.")

        #suppress any  backup errors
        try:
            # Backup the config before deletion
            backup_dir = self.old_dir
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{timestamp}_{name}.json")
            shutil.copy2(file_path, backup_path)
            logger.info(f"📦 Backed up '{name}' config to {backup_path}")
        except Exception as e:
            logger.error(f"❌ Failed to remove server '{name}': {e}")
            return False

        # Delete the original file
        os.remove(file_path)
        logger.info(f"🗑️ Removed server config '{name}'")

        # Clean up any cached errors
        self._server_load_errors.pop(name, None)

        return True

    async def server_get(self, name: str) -> Optional[str]:
        # No validation, just return whatever _fs_load_server gives us
        return await self._fs_load_server(name)

    async def server_get_valid(self, name: str) -> Dict[str, Any]:
        # Load server configuration
        config_txt = await self._fs_load_server(name)

        # Check if the config exists before trying to validate it
        if config_txt is None:
            error_msg = f"Server configuration for '{name}' not found. Make sure the server exists."
            logger.error(f"❌ server_get_valid: {error_msg}")
            raise ValueError(error_msg)

        # Now validate the existing config
        return await self.validate(config_txt)


    # Note: We're using the existing server_start method instead of a custom helper

    async def get_server_tools(self, name: str) -> List[Tool]:
        if not name or not isinstance(name, str):
            raise ValueError(f"Invalid MCP server name: {name}")

        # Detailed logging for debugging
        logger.debug(f"get_server_tools: Checking for running server '{name}'")

        # Check if the server is already running (use existing session)
        if name in self.server_tasks:
            task_info = self.server_tasks[name]
            session = task_info.get('session')

            # If we have an existing session, try to use it
            if session is not None:
                logger.debug(f"get_server_tools: Found existing session for '{name}', attempting to use it")
                try:
                    # Check if session is usable by attempting a list_tools operation
                    # Use a timeout to prevent hanging if the session is stale
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=SERVER_REQUEST_TIMEOUT)

                    # Update last successful use timestamp
                    self.server_tasks[name]['last_used'] = datetime.datetime.now(datetime.timezone.utc)
                    logger.info(f"Successfully fetched {len(tools_result.tools)} tools from running server '{name}' using existing session")
                    return tools_result.tools
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout using existing session for '{name}', session may be stale")
                    # Continue to fetch tools via a new temporary connection
                except Exception as e:
                    logger.warning(f"Could not use existing session for '{name}': {e}")
                    # Continue to fetch tools via a new temporary connection
            else:
                logger.debug(f"Server '{name}' exists in server_tasks but session is not available yet")

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
                                # Pretty print tools as JSON
                                tools_json = format_json_log([tool.model_dump() for tool in tools_result.tools])
                                logger.info(f"Tools:\n{tools_json}")
                                return tools_result.tools
                            except Exception as e:
                                logger.warning(f"Failed to use newly ready session for '{name}': {e}")
                                # Continue to temporary connection
                    except asyncio.TimeoutError:
                        logger.warning(f"Timed out waiting for server '{name}' to become ready")
                    # Continue to temporary connection if we couldn't use the session

        # Get the server config (for temporary connection or for starting the server)
        logger.debug(f"get_server_tools: Loading config for '{name}'")
        config_txt = await self.server_get(name)
        if not config_txt:
            raise ValueError(f"No config found for server '{name}'")

        # convert config txt to json and store in full_config
        full_config: Dict[str, Any] = json.loads(config_txt)

        if 'mcpServers' not in full_config or name not in full_config['mcpServers']:
            raise ValueError(f"Invalid MCP server config structure for '{name}'")

        # also possible that mcpServers is missing entirely and go straight to name attr

        server_config = full_config['mcpServers'][name]
        if 'command' not in server_config:
            raise ValueError(f"Missing 'command' in MCP server config for '{name}'")

        # Auto-start the server if it's not already running
        if name not in self.server_tasks:
            logger.info(f"Server '{name}' is not running. Auto-starting it...")
            # Start the server and wait for it to be ready
            try:
                await self.server_start({'name': name}, None)  # Use the existing pattern with correct parameters
                logger.info(f"Successfully initiated auto-start for server '{name}'")
            except Exception as e:
                logger.error(f"Failed to auto-start server '{name}': {e}")
                raise RuntimeError(f"Failed to auto-start MCP server '{name}': {e}")

            # Give it a moment to initialize fully
            logger.info(f"Waiting for server '{name}' to initialize...")
            await asyncio.sleep(1)  # Short delay to ensure initialization completes

            # Now retry getting the tools with the new session
            if name in self.server_tasks and self.server_tasks[name].get('session') is not None:
                session = self.server_tasks[name]['session']
                try:
                    # Use the new session
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=SERVER_REQUEST_TIMEOUT)
                    self.server_tasks[name]['last_used'] = datetime.datetime.now(datetime.timezone.utc)
                    logger.info(f"Successfully fetched {len(tools_result.tools)} tools from newly started server '{name}'")
                    return tools_result.tools
                except Exception as e:
                    logger.error(f"Failed to get tools from newly started server '{name}': {e}")
                    raise RuntimeError(f"Error fetching tools from newly started MCP server '{name}': {e}")
            else:
                raise RuntimeError(f"MCP server '{name}' was started but session is not available")

        # If we got here, something unexpected happened
        raise RuntimeError(f"Unexpected error: Unable to get tools from MCP server '{name}'")

    async def server_list(self) -> List[str]:
        try:
            servers_path = pathlib.Path(self.servers_dir)
            server_files = list(servers_path.glob('*.json'))
            server_names = [file.stem for file in server_files]
            return sorted(server_names, key=str.lower)
        except Exception as e:
            logger.error(f"❌ Error listing servers: {e}")
            return []

    async def is_server_running(self, name: str) -> bool:
        if not name or not isinstance(name, str):
            return False

        # Check if the server exists in server_tasks with a valid task and session
        return (name in self.server_tasks and
                'task' in self.server_tasks[name] and
                not self.server_tasks[name]['task'].done() and
                'session' in self.server_tasks[name] and
                self.server_tasks[name]['session'] is not None)

    async def get_running_servers(self) -> List[str]:
        # Filter the server_tasks to only include properly running servers
        running_servers = []
        for name, task_info in self.server_tasks.items():
            # Check for basic task existence and not done
            if 'task' not in task_info or task_info['task'].done():
                logger.debug(f"Server '{name}' skipped: task missing or done")
                continue

            # Check for active session that's properly connected
            session = task_info.get('session')
            if session is None:
                logger.debug(f"Server '{name}' skipped: no session available")
                continue

            # For Python SDK ClientSession objects, we can't check 'connected' attribute
            # (as mentioned in your memory about mcp.ClientSession)
            # Instead, we'll assume if we have a session object, it's likely still usable
            # The actual tool fetching will validate this later and handle exceptions
            # This approach aligns with the recommendation to attempt a low-impact operation
            # instead of checking specific attributes
            logger.debug(f"Server '{name}' has a session - assuming it's valid")

            # This server passes all checks and is truly running
            logger.debug(f"Server '{name}' IS running: active task and session")
            running_servers.append(name)

        return running_servers

    async def server_set(self, name: str, content: str) -> str:
        # Save the content directly without any validation or processing
        result = await self._fs_save_server(name, content)

        # Simple success message - return plain string (MCP formatting happens in _format_mcp_response)
        return f"Server '{name}' configuration saved."

    async def server_validate(self, name: str) -> Dict[str, Any]:
        config_txt = await self.server_get(name)
        if not config_txt:
            # This case should ideally be handled by server_get raising an error or returning a clear signal
            # For now, if server_get returns None (e.g., file not found), server_validate should indicate this.
            raise ValueError(f"MCP configuration for server '{name}' not found or is empty")

        # convert config txt to json and store in config
        config: Dict[str, Any] = json.loads(config_txt)

        # apparently python JSON allows arrays and strings too
        if not isinstance(config, dict):
            raise ValueError(f"MCP server config for '{name}' is not a valid dictionary")

        # Pass the original config_txt (string) to extract_server_name
        name_from_config = self.extract_server_name(config_txt)
        if not name_from_config:
            # extract_server_name might return None if it can't determine a name
            raise ValueError("Failed to extract MCP server name from the configuration content")

        if name_from_config != name:
            # This comparison works correctly even if 'name' is None.
            # - If name_from_config = "s1" and name = "s1", then "s1" != "s1" is False.
            # - If name_from_config = "s1" and name = "s2", then "s1" != "s2" is True.
            # - If name_from_config = "s1" and name = None, then "s1" != None is True.
            raise ValueError(
                f"MCP server name mismatch. Expected name based on input: '{name}', "
                f"but name found in config: '{name_from_config}'."
            )
        # Check if we have the mcpServers key which is the standard format
        if 'mcpServers' in config:
            # Modern format with mcpServers structure
            if not isinstance(config['mcpServers'], dict):
                raise ValueError(f"'mcpServers' in config for '{name}' is not a dictionary")

            if name not in config['mcpServers']:
                raise ValueError(f"Server '{name}' not found within 'mcpServers' key")

            server_config = config['mcpServers'][name]

            # Check required fields in the server config
            if not isinstance(server_config, dict):
                raise ValueError(f"MCP server config for '{name}' is not a dictionary")

            if 'command' not in server_config:
                raise ValueError(f"Missing required key 'command' in MCP server config for '{name}'")

        else:
            # Legacy format (direct config)?
            logger.warning(f"⚠️ server_validate: Config for '{name}' does not use 'mcpServers' structure")

            # Check for command directly in the old format
            if 'command' not in config:
                raise ValueError(f"Missing required key 'command' in MCP server config for '{name}'")

        # Config is valid
        return config

    # --- 3. Background Task Methods ---

    async def _run_mcp_client_session(self, name: str, params: StdioServerParameters, shutdown_event: asyncio.Event) -> None:
        logger.info(f"[{name}] Starting MCP client session task...")
        session = None  # Define session here to ensure it's accessible in finally block if needed
        try:
            logger.debug(f"[{name}] Attempting stdio_client connection with params: {params}")
            # Simply initialize an empty output buffer
            if name in self.server_tasks:
                self.server_tasks[name]['output_buffer'] = []

            # We can't directly set a callback on StdioServerParameters, but we'll handle error capturing in other ways
            # The stderr outputs will be logged by the stdio_client itself and our timeout handlers will check for errors

            async with stdio_client(params) as (reader, writer):
                logger.debug(f"[{name}] Stdio connection established. Creating ClientSession...")
                # Create the session object without using context manager so we can keep it alive
                # This is crucial for connection reuse
                async with ClientSession(reader, writer) as session:

                    try:
                        # Initialize the session
                        logger.debug(f"[{name}] ClientSession created. Initializing session...")
                        await asyncio.wait_for(session.initialize(), timeout=30.0)  # Increased timeout for reliable init
                        logger.info(f"[{name}] MCP Session initialized successfully.")

                        # Update server_tasks *after* successful initialization
                        if name in self.server_tasks:
                            logger.debug(f"[{name}] Updating server_tasks with session and timestamp...")
                            self.server_tasks[name]['session'] = session
                            self.server_tasks[name]['started_at'] = datetime.datetime.now(datetime.timezone.utc)
                            self.server_tasks[name]['status'] = 'running'  # Mark as running
                            logger.info(f"[{name}] server_tasks updated. Current state for '{name}': {{'status': '{self.server_tasks[name].get('status')}', 'started_at': '{self.server_tasks[name].get('started_at')}', 'session_present': {self.server_tasks[name].get('session') is not None}}}")

                            # Pre-fetch tools list to validate the connection works
                            try:
                                tools_result = await session.list_tools()
                                logger.info(f"[{name}] Successfully fetched {len(tools_result.tools)} tools from running server")
                                # Pretty print tools as JSON
                                tools_json = format_json_log([tool.model_dump() for tool in tools_result.tools])
                                logger.info(f"[{name}] Tools:\n{tools_json}")
                                self.server_tasks[name]['tools_count'] = len(tools_result.tools)
                            except Exception as e:
                                logger.warning(f"[{name}] Failed to fetch tools list: {e}")

                            # Signal readiness *after* updating state
                            if 'ready_event' in self.server_tasks[name]:
                                self.server_tasks[name]['ready_event'].set()
                                logger.debug(f"[{name}] Ready event set.")
                            else:
                                logger.warning(f"[{name}] 'ready_event' not found in server_tasks entry.")
                        else:
                            logger.error(f"[{name}] Entry disappeared from server_tasks before session could be stored!")
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

                                # Health check disabled to support long-running jobs
                                # The session will naturally close if the process dies or connection is lost
                                pass
                            except asyncio.TimeoutError:
                                # This is expected - it just means the shutdown_event.wait timed out
                                # We'll loop and check again
                                pass
                            except Exception as e:
                                # When we get any error in the session loop, capture it thoroughly
                                error_message = str(e)

                                # Detect package dependency errors in the error message
                                if "No solution found" in error_message or "not found in the package registry" in error_message:
                                    key_error = f"Package dependency error: {error_message.strip()}"
                                    logger.error(f"[{name}] 📦 PACKAGE ERROR DETECTED: {key_error}")
                                else:
                                    key_error = f"Startup error: {error_message.strip()}"
                                    logger.error(f"[{name}] 🚫 STARTUP ERROR: {key_error}")

                                # Save the error message in BOTH places for display in tools list
                                # This is the key - we must update both to ensure visibility
                                if name in self.server_tasks:
                                    self.server_tasks[name]['status'] = 'failed'
                                    self.server_tasks[name]['error'] = key_error
                                # Always update the persistent error storage
                                self._server_load_errors[name] = key_error

                                # Debug log the full error
                                if isinstance(e, BaseExceptionGroup):
                                    logger.error(f"[{name}] Error details:", exc_info=True)

                                # Close the session if it exists
                                if session is not None:
                                    try:
                                        await asyncio.wait_for(session.close(), timeout=5.0)
                                    except Exception as close_err:
                                        logger.debug(f"[{name}] Error closing session during cleanup: {close_err}")
                                break

                        logger.info(f"[{name}] Shutdown event received. Cleaning up session...")

                        # Close the session cleanly if we still have it
                        if session is not None:
                            try:
                                await asyncio.wait_for(session.close(), timeout=5.0)
                                logger.debug(f"[{name}] Session closed cleanly")
                            except Exception as e:
                                logger.warning(f"[{name}] Error closing session: {e}")

                    except asyncio.TimeoutError as timeout_err:
                        # Directly check for the uvx error - this is a common pattern in our server
                        common_errors = [
                            "No solution found when resolving tool dependencies",
                            "not found in the package registry",
                            "requirements are unsatisfiable",
                            "weahter-mcp"  # Detect common typo in weather
                        ]

                        # Generic error message to start
                        error_msg = "Timeout occurred during session initialization"

                        # Check for these errors in both output_buffer and recent log entries
                        # First search logs explicitly for these patterns before we time out
                        logger.warning(f"[{name}] 🔍 Searching for dependency errors that might cause timeout...")

                        # Try to look at process stdout/stderr that might be in logs but not in buffer yet
                        # This is a direct debug log of the timeout error
                        tb_str = traceback.format_exc()
                        logger.debug(f"[{name}] Timeout full traceback: {tb_str}")

                        # Look for common package dependency errors in logs
                        for line in self.server_tasks.get(name, {}).get('output_buffer', []):
                            for err_pattern in common_errors:
                                if err_pattern in line:
                                    # Found a more specific error message to use!
                                    error_msg = f"Package dependency error detected: {line.strip()}"
                                    logger.error(f"[{name}] 📦 DEPENDENCY ERROR DETECTED: {line}")
                                    # Check for common typos and provide helpful message
                                    if "weahter" in line:
                                        error_msg += " (Note: 'weather' is misspelled as 'weahter')"
                                        logger.error(f"[{name}] 📝 TYPO DETECTED: 'weahter' should be 'weather'")
                                    break

                        logger.error(f"[{name}] {error_msg}")

                        if name in self.server_tasks:
                            # Mark server as failed
                            self.server_tasks[name]['status'] = 'failed'
                            # Set ready event to unblock anything waiting for this server
                            if 'ready_event' in self.server_tasks[name]:
                                self.server_tasks[name]['ready_event'].set()
                                logger.debug(f"[{name}] Ready event set (to unblock waiters) after initialization failure")
                            # Store the detailed error message
                            self.server_tasks[name]['error'] = error_msg
                            # Add to server load errors so it shows in tools list
                            self._server_load_errors[name] = error_msg

                            # Optional: Remove zombie tasks after a short delay to allow error inspection
                            async def _delayed_cleanup(server_name):
                                await asyncio.sleep(120)  # Keep failed entry for 2 minutes for diagnostics
                                if server_name in self.server_tasks and self.server_tasks[server_name].get('status') == 'failed':
                                    logger.info(f"[{server_name}] Removing failed server from server_tasks after delay")
                                    self.server_tasks.pop(server_name, None)

                            # Schedule cleanup
                            asyncio.create_task(_delayed_cleanup(name))

                        # Close session if it exists but failed to initialize
                        if session is not None:
                            try:
                                await session.close()
                            except Exception:
                                pass  # Ignore errors when closing an uninitialized session
                    except Exception as e:
                        error_message_str = str(e)
                        logger.error(f"[{name}] Error during session.initialize() or operation: {error_message_str}", exc_info=True)

                        # Determine a more specific error key
                        key_error = f"Initialization error: {error_message_str.strip()}" # Default
                        common_patterns = [
                            "No solution found when resolving tool dependencies",
                            "not found in the package registry",
                            "requirements are unsatisfiable",
                            "weahter-mcp"  # Detect common typo in weather
                        ]

                        # Check error message first
                        package_error_detected = False
                        for pattern in common_patterns:
                            if pattern in error_message_str:
                                key_error = f"Package dependency error: {error_message_str.strip()}"
                                logger.error(f"[{name}] 📦 DETECTED PACKAGE ERROR (during init): {key_error}")
                                package_error_detected = True
                                # Check for common typos
                                if "weahter" in error_message_str:
                                    key_error += " (Note: 'weather' is misspelled as 'weahter')"
                                    logger.error(f"[{name}] 📝 TYPO DETECTED: 'weahter' should be 'weather'")
                                break

                        # If not found in the error message, check recent output buffer
                        if not package_error_detected:
                            for line in self.server_tasks.get(name, {}).get('output_buffer', []):
                                for pattern in common_patterns:
                                    if pattern in line:
                                        key_error = f"Package dependency error: {line.strip()}"
                                        logger.error(f"[{name}] 📦 DETECTED PACKAGE ERROR (in output): {key_error}")
                                        # Check for common typos
                                        if "weahter" in line:
                                            key_error += " (Note: 'weather' is misspelled as 'weahter')"
                                            logger.error(f"[{name}] 📝 TYPO DETECTED: 'weahter' should be 'weather'")
                                        package_error_detected = True
                                        break
                                if package_error_detected:
                                    break

                        if name in self.server_tasks:
                            self.server_tasks[name]['status'] = 'init_failed'
                            self.server_tasks[name]['error'] = key_error  # Store the captured error
                            if 'ready_event' in self.server_tasks[name]:
                                self.server_tasks[name]['ready_event'].set()  # Signal completion (failure)

                        # Also store in _server_load_errors for persistence in tools/list
                        self._server_load_errors[name] = key_error

                        # Close session if it exists but failed
                        if session is not None:
                            try:
                                await asyncio.wait_for(session.close(), timeout=5.0)
                            except Exception as close_err:
                                logger.debug(f"[{name}] Error closing session after init failed: {close_err}")
                        # This exception means we should exit _run_mcp_client_session, so we let it proceed to finally
                        return # Exit the _run_mcp_client_session task

        except ConnectionRefusedError:
            logger.error(f"[{name}] Connection refused when starting stdio client process.")
            if name in self.server_tasks:
                self.server_tasks[name]['status'] = 'connection_refused'
                if 'ready_event' in self.server_tasks[name]:
                    self.server_tasks[name]['ready_event'].set()  # Signal completion (failure)

        except Exception as e:
            logger.error(f"[{name}] Unexpected error in client session: {e}", exc_info=True)
            if name in self.server_tasks:
                self.server_tasks[name]['status'] = 'error'
                if 'ready_event' in self.server_tasks[name]:
                    self.server_tasks[name]['ready_event'].set()  # Signal completion (failure)

        finally:
            # Clean up if needed (task cancellation, etc.)
            logger.info(f"[{name}] Client session task exiting.")

            # CRITICAL: Always ensure the ready_event is set to prevent hanging
            if name in self.server_tasks and 'ready_event' in self.server_tasks[name]:
                if not self.server_tasks[name]['ready_event'].is_set():
                    logger.debug(f"[{name}] Setting ready_event in finally block to prevent hanging")
                    self.server_tasks[name]['ready_event'].set()

            # Force-clear session reference to help with garbage collection
            if session is not None:
                logger.debug(f"[{name}] Clearing session reference.")
                try:
                    await session.close()  # Ensure session is properly closed
                except Exception as e:
                    logger.debug(f"[{name}] Error closing session during cleanup: {e}")
                session = None

            if name in self.server_tasks:
                logger.debug(f"[{name}] Clearing session and start time from server_tasks.")
                self.server_tasks[name]['started_at'] = None
                self.server_tasks[name]['session'] = None
                # Remove status field if it exists (only needed for error states)
                self.server_tasks[name].pop('status', None)

    # --- 4. Server Start/Stop Methods ---

    async def server_start(self, args: Dict[str, Any], server) -> str:
        name = args.get('name')
        logger.info(f"▶️ server_start: Starting server '{name}'...")
        if not name or not isinstance(name, str):
            msg = "Missing or invalid parameter: 'name' must be str."
            logger.error(f"❌ server_start: {msg}")
            raise ValueError(msg)

        # Load the config using server_get (as in self_test)
        config_txt = await self.server_get(name)
        if not config_txt:
            raise ValueError(f"No config found for server '{name}'")

        # convert config txt to json and store in full_config
        full_config: Dict[str, Any] = json.loads(config_txt)

        if name in self.server_tasks:
            task_info = self.server_tasks[name]
            # Check both task state and status to determine if truly running
            if (task_info.get('started_at') is not None):
                msg = f"MCP server '{name}' is already running."
                logger.warning(f"⚠️ server_start: {msg}")
                raise ValueError(msg)

        try:
            # server_validate now returns the config dict directly or raises ValueError
            validation_result = await self.server_validate(name)
            logger.debug(f"▶️ server_start: Successfully validated config for '{name}': {validation_result}")
        except ValueError as e:
            msg = f"MCP server configuration validation failed for '{name}': {e}"
            logger.error(f"❌ server_start: {msg}")
            raise ValueError(msg) from e # Re-raise to indicate server_start failure

        # Determine the actual server configuration details from the full config
        if 'mcpServers' in validation_result:
            server_config_details = validation_result['mcpServers'].get(name)
            if not server_config_details:
                # This case should ideally be caught by server_validate, but defensive check
                err_msg = f"MCP server '{name}' not found within 'mcpServers' in validated config."
                logger.error(f"❌ server_start: {err_msg}")
                raise ValueError(err_msg)
        else:
            # Legacy format (direct config)
            server_config_details = validation_result

        command = server_config_details.get('command')

        try:
            # Prepare parameters for stdio_client
            server_config = full_config['mcpServers'][name]
            params = StdioServerParameters(
                command=server_config['command'],
                args=server_config.get('args', []),
                env=server_config.get('env', {}),
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

        # Start the MCP client session as an async task
        logger.debug(f"---> _add_server_task: Starting MCP client session for '{name}'")
        task = asyncio.create_task(self._run_mcp_client_session(name, params, shutdown_event))
        task_info = {
            'task': task,
            'config': full_config,
            'shutdown_event': shutdown_event,
            'ready_event': ready_event,
            'session': None,  # Will be populated once initialized
            'status': 'starting',  # Track status: starting -> running or failed
            'added_at': datetime.datetime.now(datetime.timezone.utc),
            'output_buffer': [],  # Store stdout/stderr from the process
            'error_buffer': []    # Store specific error messages
        }
        self.server_tasks[name] = task_info
        logger.info(f"Starting MCP server '{name}'")
        logger.debug(f"▶️ server_start: Task info recorded for '{name}'. Start time will be added upon session init.")

        # Return success - plain string (MCP formatting happens in _format_mcp_response)
        return f"Attempting to start MCP service '{name}'; please check status"

    async def server_stop(self, args: Dict[str, Any], server) -> str:
        name = args.get('name')
        if not name or not isinstance(name, str):
            msg = "Missing or invalid parameter: 'name' must be str."
            logger.error(f"❌ server_stop: {msg}")
            raise ValueError(msg)

        if name not in self.server_tasks:
            msg = f"MCP server '{name}' is either invalid or not started"
            logger.warning(f"⚠️ server_stop: {msg}")
            raise ValueError(msg)

        task_info = self.server_tasks.get(name)
        if not task_info or 'task' not in task_info:
            msg = f"Error stopping MCP server '{name}': Inconsistent state."
            logger.error(f"❌ server_stop: {msg}")
            # Clean up if entry exists but is broken
            if name in self.server_tasks:
                del self.server_tasks[name]
            raise RuntimeError(msg)

        task = task_info['task']
        if task.done():
            logger.info(f"🧹 Task for server '{name}' already stopped. Cleaning up entry.")
            return f"MCP server '{name}' not running"
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


            return f"MCP server '{name}' stopped"
