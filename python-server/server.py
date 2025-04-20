#!/usr/bin/env python3
import logging
import json
import inspect
import asyncio
import os
import importlib
import re
import signal
import sys
import time
import random
import socketio
import argparse
from utils import check_server_running, create_pid_file, remove_pid_file, clean_filename, format_json_log
from typing import Any, Callable, Dict, List, Optional, Union
import datetime

# Version
SERVER_VERSION = "0.1.0"

from mcp.server import Server
# Removed websocket_server import - implementing our own handler
from mcp.client.websocket import websocket_client
from mcp.types import Tool, TextContent, CallToolResult, ToolListChangedNotification, NotificationParams, Annotations
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
import uvicorn

# Import from our newly created modules
from state import (
    logger, HOST, PORT,
    FUNCTIONS_DIR, is_shutting_down, cloud_connection_active,
    CLOUD_SERVER_HOST, CLOUD_SERVER_PORT, CLOUD_SERVER_URL,
    CLOUD_SERVICE_NAMESPACE, CLOUD_CONNECTION_RETRY_SECONDS,
    CLOUD_CONNECTION_MAX_RETRIES, CLOUD_CONNECTION_MAX_BACKOFF_SECONDS,
    tasks, BOLD, RESET, CYAN, BRIGHT_WHITE
)

# Import dynamic function management utilities
from dynamic_manager import (
    function_set,
    function_add,
    function_remove,
    function_validate,
    function_call
)
from dynamic_manager import _fs_load_code

# Import our utility module for dynamic functions
import utils

# Import server manager functions
from server_manager import server_list, server_get, server_add, server_remove, server_set, server_validate

# NOTE: This server uses two different socket protocols:
# 1. Standard WebSockets: When acting as a SERVER to accept connections from node-mcp-client
# 2. Socket.IO: When acting as a CLIENT to connect to the cloud Node.js server
#
# Each server dictates its own protocol, and clients must adapt accordingly.
# - The node-mcp-client connects via standard WebSockets to our server.py
# - Our server.py connects via Socket.IO to the cloud server
# - Both ultimately route to the same MCP handlers in the DynamicAdditionServer class

# Define signal handler for graceful shutdown
def handle_sigint(signum, frame):
    global is_shutting_down
    if not is_shutting_down:
        is_shutting_down = True
        print("\n🐱 Meow! Graceful shutdown in progress... Press Ctrl+C again to force exit! 🐱")
        print("🧹 Cleaning up resources and closing connections...")
        # Signal the cloud connection to close
        if 'cloud_connection' in globals() and cloud_connection is not None:
            logger.info("☁️ Closing cloud server connection...")
            asyncio.create_task(cloud_connection.disconnect())
        remove_pid_file()
    else:
        print("\n🚨 Forced exit! 🚨")
        # Ensure PID file is removed even on force exit
        remove_pid_file()
        sys.exit(1)

# Register signal handler
signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# Check if server is already running
existing_pid = check_server_running()
if existing_pid:
    logger.info(f"ℹ️ Server is already running with PID: {existing_pid}. Exiting...")
    sys.exit(0)

# Create PID file
if not create_pid_file():
    logger.error("❌ Failed to create PID file! Exiting...")
    sys.exit(1)







# Function to get code for a dynamic function
async def get_function_code(args, mcp_server) -> list[TextContent]:
    """
    Get the source code for a dynamic function by name using _fs_load_code.
    Returns the code as a TextContent object.
    """
    # Get function name
    name = args.get("name")

    # Validate parameters
    if not name:
        raise ValueError("Missing required parameter: name")

    # Load the code using the existing _fs_load_code utility
    code = _fs_load_code(name)
    if code is None:
        raise ValueError(f"Function '{name}' not found or could not be read")

    logger.info(f"📋 Retrieved code for function: {name}")

    # Return the code as text content
    return [TextContent(type="text", text=code)]

# Create an MCP server with proper MCP protocol handling
class DynamicAdditionServer(Server):
    """MCP server that provides an addition tool and supports dynamic function registration"""

    def __init__(self):
        super().__init__("Dynamic Function Server")
        self._cached_tools: Optional[List[Tool]] = None # Cache for tool list
        self._last_functions_dir_mtime: float = 0.0 # Timestamp for cache invalidation

        # Register tool handlers using SDK decorators
        # These now wrap the actual logic methods defined below

        @self.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """SDK Handler for tools/list"""
            logger.info(f"🚀 Processing MCP tool list request")
            # Pass context indicating this call is from the SDK handler (likely a direct request)
            return await self._get_tools_list(caller_context="handle_list_tools_sdk")

        @self.list_prompts()
        async def handle_list_prompts() -> list:
            """SDK Handler for prompts/list"""
            return await self._get_prompts_list()

        @self.list_resources()
        async def handle_list_resources() -> list:
            """SDK Handler for resources/list"""
            return await self._get_resources_list()

        @self.call_tool()
        async def handle_call_tool(name: str, args: dict) -> list[TextContent]:
            """SDK Handler for tools/call"""
            return await self._execute_tool(name=name, args=args)

    # Initialization for function discovery
    async def initialize(self, params={}):
        """Initialize the server, sending a toolsList notification with initial tools"""
        logger.info(f"{CYAN}🔧 === ENTERING SERVER INITIALIZE METHOD ==={RESET}")
        logger.info(f"🚀 Server initialized with version {SERVER_VERSION}")

        # Set the server instance in utils module for client logging
        utils.set_server_instance(self)
        logger.info("🔌 Dynamic functions utility module initialized")

        tools_list = await self._get_tools_list(caller_context="initialize_method")
        try:
            # Pass the already fetched list to the notification function
            await self.send_tool_notification(tools=tools_list)
        except Exception as e:
            logger.warning(f"Could not send initial tool notification: {str(e)}")
        logger.info(f"{CYAN}🔧 Server initialize method completed.{RESET}")
        return {}  # Empty response per MCP protocol

    async def send_tool_notification(self, tools: Optional[list[Tool]] = None):
        """Send a tool list changed notification to clients

        Args:
            tools: An optional list of tools. If provided, this list is used directly.
                   If None, the current tool list will be fetched.
        """
        try:
            # Get the current list of tools only if not provided
            if tools is None:
                logger.info("📢 Fetching tools list for notification as it was not provided.")
                tools = await self._get_tools_list(caller_context="send_tool_notification_fallback")
            else:
                logger.info("📢 Using provided tools list for notification.")

            # Create the notification parameters
            notification_params = NotificationParams(
                changed_tools=tools
            )

            # Create the notification
            notification = ToolListChangedNotification(
                method="notifications/tools/list_changed",
                params=notification_params
            )

            # Send the notification
            if hasattr(self, 'service_connections'):
                for client in self.service_connections.values():
                    await client.send_notification('tools/listChanged', notification.params.model_dump())
                logger.info(f"📢 Sent tool list notification to {len(self.service_connections)} clients")

        except Exception as e:
            logger.error(f"❌ Error sending tool notification: {str(e)}")
            # Don't re-raise, as this is a notification and shouldn't fail the main operation

    # --- Core Logic Methods (callable directly) ---

    async def _get_tools_list(self, caller_context: str = "unknown") -> list[Tool]:
        """Core logic to return a list of available tools"""
        logger.info(f"{BRIGHT_WHITE}📋 === GETTING TOOL LIST (Called by: {caller_context}) ==={RESET}")

        # --- Caching Logic ---
        try:
            # Ensure the directory exists before checking mtime
            if not os.path.exists(FUNCTIONS_DIR):
                 os.makedirs(FUNCTIONS_DIR, exist_ok=True)
                 logger.info(f"📂 Created missing FUNCTIONS_DIR: {FUNCTIONS_DIR}")
                 current_mtime = os.path.getmtime(FUNCTIONS_DIR)
            else:
                 current_mtime = os.path.getmtime(FUNCTIONS_DIR)

            if current_mtime == self._last_functions_dir_mtime and self._cached_tools is not None:
                logger.info(f"⚡️ USING CACHED TOOL LIST (DIR UNCHANGED - mtime: {current_mtime})")
                # Return a copy to prevent external modification of the cache
                return list(self._cached_tools)
            logger.info(f"🔄 FUNCTIONS DIR MODIFIED (mtime: {current_mtime} vs last: {self._last_functions_dir_mtime}) or cache empty, REGENERATING TOOL LIST")
        except Exception as e:
             logger.error(f"❌ Error checking FUNCTIONS_DIR mtime: {e}. Proceeding without cache.")
             current_mtime = time.time() # Use current time to force regeneration

        # Start with our built-in tools
        tools_list = [
            Tool(
                name="_function_set",
                description="Sets the content of a dynamic Python function", # Updated description
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The Python source code containing a single function definition."}
                    },
                    "required": ["code"] # Only code is required now
                }
            ),
            Tool( # Add definition for get_tool_code
                name="_function_get",
                description="Gets the Python source code for a dynamic function",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the function to get code for"}
                    },
                    "required": ["name"]
                }
            ),
            Tool( # Add definition for remove_dynamic_tool
                name="_function_remove",
                description="Removes a dynamic Python function",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the function to remove"}
                    },
                    "required": ["name"]
                }
            ),
            Tool( # Add definition for add_placeholder_function
                name="_function_add",
                description="Adds a new, empty placeholder Python function with the given name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name to register the new placeholder function under."}
                    },
                    "required": ["name"]
                }
            ),
            # MCP server CRUD tools
            Tool(
                name="_server_list",
                description="Lists all configured MCP servers",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="_server_get",
                description="Gets the configuration JSON for a server",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Server name to fetch"}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_add",
                description="Adds a new MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "config": {"type": "object", "description": "Server config JSON"}
                    },
                    "required": ["name", "config"]
                }
            ),
            Tool(
                name="_server_remove",
                description="Removes an MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="_server_set",
                description="Sets (adds or updates) an MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "config": {"type": "object"}
                    },
                    "required": ["name", "config"]
                }
            ),
            Tool(
                name="_server_validate",
                description="Validates an MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            )
        ]
        # Log the static tools being included
        static_tool_names = [tool.name for tool in tools_list]
        logger.info(f"🔩 INCLUDING {len(static_tool_names)} STATIC TOOLS: {', '.join(static_tool_names)}")

        # Scan the FUNCTIONS_DIR directory for dynamic functions
        try:
            # Ensure the functions directory exists
            os.makedirs(FUNCTIONS_DIR, exist_ok=True)

            # Get all Python files in the directory
            function_files = [f for f in os.listdir(FUNCTIONS_DIR) if f.endswith('.py')]
            logger.info(f"📝 FOUND {len(function_files)} POTENTIAL DYNAMIC FUNCTIONS")

            # For each Python file, create a Tool entry
            for file_name in function_files:
                tool_name_from_file = os.path.splitext(file_name)[0]
                try:
                    # Skip if this seems to be a utility file
                    if tool_name_from_file.startswith('_') or tool_name_from_file == '__init__' or tool_name_from_file == '__pycache__':
                        continue

                    # Validate the function and get info
                    validation_result = function_validate(tool_name_from_file)
                    is_valid = validation_result.get('valid', False)
                    error_message = validation_result.get('error')
                    function_info = validation_result.get('function_info') # Should be a dict if valid

                    # --- Create Tool --- #
                    tool_name = tool_name_from_file
                    tool_description = f"Dynamic function: {tool_name_from_file}"
                    tool_input_schema = {"type": "object", "properties": {}}
                    tool_annotations = {}

                    if is_valid and function_info:
                         # Use extracted info if valid and available
                         tool_name = function_info.get('name', tool_name_from_file) # Use AST name, fallback to filename
                         tool_description = function_info.get('description', f"Dynamic function '{tool_name}'")
                         tool_input_schema = function_info.get('inputSchema', {"type": "object", "properties": {}})
                         tool_annotations["validationStatus"] = "VALID"
                    elif is_valid and not function_info:
                         # Valid syntax but failed to extract info (should ideally not happen)
                         tool_description = f"Dynamic function: {tool_name_from_file} (Details unavailable)"
                         tool_input_schema = {"type": "object", "description": "Could not parse arguments."}
                         tool_annotations["validationStatus"] = "VALID_SYNTAX_UNKNOWN_STRUCTURE"
                    else:
                         # Invalid syntax
                         tool_description = f"Dynamic function: {tool_name_from_file} (INVALID)"
                         tool_input_schema = {"type": "object", "description": "Function has syntax errors."}
                         tool_annotations["validationStatus"] = "INVALID"
                         if error_message:
                             tool_annotations["errorMessage"] = error_message

                    # Add common annotations
                    try:
                        tool_annotations["lastModified"] = datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(FUNCTIONS_DIR, file_name))
                        ).isoformat()
                    except Exception:
                        pass # Ignore if file stat fails

                    # Create and add the tool object
                    tool_obj = Tool(
                        name=tool_name,
                        description=tool_description,
                        inputSchema=tool_input_schema,
                        annotations=tool_annotations
                    )
                    tools_list.append(tool_obj)
                    logger.debug(f"📝 Added dynamic tool: {tool_name}, valid: {is_valid}")

                except Exception as e:
                    logger.warning(f"⚠️ Error processing potential tool file {file_name}: {str(e)}")
                    # Add a placeholder indicating the error
                    tools_list.append(Tool(
                        name=tool_name_from_file,
                        description=f"Error loading dynamic function: {str(e)}",
                        inputSchema={"type": "object"},
                        annotations={"validationStatus": "ERROR_LOADING"}
                    ))
                    continue

        except Exception as e:
            logger.error(f"❌ Error scanning for dynamic functions: {str(e)}")
            # Continue with just the built-in tools

        logger.info(f"📝 RETURNING {len(tools_list)} TOOLS")

        # --- Update Cache ---
        self._cached_tools = list(tools_list) # Store a copy
        self._last_functions_dir_mtime = current_mtime
        logger.info(f"💾 CACHED TOOL LIST (Timestamp: {current_mtime})")

        return tools_list

    async def _get_prompts_list(self) -> list:
        """Core logic to return a list of available prompts (empty stub)"""
        # Currently no prompts supported
        return []

    async def _get_resources_list(self) -> list:
        """Core logic to return a list of available resources (empty stub)"""
        # Currently no resources supported
        return []

    async def send_client_log(self, level: str, data: Any, logger_name: str = None, client_id: str = None):
        """Send a log message notification to connected clients using direct WebSocket communication.

        Args:
            level: The log level ("debug", "info", "warning", "error")
            data: The log message content (can be string or structured data)
            logger_name: Optional name to identify the logger source
        """
        try:
            # Normalize level to uppercase for consistency
            level = level.upper()

            # Create a simple JSON-RPC notification structure
            # This bypasses the SDK entirely and sends a direct WebSocket message
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {
                    "level": level,
                    "data": data,
                    "logger": logger_name or "dynamic_function"
                }
            }

            # Convert to JSON string
            import json
            notification_json = json.dumps(notification)

            # Log the notification for debugging
            logger.debug(f"Sending client log notification: {notification_json}")

            # Directly send to all connected WebSocket clients
            sent_count = 0

            # Get the global tracking collections
            global active_websockets, client_connections, current_request_client_id

            # If no specific client_id was provided, try to use the one from the current request
            if client_id is None and 'current_request_client_id' in globals():
                client_id = current_request_client_id

            # ONLY send to the specific client that made the request - NO broadcasting
            if client_id and client_id in client_connections:
                client_info = client_connections[client_id]
                client_type = client_info.get("type")
                connection = client_info.get("connection")

                if client_type == "websocket" and connection:
                    try:
                        await connection.send_text(notification_json)
                        sent_count += 1
                        logger.debug(f"📢 Sent notification to specific client: {client_id}")
                    except Exception as e:
                        logger.warning(f"Failed to send to client {client_id}: {e}")

                elif client_type == "cloud" and connection and connection.is_connected:
                    try:
                        await connection.send_message('mcp_notification', notification)
                        sent_count += 1
                        logger.debug(f"☁️ Sent notification to cloud client: {client_id}")
                    except Exception as e:
                        logger.warning(f"Failed to send to cloud client {client_id}: {e}")
            else:
                # If we don't know which client to send to, this is a problem - don't send to anyone
                logger.warning(f"Cannot send client log: no valid client_id provided or client not found: {client_id}")
                # Log the client connections we know about for debugging
                logger.debug(f"Known client connections: {list(client_connections.keys())}")

            if sent_count > 0:
                logger.debug(f"📢 Notification sent successfully to client {client_id}")
            else:
                logger.debug(f"Failed to send notification to client {client_id}")

        except Exception as e:
            # Don't let logging errors affect the main operation
            logger.error(f"❌ Error sending direct client log notification: {str(e)}")
            import traceback
            logger.debug(f"Log notification error details: {traceback.format_exc()}")
            # We intentionally don't re-raise here


    async def _execute_tool(self, name: str, args: dict, client_id: str = None) -> list[TextContent]:
        """Core logic to handle a tool call. Ensures result is List[TextContent(type='text')]"""
        logger.info(f"🔧 EXECUTING TOOL: {name}")
        logger.debug(f"WITH ARGUMENTS: {args}")
        # ---> ADDED: Log entry and raw args
        logger.debug(f"---> _execute_tool ENTERED. Name: '{name}', Raw Args: {args!r}") # <-- ADD THIS LINE

        try:
            result_raw = None # Initialize raw result variable
            # Handle built-in tool calls
            if name == "_function_set":
                logger.debug(f"---> Calling built-in: function_set") # <-- ADD THIS LINE
                result_raw = await function_set(args, self)
            elif name == "_function_get":
                logger.debug(f"---> Calling built-in: get_function_code") # <-- ADD THIS LINE
                result_raw = await get_function_code(args, self)
            elif name == "_function_remove":
                # Remove function
                func_name = args.get("name")
                if not func_name:
                    raise ValueError("Missing required parameter: name")

                logger.debug(f"---> Calling built-in: function_remove for '{func_name}'") # <-- ADD THIS LINE

                # Check if function exists before attempting to remove
                function_path = os.path.join(FUNCTIONS_DIR, f"{func_name}.py")
                if not os.path.exists(function_path):
                    result_raw = [TextContent( # <-- CHANGE TO result_raw
                        type="text",
                        text=f"Function '{func_name}' does not exist or was already removed."
                    )]
                else: # <-- ADD else block
                    # Remove the function using dynamic_manager.function_remove
                    try:
                        if function_remove(func_name):
                            try:
                                await self.send_tool_notification()
                            except Exception as e:
                                logger.error(f"Error sending tool notification: {str(e)}")
                            result_raw = [TextContent(type="text", text=f"Function '{func_name}' removed successfully.")] # <-- CHANGE TO result_raw
                        else:
                            result_raw = [TextContent( # <-- CHANGE TO result_raw
                                type="text",
                                text=f"Error removing function '{func_name}'. Check server logs for details."
                            )]
                    except Exception as e:
                        logger.error(f"Error removing function: {str(e)}")
                        result_raw = [TextContent( # <-- CHANGE TO result_raw
                            type="text",
                            text=f"Error removing function '{func_name}': {str(e)}"
                        )]

            elif name == "_function_add":
                # Add empty function
                func_name = args.get("name")
                if not func_name:
                    raise ValueError("Missing required parameter: name")

                logger.debug(f"---> Calling built-in: function_add for '{func_name}'") # <-- ADD THIS LINE

                # Check if function already exists
                function_path = os.path.join(FUNCTIONS_DIR, f"{func_name}.py")
                if os.path.exists(function_path):
                    # Function already exists - inform the client rather than raise error
                    result_raw = [TextContent( # <-- CHANGE TO result_raw
                        type="text",
                        text=f"Function '{func_name}' already exists"
                    )]
                else: # <-- ADD else block
                    # Create empty function (stub) using dynamic_manager.function_add
                    try:
                        if function_add(func_name):
                            try:
                                await self.send_tool_notification()
                            except Exception as e:
                                logger.error(f"Error sending tool notification: {str(e)}")
                            result_raw = [TextContent(type="text", text=f"Empty function '{func_name}' created successfully.")] # <-- CHANGE TO result_raw
                        else:
                            result_raw = [TextContent( # <-- CHANGE TO result_raw
                                type="text",
                                text=f"Error creating function '{func_name}'. Check server logs for details."
                            )]
                    except Exception as e:
                        logger.error(f"Error creating function: {str(e)}")
                        result_raw = [TextContent( # <-- CHANGE TO result_raw
                            type="text",
                            text=f"Error creating function '{func_name}': {str(e)}"
                        )]
            # MCP server CRUD tool cases
            elif name == "_server_list":
                logger.debug("---> Calling built-in: server_list")
                result_raw = server_list()
            elif name == "_server_get":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_get for '{svc_name}'")
                result_raw = server_get(svc_name)
            elif name == "_server_add":
                svc_name = args.get("name")
                config = args.get("config")
                if not svc_name or not isinstance(config, dict):
                    raise ValueError("Missing or invalid parameters: 'name' must be str and 'config' must be dict")
                logger.debug(f"---> Calling built-in: server_add for '{svc_name}'")
                success = server_add(svc_name, config)
                if success:
                    result_raw = [TextContent(type="text", text=f"Server '{svc_name}' added successfully.")]
                else:
                    result_raw = [TextContent(type="text", text=f"Failed to add server '{svc_name}'.")]
            elif name == "_server_remove":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_remove for '{svc_name}'")
                success = server_remove(svc_name)
                result_raw = [TextContent(type="text", text=f"Server '{svc_name}' removed successfully.")] if success else [TextContent(type="text", text=f"Failed to remove server '{svc_name}'.")]
            elif name == "_server_set":
                logger.debug(f"---> Calling built-in: server_set with args: {args!r}")
                result_raw = await server_set(args, self)
            elif name == "_server_validate":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_validate for '{svc_name}'")
                result_raw = server_validate(svc_name)
            # Handle dynamic function calls
            elif not name.startswith('_'):  # Only non-underscore names are potential dynamic functions
                # Check if function exists
                function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")
                if not os.path.exists(function_path):
                    raise ValueError(f"Function '{name}' not found")

                # Call the dynamic function
                try:
                    logger.info(f"🔧 CALLING DYNAMIC FUNCTION: {name}")
                    logger.debug(f"---> Calling dynamic: function_call for '{name}' with args: {args}") # <-- ADD THIS LINE
                    result_raw = function_call(name, **args)
                    logger.debug(f"<--- Dynamic function '{name}' RAW result: {result_raw!r} (type: {type(result_raw)})") # <-- ADD THIS LINE

                except Exception as e:
                    logger.error(f"❌ Error during dynamic function call '{name}': {str(e)}", exc_info=True)
                    raise ValueError(f"Error executing function '{name}': {str(e)}") from e
            else:
                # Handle unknown tool names starting with '_', or if no branch matched
                logger.error(f"❓ Unknown or unhandled tool name: {name}")
                raise ValueError(f"Unknown or unhandled tool name: {name}")

            # ---> ADDED: Process raw result into final_result format (List[TextContent])
            if isinstance(result_raw, str):
                final_result = [TextContent(type="text", text=result_raw)]
            elif isinstance(result_raw, list) and all(isinstance(item, TextContent) for item in result_raw):
                final_result = result_raw
            elif result_raw is None: # Handle cases where built-ins might not have set result_raw (e.g., error occurred before assignment)
                logger.error(f"❓ result_raw was None for tool '{name}'. This might indicate an unhandled path or early error.")
                raise ValueError(f"Internal error processing tool '{name}': result was None.")
            else:
                # Convert any other result to string
                import json
                try:
                    result_str = json.dumps(result_raw)
                except TypeError:
                    result_str = str(result_raw) # Fallback to plain string conversion
                logger.warning(f"⚠️ Tool '{name}' returned non-standard type {type(result_raw)}. Converting to JSON string: {result_str}")
                final_result = [TextContent(type="text", text=result_str)]

            # ---> ADDED: Log final result before returning
            logger.debug(f"<--- _execute_tool RETURNING final result: {final_result!r}") # <-- ADD THIS LINE
            return final_result

        except Exception as e:
            logger.error(f"❌ Error in _execute_tool for '{name}': {str(e)}", exc_info=True)
            # Return a generic error message as TextContent
            error_message = f"Error executing tool '{name}': {str(e)}"
            final_result = [TextContent(type="text", text=error_message)]
            logger.debug(f"<--- _execute_tool RETURNING error result: {final_result!r}") # <-- ADD THIS LINE
            return final_result




# ServiceClient class to manage the connection to the cloud server via Socket.IO
class ServiceClient:
    """Socket.IO client for connecting to the cloud server

    This class implements a Socket.IO CLIENT to connect TO the cloud server.
    Socket.IO is different from standard WebSockets - it adds features like:
    - Automatic reconnection
    - Fallback to long polling when WebSockets aren't available
    - Namespaces for multiplexing
    - Authentication handling

    While server.py acts as a WebSocket SERVER for the node-mcp-client,
    it must act as a Socket.IO CLIENT to connect to the cloud server.

    Manages the Socket.IO connection to the cloud server's service namespace.
    """
    def __init__(self, server_url: str, namespace: str, email: str, api_key: str, serviceName: str, mcp_server: Server, port: int):
        self.server_url = server_url
        self.namespace = namespace
        self.email = email
        self.api_key = api_key
        self.serviceName = serviceName
        self.mcp_server = mcp_server
        self.server_port = port # Store the server's listening port
        self.sio = None
        self.retry_count = 0
        self.connection_task = None
        self.is_connected = False
        self.connection_active = True

    async def connect(self):
        """Establish a Socket.IO connection to the cloud server"""
        logger.info(f"☁️ CONNECTING TO CLOUD SERVER: {self.server_url} (namespace: {self.namespace})")
        self.connection_active = True
        self.connection_task = asyncio.create_task(self._maintain_connection())
        return self.connection_task

    async def _maintain_connection(self):
        """Maintains the connection to the cloud server with retries"""
        while self.connection_active and not is_shutting_down:
            try:
                # Create a new Socket.IO client instance
                self.sio = socketio.AsyncClient()

                # Register event handlers
                self._register_event_handlers()

                logger.info(f"☁️ Attempting connection to cloud server (attempt {self.retry_count + 1})")

                # Connect with authentication data including hostname
                import socket
                hostname = socket.gethostname()
                await self.sio.connect(
                    self.server_url,
                    namespaces=[self.namespace],
                    transports=['websocket'],  # Prefer websocket
                    auth={
                        "email": self.email,
                        "apiKey": self.api_key,
                        "serviceName": self.serviceName,
                        "hostname": hostname,
                        "port": self.server_port # Send the stored port
                    },
                    retry=False # We handle retries manually with backoff
                )

                # Wait for disconnection
                await self.sio.wait()
                logger.info("☁️ Socket.IO connection closed")

            except Exception as e:
                if is_shutting_down:
                    logger.info("☁️ Shutting down, stopping cloud connection attempts")
                    break

                self.is_connected = False
                self.sio = None

                if CLOUD_CONNECTION_MAX_RETRIES is not None and self.retry_count >= CLOUD_CONNECTION_MAX_RETRIES:
                    logger.error(f"❌ FAILED TO CONNECT TO CLOUD SERVER AFTER {self.retry_count} ATTEMPTS: {str(e)}")
                    logger.error("❌ GIVING UP ON CLOUD CONNECTION!")
                    break

                self.retry_count += 1
                logger.warning(f"⚠️ CLOUD SERVER CONNECTION ERROR (attempt {self.retry_count}): {str(e)}")

                # Print a more detailed stack trace for debugging
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")

                # Calculate exponential backoff delay with jitter
                backoff_delay = CLOUD_CONNECTION_RETRY_SECONDS * (2 ** self.retry_count)
                jitter = random.uniform(0, 1) # Add random jitter (0-1 seconds)
                actual_delay = min(backoff_delay + jitter, CLOUD_CONNECTION_MAX_BACKOFF_SECONDS)

                # Wait before retrying
                logger.info(f"☁️ RETRYING CONNECTION IN {actual_delay:.2f} SECONDS...")
                await asyncio.sleep(actual_delay)

        logger.info("☁️ Cloud connection maintenance loop ended")

    def _register_event_handlers(self):
        """Register Socket.IO event handlers"""
        if not self.sio:
            return

        # Connection established event
        @self.sio.event(namespace=self.namespace)
        async def connect(): # Ensure handler is async
            self.is_connected = True
            self.retry_count = 0  # Reset retry counter on successful connection
            logger.info("✅ CONNECTED TO CLOUD SERVER!")

            # Get the list of tools to log them
            tools_list = await self.mcp_server._get_tools_list(caller_context="_handle_connect_cloud")
            tool_names = [tool.name for tool in tools_list]
            logger.info(f"📊 REGISTERING {len(tools_list)} TOOLS WITH CLOUD: {', '.join(tool_names)}")

            # Emit the client event upon successful connection
            await self.send_message('client', {'status': 'connected'})

        # Connection error event
        @self.sio.event(namespace=self.namespace)
        def connect_error(data):
            logger.error(f"❌ CONNECTION ERROR: {data}")

        # Disconnection event
        @self.sio.event(namespace=self.namespace)
        def disconnect():
            logger.info(f"☁️ DISCONNECTED FROM CLOUD SERVER")
            self.is_connected = False


        # Service message event
        @self.sio.event(namespace=self.namespace)
        async def service_message(data):
            logger.debug(f"☁️ RAW RECEIVED SERVICE MESSAGE: {data}")
            # Check if this is an MCP JSON-RPC request
            if isinstance(data, dict) and 'jsonrpc' in data and 'method' in data:
                # This is an MCP JSON-RPC request
                response = await self._process_mcp_request(data)
                if response:
                    await self.send_message('mcp_response', response)
            else:
                # Ignore non-JSON-RPC messages or log a warning
                logger.warning(f"⚠️ Received non-JSON-RPC message, ignoring: {data}")

    async def _process_mcp_request(self, request: dict) -> Union[dict, None]:
        # Generate a cloud client ID
        client_id = f"cloud_{int(time.time())}_{id(request)}"
        """Process an MCP JSON-RPC request from the cloud server by manually routing
        to the appropriate logic method in the DynamicAdditionServer.
        """
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        logger.info(f"☁️ Processing MCP request via manual routing: {method} (ID: {request_id})")

        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }

        try:
            if method == "tools/list":
                # Call the core logic method directly (pass client_id)
                result_list = await self.mcp_server._get_tools_list(caller_context="process_mcp_request")
                # Convert Tool objects to dictionaries for JSON serialization using model_dump()
                response["result"] = {"tools": [tool.model_dump() for tool in result_list]} # Use model_dump()

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments") # MCP spec uses 'arguments'

                if tool_name is None or tool_args is None:
                    response["error"] = {"code": -32602, "message": "Invalid params: missing tool name or arguments"}
                else:
                    logger.debug(f"☁️ Calling _execute_tool for: {tool_name}")
                    # Register this client connection
                    global client_connections
                    if not hasattr(self, "cloud_client_id"):
                        self.cloud_client_id = client_id
                    client_connections[client_id] = {"type": "cloud", "connection": self}

                    # Call the core logic method directly with client ID
                    call_result_list = await self.mcp_server._execute_tool(name=tool_name, args=tool_args, client_id=client_id)
                    # The _execute_tool method ensures result is List[TextContent]
                    # Convert TextContent objects to dictionaries for JSON serialization using model_dump()
                    # IMPORTANT: We use "contents" (plural) key to match format between Python and Node servers
                    response["result"] = {"contents": [content.model_dump() for content in call_result_list]}
            else:
                # Unknown method
                response["error"] = {"code": -32601, "message": f"Method not found: {method}"}

            return response

        except Exception as e:
            logger.error(f"❌ ERROR PROCESSING MCP REQUEST: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            response["error"] = {"code": -32000, "message": str(e)}
            return response

    async def send_message(self, event: str, data: dict):
        """Send a message to the cloud server via a named event"""
        if not self.sio or not self.is_connected:
            logger.warning(f"⚠️ ATTEMPTED TO SEND {event} BUT NOT CONNECTED")
            return False

        try:
            # Format data nicely if it's an mcp_response
            #print_log_data = format_json_log(data)
            log_data = str(data)
            logger.debug(f"☁️ SENDING MESSAGE: {event} - \n{log_data}") # Log formatted data on new line
            await self.sio.emit(event, data, namespace=self.namespace)

            return True
        except Exception as e:
            logger.error(f"❌ ERROR SENDING MESSAGE: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from the cloud server"""
        logger.info("☁️ DISCONNECTING FROM CLOUD SERVER")
        self.connection_active = False
        if self.sio and self.is_connected:
            try:
                await self.sio.disconnect()
            except Exception as e:
                logger.warning(f"⚠️ ERROR DURING DISCONNECT: {str(e)}")
        if self.connection_task and not self.connection_task.done():
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
        logger.info("☁️ CLOUD SERVER CONNECTION CLOSED")

# Create an instance of our MCP server
mcp_server = DynamicAdditionServer()

# Global collection to track active websocket connections
active_websockets = set()

# Global dictionary to track client connections by ID
client_connections = {}

# Custom WebSocket handler for the MCP server
async def handle_websocket(websocket):
    # Accept the WebSocket connection with MCP subprotocol
    await websocket.accept(subprotocol="mcp")

    # Generate a unique client ID based on address
    client_id = f"ws_{websocket.client.host}_{id(websocket)}"

    # Track this websocket connection both globally and by ID
    global active_websockets, client_connections
    active_websockets.add(websocket)
    client_connections[client_id] = {"type": "websocket", "connection": websocket}
    connection_count = len(active_websockets)

    logger.info(f"🔌 New WebSocket connection established from {websocket.client.host} (ID: {client_id}, Active: {connection_count})")

    try:
        # Message loop
        while True:
            # Wait for a message from the client
            message = await websocket.receive_text()

            try:
                # Parse the message as JSON
                request = json.loads(message)
                logger.debug(f"📥 Received: {request}")

                # Process the request using our MCP server (include client_id)
                response = await process_mcp_request(mcp_server, request, client_id)

                # Send the response back to the client
                logger.debug(f"📤 Sending: {response}")
                await websocket.send_text(json.dumps(response))

            except json.JSONDecodeError:
                logger.error(f"🚫 Invalid JSON received: {message}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                logger.error(f"🚫 Error processing request: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except Exception as e:
        logger.error(f"🛑 WebSocket error: {e}")
    finally:
        # Remove this connection from all tracking
        active_websockets.discard(websocket)

        # Find and remove from client_connections
        to_remove = []
        for cid, info in client_connections.items():
            if info.get("type") == "websocket" and info.get("connection") is websocket:
                to_remove.append(cid)
        for cid in to_remove:
            client_connections.pop(cid, None)

        connection_count = len(active_websockets)
        logger.info(f"👋 WebSocket connection closed with {websocket.client.host} (Active: {connection_count})")

# Process MCP request and generate response
async def process_mcp_request(server, request, client_id=None):
    """Process an MCP request and return a response

    Args:
        server: The MCP server instance
        request: The request to process
        client_id: Optional ID of the requesting client for tracking
    """

    logger.info(f"🚀 Processing MCP request")

    if "id" not in request:
        return {"error": "Missing request ID"}

    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    # Store client_id in thread-local storage or other request context
    # This allows tools called during this request to know who's calling
    global current_request_client_id
    current_request_client_id = client_id

    # Route the request to the appropriate handler
    try:
        if method == "initialize":
            # Process initialize request
            logger.info(f"🚀 Processing 'initialize' request with params: {params}")
            result = await server.initialize(params)
            logger.info(f"✅ Successfully processed 'initialize' request")
            # Return empty object per MCP protocol spec
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        elif method == "tools/list":
            logger.info(f"🧰 Processing 'tools/list' request")
            # Pass context indicating this call is from the SDK handler (likely a direct request)
            tool_list = await server._get_tools_list(caller_context="process_mcp_request_websocket")
            # Convert Tool objects to dictionaries for JSON serialization
            try:
                # Ensure model_dump is called correctly for each tool
                tools_dict_list = [tool.model_dump() for tool in tool_list]
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": tools_dict_list}
                }
                logger.debug(f"📦 Prepared tools/list response (ID: {req_id}) with {len(tools_dict_list)} tools.")
                return response
            except Exception as e:
                 logger.error(f"💥 Error serializing tool list for JSON-RPC response: {e}")
                 # Return a valid JSON-RPC error response
                 return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": f"Internal server error: Failed to serialize tool list - {e}"}
                 }

        elif method == "prompts/list":
            result = await server._get_prompts_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": result}}
        elif method == "resources/list":
            result = await server._get_resources_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": result}}
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {}) # MCP spec uses 'arguments'
            logger.info(f"🔧 Processing 'tools/call' for tool '{name}' with args: {args}")

            # Log the tool name and arguments for debugging
            logger.debug(f"Tool name: '{name}', Arguments: {json.dumps(args, default=str)}")

            # Execute the tool
            result = await server._execute_tool(name, args, client_id)
            logger.info(f"🎯 Tool '{name}' execution completed with {len(result)} content items")

            # Debug what kind of result we got
            logger.debug(f"Result type: {type(result)}, Items: {len(result)}")
            for i, item in enumerate(result):
                logger.debug(f"  Item {i}: {type(item).__name__}")

            # Format response according to JSON-RPC 2.0 spec for MCP
            try:
                # Manually convert TextContent objects to dictionaries
                contents_list = []
                for content in result:
                    try:
                        if hasattr(content, 'model_dump'):
                            content_data = content.model_dump()
                            contents_list.append(content_data)
                        else:
                            # Handle non-pydantic objects
                            logger.warning(f"⚠️ Non-pydantic content object: {type(content)}")
                            if hasattr(content, 'to_dict'):
                                contents_list.append(content.to_dict())
                            else:
                                # Fallback for simple objects
                                contents_list.append({"type": "text", "text": str(content)})
                    except Exception as e:
                        logger.error(f"❌ Error serializing content result: {e}")
                        # Add simple text content as fallback
                        contents_list.append({"type": "text", "text": str(content)})

                # Construct the response according to JSON-RPC 2.0 and MCP spec
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": contents_list
                    }
                }
                logger.debug(f"📦 Formatted response: {json.dumps(response, default=str)[:200]}...")
                return response
            except Exception as e:
                logger.error(f"💥 Error formatting tool call response: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return {"jsonrpc": "2.0", "id": req_id, "error": f"Error formatting tool call response: {e}"}
        else:
            logger.warning(f"⚠️ Unknown method requested: {method}")
            return {"jsonrpc": "2.0", "id": req_id, "error": f"Unknown method: {method}"}
    except Exception as e:
        import traceback
        logger.error(f"🚫 Error processing request '{method}': {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {"jsonrpc": "2.0", "id": req_id, "error": f"Error processing request: {e}"}

# Set up the Starlette application with a WebSocket route
app = Starlette(
    routes=[
        WebSocketRoute("/mcp", endpoint=handle_websocket),
    ]
)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP WebSocket Server")
    parser.add_argument("--host", default=HOST, help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind the server to")
    parser.add_argument("--cloud-host", default=CLOUD_SERVER_HOST, help="Cloud server host to connect to")
    parser.add_argument("--cloud-port", type=int, default=CLOUD_SERVER_PORT, help="Cloud server port to connect to")
    parser.add_argument("--cloud-namespace", default=CLOUD_SERVICE_NAMESPACE, help="Cloud server Socket.IO namespace")
    parser.add_argument("--email", help="Service email for cloud authentication")
    parser.add_argument("--api-key", help="Service API key for cloud authentication")
    parser.add_argument("--service-name", help="Desired service name")
    parser.add_argument("--no-cloud", action="store_true", help="Disable cloud server connection")
    args = parser.parse_args()

    # Update host and port from command line arguments
    HOST = args.host
    PORT = args.port

    # Update cloud server settings if provided
    if args.cloud_host != CLOUD_SERVER_HOST or args.cloud_port != CLOUD_SERVER_PORT:
        CLOUD_SERVER_HOST = args.cloud_host
        CLOUD_SERVER_PORT = args.cloud_port
        CLOUD_SERVER_URL = f"http://{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"

    # Set up the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize the MCP server
    logger.info(f"{BRIGHT_WHITE}🔧 === CALLING SERVER INITIALIZE FROM MAIN ==={RESET}")
    loop.run_until_complete(mcp_server.initialize())

    # Start the Uvicorn server
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)

    # Create tasks for the server and cloud connection
    server_task = loop.create_task(server.serve())
    cloud_task = None

    try:
        # Start the server
        logger.info(f"🌟 STARTING MCP WEBSOCKET SERVER AT ws://{HOST}:{PORT}/mcp")

        # Connect to cloud server if enabled
        if not args.no_cloud:
            if not args.email or not args.api_key or not args.service_name:
                logger.error("❌ CLOUD SERVER CONNECTION REQUIRES EMAIL, API KEY AND SERVICE NAME")
                logger.error("❌ Use --email and --api-key to specify credentials, --service-name to specify desired service name")
                logger.info("☁️ CLOUD SERVER CONNECTION DISABLED")
            else:
                logger.info(f"☁️ CLOUD SERVER CONNECTION ENABLED: {CLOUD_SERVER_URL}")
                # Create the cloud connection with the provided credentials
                cloud_connection = ServiceClient(
                    server_url=CLOUD_SERVER_URL,
                    namespace=CLOUD_SERVICE_NAMESPACE,
                    email=args.email,
                    api_key=args.api_key,
                    serviceName=args.service_name,
                    mcp_server=mcp_server,
                    port=PORT # Pass the listening port
                )
                cloud_task = loop.create_task(cloud_connection.connect())
        else:
            logger.info("☁️ CLOUD SERVER CONNECTION DISABLED")

        # Run the event loop until the server is interrupted
        loop.run_until_complete(server_task)
    except KeyboardInterrupt:
        logger.info("👋 RECEIVED KEYBOARD INTERRUPT")
    finally:
        # Cancel any pending tasks
        if cloud_task and not cloud_task.done():
            cloud_task.cancel()
        server_task.cancel()

        # Clean up the loop
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()

        logger.info("🧹 CLEANING UP TASKS")
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        remove_pid_file() # Ensure PID file is removed on exit
        logger.info("👋 SERVER SHUTDOWN COMPLETE")
