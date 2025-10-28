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
import socket
import socketio
import argparse
import uuid
import secrets
from utils import clean_filename, format_json_log
from PIDManager import PIDManager
from typing import Any, Callable, Dict, List, Optional, Union
import datetime
import humanize
import subprocess
import traceback
from collections import defaultdict

# Version
SERVER_VERSION = "2.3.4"

from mcp.server import Server

from mcp.client.websocket import websocket_client
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import ValidationError
from websockets.asyncio.client import connect as ws_connect
from websockets.typing import Subprotocol
from mcp.shared.message import SessionMessage
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    NotificationParams,
    Annotations, # Ensure Annotation, ToolErrorAnnotation are NOT imported
    ToolAnnotations as McpToolAnnotations,
    ErrorData
)
from pydantic import ConfigDict
from typing import Any, Dict
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute, Route
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from mcp.shared.exceptions import McpError # <--- ADD THIS IMPORT

# Custom ToolAnnotations class that allows extra fields
class ToolAnnotations(McpToolAnnotations):
    """Custom ToolAnnotations that allows extra fields to avoid Pydantic warnings."""
    model_config = ConfigDict(extra='allow')

# Monkey patch websockets to support larger message sizes globally
import websockets.asyncio.server
original_serve = websockets.asyncio.server.serve
def patched_serve(*args, **kwargs):
    kwargs.setdefault('max_size', 100 * 1024 * 1024)  # 100MB default
    return original_serve(*args, **kwargs)
websockets.asyncio.server.serve = patched_serve

# Import shared state and utilities
from state import (
    logger, # Use the configured logger from state
    HOST, PORT,
    FUNCTIONS_DIR, SERVERS_DIR, is_shutting_down,
    SERVER_REQUEST_TIMEOUT
)

# --- Path Configuration ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
TOOL_CALL_LOG_PATH = os.path.join(LOG_DIR, "tool_call_log.json")
OWNER_LOG_PATH = os.path.join(LOG_DIR, "owner_log.json")

# --- Add dynamic_functions parent to sys.path for imports ---

try:
    # Comment out debug print to avoid interfering with JSON-RPC communication
    # print(f"DEBUG: Python version running server.py: {sys.version}")


    # Get the current timestamp of the dynamic_functions directory
    current_mtime = 0.0
    server_mtime = 0.0

    # Track runtime errors from dynamic functions
    _runtime_errors = {}

    # Get the parent directory of dynamic_functions
    parent_dir = os.path.dirname(FUNCTIONS_DIR)

    # Add the parent directory to sys.path if not already present
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        # Use state.logger if available, otherwise print
        try:
            from state import logger
            logger.info(f"✅ Added '{parent_dir}' to sys.path for dynamic function imports.")
        except ImportError:
            print(f"INFO: Added '{parent_dir}' to sys.path for dynamic function imports.")
except Exception as e:
    print(f"ERROR: Failed to add dynamic_functions parent directory to sys.path: {e}")
# -------------------------------------------------------------

# Import Uvicorn for running the server
import uvicorn


# Cloud server configuration
CLOUD_SERVER_HOST = "ws.projectatlantis.ai"
CLOUD_SERVER_PORT = 3010
CLOUD_SERVER_URL = f"{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"
CLOUD_SERVICE_NAMESPACE = "/service"  # Socket.IO namespace for service-to-service communication
CLOUD_CONNECTION_RETRY_SECONDS = 5  # Initial delay in seconds
CLOUD_CONNECTION_MAX_RETRIES = None  # Maximum number of retries before giving up (None for infinite)
CLOUD_CONNECTION_MAX_BACKOFF_SECONDS = 15  # Maximum delay for exponential backoff



# Import color constants directly from ColoredFormatter
from ColoredFormatter import (
    BOLD, RESET, CYAN, BRIGHT_WHITE, PINK,
    CYAN as CYAN_COLOR, YELLOW, GREY as GREY_COLOR,
    RESET as RESET_COLOR, BOLD as BOLD_COLOR, RED, MAGENTA
)

# Import dynamic management classes
from DynamicFunctionManager import DynamicFunctionManager
from DynamicServerManager import DynamicServerManager

# Import our utility module for dynamic functions
import utils
import atlantis

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
    global is_shutting_down, pid_manager
    if not is_shutting_down:
        is_shutting_down = True
        print("\n🐱 Meow! Graceful shutdown in progress... Press Ctrl+C again to force exit! 🐱")
        print("🧹 Cleaning up resources and closing connections...")
        # Signal the cloud connection to close
        if 'cloud_connection' in globals() and cloud_connection is not None:
            logger.info("☁️ Closing cloud server connection...")
            asyncio.create_task(cloud_connection.disconnect())
            pid_manager.remove_pid_file()
    else:
        print("\n🚨 Forced exit! 🚨")
        # Ensure PID file is removed even on force exit
        pid_manager.remove_pid_file()
        sys.exit(1)

# Register signal handler
signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# Initialize PID Manager variable (will be set after args parsing)
pid_manager = None

# PID check moved to after argument parsing

# --- File Watcher Setup ---

import threading # Added for watcher thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DynamicConfigEventHandler(FileSystemEventHandler):
    """Handles file system events in dynamic_functions and dynamic_servers directories."""
    def __init__(self, mcp_server, loop, watched_function_dirs=None, watched_server_dirs=None):
        self.mcp_server = mcp_server
        self.loop = loop
        self._debounce_timer = None
        self._debounce_interval = 1.0 # seconds
        self._last_triggered_path = None # Store path for _do_reload
        self.watched_function_dirs = watched_function_dirs or []
        self.watched_server_dirs = watched_server_dirs or []

    def _trigger_reload(self, event_path):
        # Check if the change is relevant (Python file in any watched function dir or JSON file in any watched server dir)
        is_function_change = event_path.endswith(".py") and any(
            event_path.startswith(watched_dir + os.sep) or os.path.dirname(event_path) == watched_dir
            for watched_dir in self.watched_function_dirs
        )
        is_server_change = event_path.endswith(".json") and any(
            event_path.startswith(watched_dir + os.sep) or os.path.dirname(event_path) == watched_dir
            for watched_dir in self.watched_server_dirs
        )

        if not is_function_change and not is_server_change:
            # logger.debug(f"Ignoring irrelevant change: {event_path}")
            return

        change_type = "function" if is_function_change else "server configuration"
        logger.info(f"🐍 Change detected in dynamic {change_type}: {os.path.basename(event_path)}. Debouncing...")

        # Store the path that triggered this potential reload
        self._last_triggered_path = event_path

        # Debounce: Cancel existing timer if a new event comes quickly
        if self._debounce_timer:
            self._debounce_timer.cancel()

        # Schedule the actual reload after a short delay
        # Pass the event path to the target function
        self._debounce_timer = threading.Timer(self._debounce_interval, self._do_reload, args=[event_path])
        self._debounce_timer.start()

    def _do_reload(self, event_path: Optional[str] = None):
        # Use the path passed from the timer, or the last stored one as fallback
        path_to_process = event_path or self._last_triggered_path
        if not path_to_process:
            logger.warning("⚠️ _do_reload called without a valid event path.")
            return

        logger.info(f"⏰ Debounce finished for {os.path.basename(path_to_process)}. Processing file change.")

        # Clear the cache - Must run in the main event loop
        async def _process_file_change(file_path: str):
            # --- Invalidate ALL Dynamic Function Runtime Caches ---
            # Check if the change was in the functions or servers directory
            is_function_change = file_path.endswith(".py") and (os.path.dirname(file_path) == FUNCTIONS_DIR or
                                                                file_path.startswith(FUNCTIONS_DIR + os.sep))
            is_server_change = file_path.endswith(".json") and (os.path.dirname(file_path) == SERVERS_DIR or
                                                                file_path.startswith(SERVERS_DIR + os.sep))

            if is_function_change:
                logger.info(f"⚡ File watcher triggering flush of all dynamic function runtime caches due to change in {os.path.basename(file_path)}.")
                try:
                    await self.mcp_server.function_manager.invalidate_all_dynamic_module_cache()
                    # NEW: Also invalidate function mapping cache
                    await self.mcp_server.function_manager.invalidate_function_mapping_cache()
                except Exception as e:
                    logger.error(f"❌ Error invalidating all dynamic modules cache from file watcher: {e}")
            # --- End Runtime Cache Invalidation ---

            # --- Existing Tool List Cache Clearing & Notification ---
            # This still runs regardless of whether it was a function or server config change
            logger.info(f"🧹 Clearing tool list cache on server due to file change in {os.path.basename(file_path)}.")
            #self.mcp_server._cached_tools = None
            self.mcp_server._last_functions_dir_mtime = None # Reset functions mtime to force reload
            self.mcp_server._last_servers_dir_mtime = None   # Reset servers mtime to force reload
            # Notify clients
            if hasattr(self.mcp_server, '_notify_tool_list_changed'):
                try:
                    # Pass placeholder args as the watcher doesn't know specifics
                    await self.mcp_server._notify_tool_list_changed(change_type="unknown", tool_name="unknown/file_watcher")
                except Exception as e:
                    # Log the specific error from the notification call
                    logger.error(f"❌ Failed to notify clients after file change: {e}")
            else:
                logger.warning("⚠️ Server object lacks _notify_tool_list_changed method.")

            # Report tools to cloud if connected
            if self.mcp_server.cloud_client and self.mcp_server.cloud_client.is_connected:
                try:
                    logger.info("📊 Reporting updated tool list to cloud after file change...")
                    await self.mcp_server.cloud_client._report_tools_to_console()
                except Exception as e:
                    logger.error(f"❌ Failed to report tools to cloud after file change: {e}")
            # --- End Tool List Cache Clearing ---

        # Schedule the coroutine to run in the event loop from this thread
        if self.loop.is_running():
             asyncio.run_coroutine_threadsafe(_process_file_change(path_to_process), self.loop)
        else:
             logger.warning("⚠️ Event loop not running, cannot process file change for {os.path.basename(path_to_process)}.")

    def on_modified(self, event):
        if not event.is_directory:
            self._trigger_reload(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._trigger_reload(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._trigger_reload(event.src_path)
        # Also trigger on directory deletion (removed app directories)
        elif event.src_path.startswith(FUNCTIONS_DIR + os.sep):
            logger.info(f"📁 Directory deleted: {event.src_path}. Triggering reload...")
            self._trigger_reload(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            # Trigger reload for both source and destination paths
            self._trigger_reload(event.src_path)
            self._trigger_reload(event.dest_path)

# --- End File Watcher Setup ---

# Create an MCP server with proper MCP protocol handling
class DynamicAdditionServer(Server):
    """MCP server that provides an addition tool and supports dynamic function registration"""

    def __init__(self):
        super().__init__("Dynamic Function Server")
        self.websocket_connections = set()  # Store active WebSocket connections
        self.service_connections = {} # Store active Service connections (e.g. cloud)
        self.awaitable_requests: Dict[str, asyncio.Future] = {} # For tracking awaitable commands
        self.awaitable_request_timeout: float = SERVER_REQUEST_TIMEOUT # Timeout for these commands

        # TODO: Add prompts and resources
        self._cached_tools: Optional[List[Tool]] = None # Cache for tool list
        self._last_functions_dir_mtime: float = 0.0 # Timestamp for cache invalidation
        self._last_servers_dir_mtime: float = 0.0 # Timestamp for dynamic servers cache invalidation
        self._last_active_server_keys: Optional[set] = None # Store active server keys for cache invalidation
        self._server_configs: Dict[str, dict] = {} # Store server configurations

        # Track function visibility overrides (can show/hide any function)
        self._temporarily_visible_functions: set = set()
        self._temporarily_hidden_functions: set = set()

        # Store cloud client reference for tool reporting
        self.cloud_client: Optional['ServiceClient'] = None

        # Initialize the dynamic function and server managers
        self.function_manager = DynamicFunctionManager(FUNCTIONS_DIR)
        self.server_manager = DynamicServerManager(SERVERS_DIR)

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
        logger.info(f"{CYAN}🔧 Server initialize method completed.{RESET}")
        # Return required InitializeResult fields with proper server capabilities
        return {
            "protocolVersion": params.get("protocolVersion"),
            "capabilities": {
                "tools": {},
                "prompts": {},
                "resources": {},
                "logging": {}
            },
            "serverInfo": {"name": self.name, "version": SERVER_VERSION}
        }

    async def send_awaitable_client_command(self,
                                      client_id_for_routing: str,
                                      request_id: str, # Original MCP request ID for client context
                                      command: str, # The command string for the client
                                      command_data: Optional[Any] = None, # Optional data for the command
                                      seq_num: Optional[int] = None, # Sequence number for client-side ordering
                                      entry_point_name: Optional[str] = None # Entry point name for logging
                                      ) -> Any:
        """Sends a command to a specific client and waits for a response with a correlation ID.

        Args:
            client_id_for_routing: The ID of the client to send the command to.
            request_id: The original MCP request ID, for client-side context.
            command: The command identifier string.
            command_data: Optional data payload for the command.
            seq_num: Optional sequence number for client-side ordering.
            entry_point_name: Optional name of the entry point function for logging.

        Returns:
            The result from the client's command execution.

        Raises:
            McpError: If the client response times out or the client returns an error.
            Various other exceptions if sending or future handling fails.
        """
        correlation_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self.awaitable_requests[correlation_id] = future

        logger.info(f"⏳ Preparing awaitable command '{command}' for client {client_id_for_routing} (correlationId: {correlation_id}, MCP_reqId: {request_id})")

        payload = {
            "jsonrpc": "2.0",
            "method": "atlantis/executeAwaitableCommand", # Distinct method for client to identify
            "params": {
                "correlationId": correlation_id,
                "command": command,
                "data": command_data,
                "requestId": request_id # Original request_id for client context
            }
        }
        payload_json = json.dumps(payload)

        global client_connections # Access the global client connection tracking

        if not client_id_for_routing or client_id_for_routing not in client_connections:
            self.awaitable_requests.pop(correlation_id, None) # Clean up future
            logger.error(f"❌ Cannot send awaitable command '{command}': Client ID '{client_id_for_routing}' not found or invalid.")
            error_data = ErrorData(
                code=-32602,
                message=f"Client ID '{client_id_for_routing}' not found for awaitable command."
            )
            raise McpError(error_data)

        client_info = client_connections[client_id_for_routing]
        client_type = client_info.get("type")
        connection = client_info.get("connection")

        try:
            if client_type == "websocket" and connection:
                await connection.send_text(payload_json)
                logger.info(f"✉️ Sent awaitable command '{command}' to WebSocket client {client_id_for_routing} (correlationId: {correlation_id})")
            elif client_type == "cloud" and connection and hasattr(connection, 'is_connected') and connection.is_connected:
                # Cloud clients get it wrapped in a 'notifications/message' structure, sent via 'mcp_notification' event
                cloud_notification_params = {
                    "messageType": "command",
                    "requestId": request_id,       # Original MCP request_id for cloud client context
                    "correlationId": correlation_id, # The ID for awaiting the response
                    "command": command,            # The actual command string
                    "data": command_data           # Associated data for the command
                }

                # Add seqNum if provided
                if seq_num is not None:
                    cloud_notification_params["seqNum"] = seq_num
                else:
                    # Log an error if seq_num is None - this shouldn't happen
                    logger.error(f"❌ Missing sequence number in send_awaitable_client_command for command '{command}', client {client_id_for_routing}, request {request_id}")
                    # We continue without seqNum for backward compatibility, but this is an error condition

                # Add entryPoint if provided
                if entry_point_name is not None:
                    cloud_notification_params["entryPoint"] = entry_point_name
                cloud_wrapper_payload = {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": cloud_notification_params
                }
                logger.info(f"☁️ About to send cloud payload with params:\n{format_json_log(cloud_notification_params)}")
                await connection.send_message('mcp_notification', cloud_wrapper_payload)
                logger.info(f"☁️ Sent awaitable command '{command}' (as flat notifications/message) to cloud client {client_id_for_routing} (correlationId: {correlation_id}) via 'mcp_notification' event")
            else:
                self.awaitable_requests.pop(correlation_id, None) # Clean up future
                logger.error(f"❌ Cannot send awaitable command '{command}': Client '{client_id_for_routing}' has no valid/active connection.")
                error_data = ErrorData(
                    code=-32602,
                    message=f"Client '{client_id_for_routing}' has no active connection for awaitable command."
                )
                raise McpError(error_data)

            # Wait for the future to be resolved
            logger.debug(f"⏳ Waiting for response for command '{command}' (correlationId: {correlation_id}) timeout: {self.awaitable_request_timeout}s")
            result = await asyncio.wait_for(future, timeout=self.awaitable_request_timeout)
            logger.info(f"✅ Received response for awaitable command '{command}' (correlationId: {correlation_id})")
            return result
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout waiting for response for command '{command}' (correlationId {correlation_id}) from client {client_id_for_routing}")
            self.awaitable_requests.pop(correlation_id, None)
            error_data = ErrorData(
                code=-32000,
                message=f"Timeout waiting for client response for command: {command} (correlationId: {correlation_id})"
            )
            raise McpError(error_data)
        except Exception as e:
            logger.error(f"❌ Error during awaitable command '{command}' processing (correlationId {correlation_id}): {type(e).__name__} - {e}")
            self.awaitable_requests.pop(correlation_id, None) # Ensure cleanup
            if isinstance(e, McpError):
                # Enhance McpError message to include command text
                # Access the error data from the McpError
                original_error = e.error if hasattr(e, 'error') else None
                if original_error:
                    error_msg = original_error.message if hasattr(original_error, 'message') else str(e)
                    error_code = original_error.code if hasattr(original_error, 'code') else -32000
                else:
                    error_msg = str(e)
                    error_code = -32000

                enhanced_message = f"Command '{command}' failed: {error_msg}"
                enhanced_error_data = ErrorData(code=error_code, message=enhanced_message)
                raise McpError(enhanced_error_data) from e
            else:
                # For other exceptions, wrap in a new exception with command context
                enhanced_message = f"Command '{command}' failed: {str(e)}"
                raise Exception(enhanced_message) from e
        finally:
            # Final cleanup, though most paths should handle it.
            if correlation_id in self.awaitable_requests:
                 logger.warning(f"🧹 Final cleanup: Future for {correlation_id} was still in awaitable_requests.")
                 self.awaitable_requests.pop(correlation_id, None)

    async def _create_tools_from_app_mappings(self) -> list[Tool]:
        """Create tools from app-specific function mappings, showing all function variants"""
        tools_list = []

        function_mapping_by_app = self.function_manager._function_file_mapping_by_app

        # Count total functions across all apps
        total_functions = sum(len(app_mapping) for app_mapping in function_mapping_by_app.values())
        logger.info(f"📝 FOUND {total_functions} FUNCTIONS FROM APP MAPPINGS")

        # Track processed files to avoid redundant validation
        processed_files = {}  # file_path -> functions_info cache

        # For each app and function in the app-specific mappings, create Tool entries
        for app_name, app_mapping in function_mapping_by_app.items():
            for func_name, file_path in app_mapping.items():
                try:
                    # Skip if this seems to be a utility file
                    if func_name.startswith('_') or func_name == '__init__' or func_name == '__pycache__':
                        continue

                    # Check if we've already processed this file
                    if file_path in processed_files:
                        # Already processed this file, skip to avoid duplicates
                        continue
                    else:
                        # Load and validate the function code directly from the file
                        full_file_path = os.path.join(FUNCTIONS_DIR, file_path)
                        try:
                            with open(full_file_path, 'r', encoding='utf-8') as f:
                                code = f.read()

                            # Validate function syntax without actually loading it
                            is_valid, error_message, functions_info = self.function_manager._code_validate_syntax(code)
                            # Cache the result (even if None/invalid)
                            processed_files[file_path] = functions_info if is_valid else None
                        except Exception as e:
                            logger.warning(f"⚠️ Error loading file {file_path}: {str(e)}")
                            processed_files[file_path] = None
                            continue

                    # Handle multiple functions per file
                    if is_valid and functions_info:
                        # Create one tool per function in the file
                        for func_info in functions_info:
                            tool_name = func_info.get('name', func_name)

                            # CRITICAL: Only process functions that are in the mapping for this app
                            # This prevents duplicates that were already removed by DynamicFunctionManager
                            if tool_name not in app_mapping:
                                logger.debug(f"🙈 Skipping function not in mapping for app '{app_name}': {tool_name} from {file_path}")
                                continue

                            # NEW OPT-IN VISIBILITY: Check if function has @visible, @public, @protected, or @tick decorator or is internal
                            decorators_from_info = func_info.get("decorators", [])
                            is_internal = tool_name.startswith('_function') or tool_name.startswith('_server') or tool_name.startswith('_admin')
                            is_visible = ("visible" in decorators_from_info or "public" in decorators_from_info or "protected" in decorators_from_info or "tick" in decorators_from_info) if decorators_from_info else False
                            is_hidden = "hidden" in decorators_from_info if decorators_from_info else False

                            # Skip if explicitly hidden OR if not visible and not internal
                            if is_hidden or (not is_visible and not is_internal):
                                logger.debug(f"🙈 Skipping non-visible function in tool creation: {tool_name} from {file_path}")
                                continue

                            tool_description = func_info.get('description', f"Dynamic function '{tool_name}'")
                            tool_input_schema = func_info.get('inputSchema', {"type": "object", "properties": {}})
                            tool_annotations = {}

                            tool_annotations["type"] = "function"
                            tool_annotations["validationStatus"] = "VALID"
                            tool_annotations["sourceFile"] = file_path

                            # Add decorator info if present (decorators_from_info already retrieved above)
                            if decorators_from_info:
                                tool_annotations["decorators"] = decorators_from_info

                            # Use app name from decorator or from app mapping
                            app_name_from_info = func_info.get("app_name")
                            if app_name_from_info is not None:
                                actual_app_name = app_name_from_info
                                app_source = "decorator"
                                #logger.info(f"🎯 AUTO-ASSIGNED APP: {func_name} -> {actual_app_name} (from @app decorator)")
                            else:
                                # app_name from mapping is in slash notation, keep it for internal use
                                actual_app_name = app_name
                                app_source = "directory" if app_name else "NA"
                                #logger.info(f"🎯 AUTO-ASSIGNED APP: {func_name} -> {actual_app_name} (from app mapping)")

                            # Convert slash path to dot notation for MCP output
                            tool_annotations["app_name"] = self.function_manager._path_to_app_name(actual_app_name) if actual_app_name else None
                            tool_annotations["app_source"] = app_source

                            # Check visibility overrides first (these take precedence over decorators)
                            if tool_name in self._temporarily_hidden_functions:
                                logger.info(f"{PINK}🙈 Skipping temporarily hidden function: {tool_name} (app: {actual_app_name}){RESET}")
                                continue
                            elif tool_name in self._temporarily_visible_functions:
                                logger.info(f"{PINK}👁️ Showing temporarily visible function: {tool_name} (app: {actual_app_name}){RESET}")
                                tool_annotations["temporarilyVisible"] = True
                            elif decorators_from_info and "hidden" in decorators_from_info:
                                # Skip this function if it has the @hidden decorator (and no override)
                                # Note: This should never happen as hidden functions are filtered earlier
                                logger.info(f"{PINK}🙈 Skipping hidden function: {tool_name} (app: {actual_app_name}){RESET}")
                                continue

                            # Add location_name to annotations if present in function_info
                            location_name_from_info = func_info.get("location_name")
                            if location_name_from_info is not None:
                                tool_annotations["location_name"] = location_name_from_info

                            # Add protection_name to annotations if present in function_info
                            protection_name_from_info = func_info.get("protection_name")
                            if protection_name_from_info is not None:
                                tool_annotations["protection_name"] = protection_name_from_info

                            # Add is_index flag to annotations if present in function_info
                            is_index_from_info = func_info.get("is_index")
                            if is_index_from_info:
                                tool_annotations["is_index"] = True

                            # Add pricing information to annotations if present in function_info
                            price_per_call = func_info.get("price_per_call")
                            price_per_sec = func_info.get("price_per_sec")
                            if price_per_call is not None and price_per_sec is not None:
                                tool_annotations["price_per_call"] = price_per_call
                                tool_annotations["price_per_sec"] = price_per_sec

                            # Add runtime error message if present in cache
                            if tool_name in _runtime_errors:
                                tool_annotations["runtimeError"] = _runtime_errors[tool_name]

                            # Add server config load error if present in cache
                            if tool_name in self.server_manager._server_load_errors:
                                tool_annotations["loadError"] = self.server_manager._server_load_errors[tool_name]

                            # Add common annotations
                            try:
                                tool_annotations["lastModified"] = datetime.datetime.fromtimestamp(
                                    os.path.getmtime(full_file_path),
                                    tz=datetime.timezone.utc
                                ).isoformat()
                                #logger.debug(f"🔍 SET lastModified for {tool_name}: {tool_annotations['lastModified']}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to get lastModified for {tool_name}: {e}")
                                pass

                            # Create and add the tool object
                            custom_annotations = ToolAnnotations(**tool_annotations)
                            tool_obj = Tool(
                                name=tool_name,
                                description=tool_description,
                                inputSchema=tool_input_schema,
                                annotations=custom_annotations
                            )

                            # No need for duplicate detection here - only processing functions in the mapping
                            # Duplicates were already detected and removed by DynamicFunctionManager
                            tools_list.append(tool_obj)
                            #logger.debug(f"📝 Added dynamic tool: {tool_name} (app: {actual_app_name}), valid: {is_valid}")

                except Exception as e:
                    logger.warning(f"⚠️ Error processing function {func_name} from {file_path}: {str(e)}")
                    continue

        return tools_list

    async def _get_tools_list(self, caller_context: str = "unknown") -> list[Tool]:
        """Core logic to return a list of available tools"""
        logger.info(f"{BRIGHT_WHITE}📋 === GETTING TOOL LIST (Called by: {caller_context}) ==={RESET}")

        # --- Caching Logic ---
        try:
            # Ensure the directory exists before checking mtime
            if not os.path.exists(FUNCTIONS_DIR):
                 os.makedirs(FUNCTIONS_DIR, exist_ok=True)
                 current_mtime = os.path.getmtime(FUNCTIONS_DIR)
            else:
                 current_mtime = os.path.getmtime(FUNCTIONS_DIR)

            # Ensure the dynamic servers directory exists and get its mtime
            if not os.path.exists(SERVERS_DIR):
                os.makedirs(SERVERS_DIR, exist_ok=True)
                server_mtime = os.path.getmtime(SERVERS_DIR)
            else:
                server_mtime = os.path.getmtime(SERVERS_DIR)

            dirs_changed = current_mtime != self._last_functions_dir_mtime or server_mtime != self._last_servers_dir_mtime
            # First identify running vs. non-running servers            # Get running servers using proper accessor
            running_servers = await self.server_manager.get_running_servers()
            current_active_keys = set(running_servers)
            servers_changed = current_active_keys != self._last_active_server_keys

            # Use current_active_keys here too
            if not dirs_changed and not servers_changed and self._cached_tools is not None:
                logger.info(f"⚡️ USING CACHED TOOL LIST (Dirs unchanged - func mtime: {current_mtime}, server mtime: {server_mtime}; Active Servers unchanged: {self._last_active_server_keys})")
                return list(self._cached_tools)
            # Log reason for regeneration
            reason = []
            if dirs_changed:
                reason.append(f"Dirs changed (func mtime: {current_mtime} vs {self._last_functions_dir_mtime}, server mtime: {server_mtime} vs {self._last_servers_dir_mtime})")
            if servers_changed:
                reason.append(f"Active servers changed ({running_servers} vs {self._last_active_server_keys})")
            if self._cached_tools is None:
                 reason.append("Cache empty")
            logger.info(f"🔄 Cache invalid ({', '.join(reason)}). REGENERATING TOOL LIST")

        except FileNotFoundError as e:
             logger.error(f"❌ Error checking FUNCTIONS_DIR mtime: {e}. Proceeding without cache.")
             current_mtime = time.time() # Use current time to force regeneration

                # Start with our built-in tools
        tools_list = [
            Tool(
                name="_function_set",
                description="Sets the content of a dynamic Python function. Use 'app' parameter to target specific functions when multiple exist with the same name.", # Updated description
                inputSchema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": "Optional: The app name to target a specific function when multiple exist with the same name"},
                        "code": {"type": "string", "description": "The Python source code containing a single function definition."}
                    },
                    "required": ["code"] # Only code is required now
                },
                annotations=ToolAnnotations(title="_function_set")
            ),
            Tool( # Add definition for get_tool_code
                name="_function_get",
                description="Gets the Python source code for a dynamic function. Use 'app' parameter to target specific functions when multiple exist with the same name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": "Optional: The app name to target a specific function when multiple exist with the same name"},
                        "name": {"type": "string", "description": "The name of the function to get code for"}
                    },
                    "required": ["name"]
                },
                annotations=ToolAnnotations(title="_function_get")
            ),
            Tool( # Add definition for remove_dynamic_tool
                name="_function_remove",
                description="Removes a dynamic Python function. Use 'app' parameter to target specific functions when multiple exist with the same name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": "Optional: The app name to target a specific function when multiple exist with the same name"},
                        "name": {"type": "string", "description": "The name of the function to remove"}
                    },
                    "required": ["name"]
                },
                annotations=ToolAnnotations(title="_function_remove")
            ),
            Tool( # Add definition for add_placeholder_function
                name="_function_add",
                description="Adds a new, empty placeholder Python function with the given name. Use 'app' parameter to create function in specific app directory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": "Optional: The app name to create the function in a specific app directory"},
                        "name": {"type": "string", "description": "The name to register the new placeholder function under."}
                    },
                    "required": ["name"]
                },
                annotations=ToolAnnotations(title="_function_add")
            ),
            Tool(
                name="_function_history",
                description="Gets the tool call history for a specific function. Requires both app and function name since multiple apps can have functions with the same name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": "The app name containing the function"},
                        "name": {"type": "string", "description": "The function name to get history for"}
                    },
                    "required": ["app", "name"]
                },
                annotations=ToolAnnotations(title="_function_history")
            ),
            Tool(
                name="_function_log",
                description="Gets the tool owner log. Use 'app' parameter to filter log for specific app functions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": "Optional: The app name to filter log for functions from a specific app"}
                    },
                    "required": []
                },
                annotations=ToolAnnotations(title="_function_log")
            ),
            Tool(
                name="_server_get",
                description="Gets the configuration JSON for a server",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                },
                annotations=ToolAnnotations(title="_server_get")
            ),
            Tool(
                name="_server_add",
                description="Adds a new MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                },
                annotations=ToolAnnotations(title="_server_add")
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
                },
                annotations=ToolAnnotations(title="_server_remove")
            ),
            Tool(
                name="_server_set",
                description="Sets (adds or updates) an MCP server configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "description": "Server config JSON, must contain 'mcpServers' key with the server name as its child.",
                            "properties": {
                                "mcpServers": {
                                    "type": "object",
                                    "description": "A dictionary where the key is the server name.",
                                    # We don't strictly enforce the structure *within* the named server config here,
                                    # relying on server_manager's validation, but describe the expectation.
                                    "additionalProperties": {
                                         "type": "object",
                                         "properties": {
                                            "command": {"type": "string"},
                                            "args": {"type": "array", "items": {"type": "string"}},
                                            "env": {"type": "object"},
                                            "cwd": {"type": "string"}
                                         },
                                         "required": ["command"]
                                    }
                                }
                            },
                            "required": ["mcpServers"]
                        }
                    },
                    "required": ["config"] # Only config is required top-level
                },
                annotations=ToolAnnotations(title="_server_set")
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
                },
                annotations=ToolAnnotations(title="_server_validate")
            ),
            Tool(
                name="_server_start",
                description="Starts a managed MCP server background task using its configuration name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the server config to start."}
                    },
                    "required": ["name"]
                },
                annotations=ToolAnnotations(title="_server_start")
            ),
            Tool(
                name="_server_stop",
                description="Stops a managed MCP server background task by its configuration name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the server config to stop."}
                    },
                    "required": ["name"]
                },
                annotations=ToolAnnotations(title="_server_stop")
            ),
            Tool(
                name="_server_get_tools",
                description="Gets the list of tools from a specific *running* managed server.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                },
                annotations=ToolAnnotations(title="_server_get_tools")
            ),
            Tool( # Add definition for _admin_restart
                name="_admin_restart",
                description="Restart the server by terminating the process. Assumes a wrapper script will restart the server automatically.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                annotations=ToolAnnotations(title="_admin_restart")
            ),
            Tool( # Add definition for _admin_pip_install
                name="_admin_pip_install",
                description="Install Python packages using pip. Only available to server owner.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "package": {"type": "string", "description": "The package name to install (e.g., 'requests', 'numpy==1.21.0')"}
                    },
                    "required": ["package"]
                },
                annotations=ToolAnnotations(title="_admin_pip_install")
            ),
            Tool( # Add definition for _admin_click
                name="_admin_click",
                description="Handles click events from the client UI. Only available to server owner.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The click key to identify which callback to invoke"}
                    },
                    "required": ["key"]
                },
                annotations=ToolAnnotations(title="_admin_click")
            ),
            Tool( # Add definition for _admin_upload
                name="_admin_upload",
                description="Handles upload calls from the client UI. Only available to server owner.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The upload key/identifier"},
                        "filename": {"type": "string", "description": "The name of the uploaded file"},
                        "filetype": {"type": "string", "description": "The file extension/type e.g. mp3"},
                        "base64Content": {"type": "string", "description": "The base64 encoded file data"}
                    },
                    "required": ["key", "filename", "filetype", "base64Content"]
                },
                annotations=ToolAnnotations(title="_admin_upload")
            ),
            Tool( # Add definition for _admin_app_create
                name="_admin_app_create",
                description="Create a new app directory with main.py containing empty index() function. Only available to server owner.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "appName": {"type": "string", "description": "The name of the app directory to create"}
                    },
                    "required": ["appName"]
                },
                annotations=ToolAnnotations(title="_admin_app_create")
            ),
            Tool( # Add definition for _admin_git_update
                name="_admin_git_update",
                description="Update the server code by running git fetch and git merge. Only available to server owner.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "remote": {"type": "string", "description": "The remote to fetch from (defaults to 'origin')", "default": "origin"},
                        "branch": {"type": "string", "description": "The branch to merge (defaults to 'main')", "default": "main"}
                    },
                    "required": []
                },
                annotations=ToolAnnotations(title="_admin_git_update")
            ),

        ]
        # Log the static tools being included
        static_tool_names = [tool.name for tool in tools_list]
        logger.info(f"🔩 INCLUDING {len(static_tool_names)} STATIC TOOLS: {', '.join(static_tool_names)}")

        # Scan the FUNCTIONS_DIR directory for dynamic functions
        try:
            # Ensure the functions directory exists
            os.makedirs(FUNCTIONS_DIR, exist_ok=True)

                        # Use the function mapping to get all discovered functions
            await self.function_manager._build_function_file_mapping()
            # Get all functions from app-specific mappings to show all variants
            dynamic_tools = await self._create_tools_from_app_mappings()
            tools_list.extend(dynamic_tools)
        except Exception as e:
            logger.error(f"❌ Error scanning for dynamic functions: {str(e)}")
            # Continue with just the built-in tools

        # Function-to-file mapping was already built at the beginning of this method

        # --- DEBUG: Log current server_tasks state before processing servers ---
        # Access server_tasks via module namespace
        logger.debug(f"🔧 _get_tools_list: Current server_tasks state:\n{format_json_log(self.server_manager.server_tasks)}")

        # Scan dynamic servers
        servers_found = []
        self._server_configs = {} # Reset server configs
        server_statuses = {}
        try:
            # server_list now returns List[TextContent]
            # Use server_manager.server_list if it wasn't imported specifically
            server_list_results = await self.server_manager.server_list() # Using server_manager instance
            logger.info(f"📝 FOUND {len(server_list_results)} MCP server configs")
            for server_name in server_list_results:
                # server_list_results is now a list of strings, not TextContent objects
                # so we can use the server_name directly
                servers_found.append(server_name)
                # First identify running vs. non-running servers using proper accessor
                is_running = await self.server_manager.is_server_running(server_name)

                # Check if server is in failed state in server_tasks
                task_info = self.server_manager.server_tasks.get(server_name, {})
                if task_info and task_info.get('status') == 'failed':
                    status = "failed"  # Mark explicitly as failed
                else:
                    status = "running" if is_running else "stopped" # Normal running/stopped state

                # --- AUTO-START LOGIC --- #
                '''
                if status == "stopped":
                    logger.info(f"⚙️ Server '{server_name}' is stopped. Attempting auto-start during tool list generation...")
                    try:
                        # Call server_start logic internally
                        start_result = await self.server_manager.server_start({'name': server_name}, self)
                        logger.info(f"✅ Auto-start initiated for server '{server_name}'. Result: {start_result}")
                        # Re-check status *after* attempting start
                        is_running = await self.server_manager.is_server_running(server_name)
                        status = "running" if is_running else "stopped"
                        if status == "stopped":
                             logger.warning(f"⚠️ Auto-start attempt for '{server_name}' did not result in an active task. Status remains 'stopped'.")
                        else:
                             logger.info(f"👍 Server '{server_name}' successfully auto-started. Status is now 'running'.")
                    except Exception as start_err:
                        logger.error(f"❌ Failed to auto-start server '{server_name}' during tool list generation: {start_err}", exc_info=False) # Less noisy log
                        status = "stopped"
                # --- END AUTO-START LOGIC --- #
                '''
                server_statuses[server_name] = status # Store final status

                # Always try to get config and add server entry (even if stopped)
                try:
                    # Get server config
                    config = await self.server_manager.server_get(server_name)
                    self._server_configs[server_name] = config # Populate instance variable

                    # Create annotations object
                    annotations = {}
                    annotations["type"] = "server" # Mark this tool entry as representing a server config
                    annotations["serverConfig"] = config or {}
                    annotations["runningStatus"] = status # Add status from outer loop

                    # Try to get file modified time and store source file
                    try:
                        server_file = os.path.join(SERVERS_DIR, f"{server_name}.json")
                        mtime = os.path.getmtime(server_file)
                        annotations["lastModified"] = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).isoformat()
                        # Store relative path for display
                        annotations["sourceFile"] = f"{server_name}.json"
                    except Exception as me:
                        logger.warning(f"⚠️ Could not get mtime for server '{server_name}': {me}")

                    # Define task_info FIRST, outside any status checks
                    task_info = self.server_manager.server_tasks.get(server_name)

                    # Better check for running status - don't rely on 'status' field which might not be set
                    # Instead check if task exists, is not done, and session exists
                    is_running = (
                        task_info is not None and
                        'task' in task_info and
                        not task_info['task'].done() and
                        'session' in task_info and
                        task_info['session'] is not None
                    )
                    status = "running" if is_running else "stopped"

                    if is_running:
                        logger.debug(f"🔄 Server '{server_name}' is running (task active and session present)")
                    else:
                        logger.debug(f"⏹️ Server '{server_name}' is not running")
                        if task_info:
                            logger.debug(f"📊 Task info: task present={('task' in task_info)}, session present={('session' in task_info and task_info['session'] is not None)}")
                            if 'task' in task_info and task_info['task'].done():
                                logger.debug(f"⚠️ Task for '{server_name}' is marked as done")

                    # Update annotations with final status
                    annotations["runningStatus"] = status

                    # Get timestamp if server is running
                    if status == "running":
                        # Try to get started_at from task_info
                        start_time = task_info.get('started_at')

                        # If not found, and we know it's running, set it now with current time
                        if not start_time or not isinstance(start_time, datetime.datetime):
                            logger.debug(f"🕒 Setting started_at timestamp for '{server_name}' to current time")
                            start_time = datetime.datetime.now(datetime.timezone.utc)
                            # Save it in server_tasks for future use
                            if task_info:
                                task_info['started_at'] = start_time

                        # Add timestamp to annotations
                        logger.debug(f"🕒 Using started_at timestamp for '{server_name}': {start_time}")
                        annotations["started_at"] = start_time.isoformat()
                    else:
                        logger.debug(f"📚 No started_at timestamp set for non-running server '{server_name}'")

                        # If this is a stoppped server with a start time in the past, include it anyway
                        # This lets us preserve timestamps for servers that were running but are now stopped
                        past_time = task_info.get('started_at') if task_info else None
                        if past_time and isinstance(past_time, datetime.datetime):
                            logger.debug(f"📚 Found historical started_at timestamp for '{server_name}': {past_time}")
                            annotations["started_at"] = past_time.isoformat()

                        # Always check ALL error sources thoroughly
                    error_msg = None
                    task_info = self.server_manager.server_tasks.get(server_name, {})

                    # STEP 1: Log ALL possible error sources
                    logger.info(f"🔍 SERVER STATUS CHECK FOR '{server_name}':")
                    logger.info(f"  - Current status: {status}")
                    logger.info(f"  - Task info available: {task_info is not None}")
                    logger.info(f"  - Has error in task_info: {'error' in task_info if task_info else False}")
                    logger.info(f"  - Error in _server_load_errors: {server_name in self.server_manager._server_load_errors}")

                    # STEP 2: Always show the full error contents from both sources
                    if task_info and 'error' in task_info:
                        task_error = task_info['error']
                        logger.info(f"  - TASK ERROR: {task_error}")

                    if server_name in self.server_manager._server_load_errors:
                        load_error = self.server_manager._server_load_errors[server_name]
                        logger.info(f"  - LOAD ERROR: {load_error}")

                    # STEP 3: Check for failures
                    # First from failed task status
                    if status == 'failed' or (task_info and task_info.get('status') == 'failed'):
                        logger.info(f"🚨 Server '{server_name}' has FAILED status")
                        # Update status to reflect the failure
                        status = "failed"
                        annotations["runningStatus"] = "failed"

                        # Get error from task_info first (most accurate)
                        if task_info and 'error' in task_info:
                            error_msg = task_info['error']
                            logger.info(f"🚨 Using error from task_info: {error_msg}")
                        # Fall back to _server_load_errors
                        elif server_name in self.server_manager._server_load_errors:
                            error_msg = self.server_manager._server_load_errors[server_name]
                            logger.info(f"🚨 Using error from _server_load_errors: {error_msg}")
                        # Last resort generic error
                        else:
                            error_msg = "Server failed to initialize (unknown error)"
                            logger.info(f"⚠️ Using generic error: {error_msg}")

                    # Even for non-failed servers, check for errors
                    elif server_name in self.server_manager._server_load_errors:
                        error_msg = self.server_manager._server_load_errors[server_name]
                        logger.info(f"🚨 Found error for '{server_name}' in load errors: {error_msg}")
                        # If we found an error but status isn't failed, update it
                        status = "failed"
                        annotations["runningStatus"] = "failed"
                    elif task_info and 'error' in task_info:
                        error_msg = task_info['error']
                        logger.info(f"🚨 Found error for '{server_name}' in task info: {error_msg}")
                        # If we found an error but status isn't failed, update it
                        status = "failed"
                        annotations["runningStatus"] = "failed"

                    # Add error info to annotations if found
                    if error_msg:
                        logger.info(f"📣 Final error for '{server_name}': {error_msg}")
                        # Add to both error fields for maximum visibility
                        annotations["loadError"] = error_msg
                        annotations["error"] = error_msg

                    # Simplified logging that only uses defined variables
                    if task_info:
                        logger.debug(f"🔧 _get_tools_list: Preparing Tool for '{server_name}': task_info exists={task_info is not None}, status='{status}'")
                    else:
                        logger.warning(f"🔧 _get_tools_list: MCP server '{server_name}' installed but not yet started")

                    # Create tool with correct parameters
                    description = f"MCP server: {server_name}"
                    # do not append error to the description

                    # Create custom ToolAnnotations object to allow extra fields
                    custom_annotations = ToolAnnotations(**annotations)
                    server_tool = Tool(
                        name=server_name,
                        description=description,
                        inputSchema={"type": "object"}, # Use inputSchema (camelCase), not input_schema
                        annotations=custom_annotations # Use annotations object that contains started_at if applicable
                    )

                    # Check for duplicates before adding (server names are unique)
                    existing_tool_names = {tool.name.lower() for tool in tools_list}
                    if server_name.lower() not in existing_tool_names:
                        tools_list.append(server_tool)
                        logger.debug(f"📝 Added dynamic server config entry: {server_name} (Status: {status})")
                    else:
                        logger.warning(f"⚠️ Skipping duplicate server config: {server_name} - already exists")
                except Exception as se:
                    logger.warning(f"⚠️ Error processing MCP server config '{server_name}': {se}")
                    # Create custom ToolAnnotations object to allow extra fields
                    error_annotations = ToolAnnotations(
                        validationStatus="ERROR_LOADING_SERVER",
                        runningStatus=status # Still show status even if config load failed
                    )
                    # Check for duplicates before adding (server names are unique)
                    existing_tool_names = {tool.name.lower() for tool in tools_list}
                    if server_name.lower() not in existing_tool_names:
                        tools_list.append(Tool(
                            name=server_name,
                            description=f"Error loading MCP server config: {se}",
                            inputSchema={"type": "object"},
                            annotations=error_annotations
                        ))
                    else:
                        logger.warning(f"⚠️ Skipping duplicate error server config: {server_name} - already exists")
        except Exception as ee:
            logger.error(f"❌ Error scanning for dynamic servers: {ee}")

        # --- Fetch Tools from RUNNING Servers Concurrently --- #
        running_servers = await self.server_manager.get_running_servers()
        if running_servers:
            logger.info(f"📡 Fetching tools from {len(running_servers)} running servers: {running_servers}")
            fetch_tasks = []
            task_to_server = {}
            for server_name in running_servers:
                # Use the imported get_server_tools function
                task = asyncio.create_task(self.server_manager.get_server_tools(server_name))
                fetch_tasks.append(task)
                task_to_server[task] = server_name

            # Wait for all tasks to complete
            gather_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Process results
            for task, result in zip(fetch_tasks, gather_results):
                server_name = task_to_server[task]
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to fetch tools from running server '{server_name}': {result}")
                    # Optionally update the server's entry in tools_list with an error annotation
                    for tool in tools_list:
                        if tool.name == server_name and tool.annotations.get("type") == "server":
                            tool.annotations["toolFetchError"] = str(result)
                            #tool.description += f" (Error fetching tools: {result})"
                            break
                elif isinstance(result, list): # Could be list[dict] or list[Tool]
                    logger.info(f"✅ Fetched {len(result)} tools from server '{server_name}'")
                    for tool_item in result:
                        # Handle both Tool objects and dictionaries
                        if hasattr(tool_item, 'name') and hasattr(tool_item, 'description') and hasattr(tool_item, 'inputSchema'):
                            # It's a Tool object
                            original_name = tool_item.name
                            original_description = tool_item.description
                            original_schema = tool_item.inputSchema
                            original_annotations = getattr(tool_item, 'annotations', {}) or {}
                            # Convert ToolAnnotations object to dict if needed
                            if isinstance(original_annotations, (McpToolAnnotations, ToolAnnotations)):
                                original_annotations = original_annotations.model_dump()
                        elif isinstance(tool_item, dict):
                            # It's a dictionary
                            original_name = tool_item.get('name')
                            if not original_name:
                                logger.warning(f"⚠️ Received tool dictionary from {server_name} with missing name: {tool_item}")
                                continue
                            original_description = tool_item.get('description', original_name)
                            original_schema = tool_item.get('inputSchema', {"type": "object"})
                            original_annotations = tool_item.get('annotations', {}) or {}
                            # Convert ToolAnnotations object to dict if needed
                            if isinstance(original_annotations, (McpToolAnnotations, ToolAnnotations)):
                                original_annotations = original_annotations.model_dump()
                        else:
                            logger.warning(f"⚠️ Received unsupported item type from {server_name}: {type(tool_item)} - {tool_item}")
                            continue

                        # We've already extracted the data in the previous block

                        new_tool_name = f"{server_name}.{original_name}"
                        #new_description = f"[From {server_name}] {original_description}"
                        new_description = original_description
                        new_annotations = {
                            **original_annotations,
                            "originServer": server_name,
                            "type": "server_tool" # Mark as tool provided by a dynamic server
                        }

                        # Create the new tool entry using extracted data
                        # Create custom ToolAnnotations object to allow extra fields
                        custom_annotations = ToolAnnotations(**new_annotations)
                        new_tool = Tool(
                            name=new_tool_name,
                            description=new_description,
                            inputSchema=original_schema,
                            annotations=custom_annotations
                        )
                        # Check for duplicates before adding (server tools are prefixed with server name)
                        existing_tool_names = {tool.name.lower() for tool in tools_list}
                        if new_tool_name.lower() not in existing_tool_names:
                            tools_list.append(new_tool)
                            logger.debug(f"  -> Added tool from server: {new_tool_name}")
                        else:
                            logger.warning(f"⚠️ Skipping duplicate server tool: {new_tool_name} - already exists")
                else:
                     logger.warning(f"❓ Unexpected result type from get_server_tools for '{server_name}': {type(result)}")

        logger.info(f"📝 FOUND {len(tools_list)} TOTAL TOOLS (including from servers)")

                        # Patch all tools with missing required fields for MCP client compatibility
        logger.info(f"🔧 Patching {len(tools_list)} tools with required MCP fields")
        for i, tool in enumerate(tools_list):
            #logger.debug(f"🔧 Patching tool {i+1}/{len(tools_list)}: {tool.name}")

            # Log original annotations before patching
            original_annotations = getattr(tool, 'annotations', None)
            #logger.debug(f"  -> Original annotations: {original_annotations}")
            #logger.debug(f"  -> Original annotations type: {type(original_annotations)}")
            #logger.debug(f"  -> Is annotations a dict? {isinstance(original_annotations, dict)}")
            if original_annotations and isinstance(original_annotations, dict):
                pass
                #logger.debug(f"  -> Original lastModified: {original_annotations.get('lastModified', 'NOT_FOUND')}")
                #logger.debug(f"  -> Original type: {original_annotations.get('type', 'NOT_FOUND')}")
            elif hasattr(original_annotations, 'lastModified'):
                pass
                #logger.debug(f"  -> Original lastModified (attr): {getattr(original_annotations, 'lastModified', 'NOT_FOUND')}")
            else:
                pass
                #logger.debug(f"  -> No lastModified found in original annotations")

            # Add title if missing
            if not hasattr(tool, 'title') or tool.title is None:
                tool.title = tool.name
                #logger.debug(f"  -> Added title: {tool.title}")

            # Add outputSchema if missing
            if not hasattr(tool, 'outputSchema') or tool.outputSchema is None:
                tool.outputSchema = {"type": "object"}
                #logger.debug(f"  -> Added outputSchema")

            # Add annotations if missing
            if not hasattr(tool, 'annotations') or tool.annotations is None:
                tool.annotations = {}
                #logger.debug(f"  -> Created empty annotations")

            # Handle annotations - preserve our custom ToolAnnotations objects
            if not isinstance(tool.annotations, dict):
                # If it's our custom ToolAnnotations object, keep it as is
                if isinstance(tool.annotations, ToolAnnotations):
                    #logger.debug(f"  -> Keeping custom ToolAnnotations object for {tool.name}")
                    # Add missing fields directly to the object
                    if not hasattr(tool.annotations, 'title') or tool.annotations.title is None:
                        tool.annotations.title = tool.name
                    if not hasattr(tool.annotations, 'readOnlyHint') or tool.annotations.readOnlyHint is None:
                        tool.annotations.readOnlyHint = False
                    if not hasattr(tool.annotations, 'destructiveHint') or tool.annotations.destructiveHint is None:
                        tool.annotations.destructiveHint = False
                    if not hasattr(tool.annotations, 'idempotentHint') or tool.annotations.idempotentHint is None:
                        tool.annotations.idempotentHint = True
                    if not hasattr(tool.annotations, 'openWorldHint') or tool.annotations.openWorldHint is None:
                        tool.annotations.openWorldHint = False
                else:
                    # For other non-dict annotations, convert to dict as before
                    old_annotations = tool.annotations
                    logger.debug(f"  -> Converting non-dict annotations of type {type(old_annotations)}")

                    if hasattr(old_annotations, 'model_dump'):
                        # Handle Pydantic models - this preserves all fields including extra ones
                        tool.annotations = old_annotations.model_dump()
                        logger.debug(f"  -> Converted Pydantic annotations to dict: {tool.annotations}")
                    elif hasattr(old_annotations, 'dict'):
                        # Handle older Pydantic models
                        tool.annotations = old_annotations.dict()
                        logger.debug(f"  -> Converted Pydantic annotations to dict: {tool.annotations}")
                    elif hasattr(old_annotations, '__dict__'):
                        # Convert object attributes to dictionary (fallback for non-Pydantic objects)
                        tool.annotations = old_annotations.__dict__.copy()
                        logger.debug(f"  -> Converted annotations object to dict: {tool.annotations}")
                    else:
                        # Try to convert to dict using vars() or get all attributes
                        try:
                            tool.annotations = vars(old_annotations)
                            logger.debug(f"  -> Converted annotations using vars(): {tool.annotations}")
                        except TypeError:
                            # Last resort: try to get all attributes manually
                            tool.annotations = {}
                            logger.debug(f"  -> Attempting manual conversion of annotations object")
                            for attr in dir(old_annotations):
                                if not attr.startswith('_'):
                                    try:
                                        value = getattr(old_annotations, attr)
                                        if not callable(value):
                                            tool.annotations[attr] = value
                                            logger.debug(f"  -> Added attribute {attr}: {value}")
                                    except Exception as e:
                                        logger.debug(f"  -> Failed to get attribute {attr}: {e}")
                            logger.debug(f"  -> Converted annotations manually: {tool.annotations}")
            else:
                logger.debug(f"  -> Annotations already a dict, lastModified: {tool.annotations.get('lastModified', 'NOT_FOUND')}")

            # Add required annotation fields (preserve existing ones) - only for dict annotations
            if isinstance(tool.annotations, dict):
                if 'title' not in tool.annotations or tool.annotations['title'] is None:
                    tool.annotations['title'] = tool.name
                if 'readOnlyHint' not in tool.annotations or tool.annotations['readOnlyHint'] is None:
                    tool.annotations['readOnlyHint'] = False
                if 'destructiveHint' not in tool.annotations or tool.annotations['destructiveHint'] is None:
                    tool.annotations['destructiveHint'] = False
                if 'idempotentHint' not in tool.annotations or tool.annotations['idempotentHint'] is None:
                    tool.annotations['idempotentHint'] = True
                if 'openWorldHint' not in tool.annotations or tool.annotations['openWorldHint'] is None:
                    tool.annotations['openWorldHint'] = False

            # Ensure dynamic function annotations are preserved
            if original_annotations:
                if isinstance(original_annotations, dict):
                    # Preserve all original annotations that aren't being overwritten by required fields
                    for key, value in original_annotations.items():
                        if key not in ['title', 'readOnlyHint', 'destructiveHint', 'idempotentHint', 'openWorldHint']:
                            if isinstance(tool.annotations, dict):
                                if key not in tool.annotations:
                                    tool.annotations[key] = value
                                    logger.debug(f"  -> Preserved original annotation {key}: {value}")
                                else:
                                    logger.debug(f"  -> Annotation {key} already present, keeping existing value")
                            elif isinstance(tool.annotations, ToolAnnotations):
                                # For our custom ToolAnnotations object, set attributes directly
                                if not hasattr(tool.annotations, key):
                                    setattr(tool.annotations, key, value)
                                    logger.debug(f"  -> Preserved original annotation {key}: {value}")
                                else:
                                    logger.debug(f"  -> Annotation {key} already present, keeping existing value")
                elif hasattr(original_annotations, '__dict__'):
                    # Handle object annotations
                    for key, value in original_annotations.__dict__.items():
                        if key not in ['title', 'readOnlyHint', 'destructiveHint', 'idempotentHint', 'openWorldHint']:
                            if isinstance(tool.annotations, dict):
                                if key not in tool.annotations:
                                    tool.annotations[key] = value
                                    logger.debug(f"  -> Preserved original annotation {key}: {value}")
                            elif isinstance(tool.annotations, ToolAnnotations):
                                if not hasattr(tool.annotations, key):
                                    setattr(tool.annotations, key, value)
                                    logger.debug(f"  -> Preserved original annotation {key}: {value}")

            # Log final annotations after patching
            if isinstance(tool.annotations, dict):
                pass
                #logger.debug(f"  -> Final annotations: {tool.annotations}")
                #logger.debug(f"  -> Final lastModified: {tool.annotations.get('lastModified', 'NOT_FOUND')}")
                #logger.debug(f"  -> Final type: {tool.annotations.get('type', 'NOT_FOUND')}")
            elif isinstance(tool.annotations, ToolAnnotations):
                pass
                #logger.debug(f"  -> Final annotations object: {type(tool.annotations)}")
                #logger.debug(f"  -> Final lastModified: {getattr(tool.annotations, 'lastModified', 'NOT_FOUND')}")
                #logger.debug(f"  -> Final type: {getattr(tool.annotations, 'type', 'NOT_FOUND')}")

            # Check if lastModified was preserved
            if original_annotations:
                if isinstance(original_annotations, dict) and 'lastModified' in original_annotations:
                    if isinstance(tool.annotations, dict):
                        if 'lastModified' not in tool.annotations:
                            logger.error(f"❌ LOST lastModified for tool {tool.name}!")
                            logger.error(f"❌ Original had: {original_annotations['lastModified']}")
                        else:
                            pass
                            #logger.debug(f"  -> Preserved lastModified: {tool.annotations['lastModified']}")
                    elif isinstance(tool.annotations, ToolAnnotations):
                        if not hasattr(tool.annotations, 'lastModified'):
                            logger.error(f"❌ LOST lastModified for tool {tool.name}!")
                            logger.error(f"❌ Original had: {original_annotations['lastModified']}")
                        else:
                            #logger.debug(f"  -> Preserved lastModified: {getattr(tool.annotations, 'lastModified')}")
                            pass
                elif hasattr(original_annotations, 'lastModified'):
                    if isinstance(tool.annotations, dict):
                        if 'lastModified' not in tool.annotations:
                            logger.error(f"❌ LOST lastModified for tool {tool.name} (from object attr)!")
                            logger.error(f"❌ Original had: {getattr(original_annotations, 'lastModified')}")
                        else:
                            #logger.debug(f"  -> Preserved lastModified: {tool.annotations['lastModified']}")
                            pass
                    elif isinstance(tool.annotations, ToolAnnotations):
                        if not hasattr(tool.annotations, 'lastModified'):
                            logger.error(f"❌ LOST lastModified for tool {tool.name} (from object attr)!")
                            logger.error(f"❌ Original had: {getattr(original_annotations, 'lastModified')}")
                        else:
                            #logger.debug(f"  -> Preserved lastModified: {getattr(tool.annotations, 'lastModified')}")
                            pass
                else:
                    # Don't complain about missing lastModified for internal tools
                    is_internal = tool.name.startswith('_admin') or tool.name.startswith('_function') or tool.name.startswith('_server')
                    if not is_internal:
                        logger.debug(f"  -> No lastModified in original annotations for {tool.name}")

        # --- Update Cache --- #
        # Final verification: report any duplicates as validation errors (app-aware)
        tool_keys = []
        for tool in tools_list:
            app_name = getattr(tool.annotations, 'app_name', 'unknown') if hasattr(tool, 'annotations') and tool.annotations else 'unknown'
            tool_keys.append(f"{app_name}.{tool.name}")

        unique_tool_keys = set(tool_keys)
        if len(tool_keys) != len(unique_tool_keys):
            # Find duplicates by app.name combination
            seen_keys = set()
            for i, tool in enumerate(tools_list):
                app_name = getattr(tool.annotations, 'app_name', 'unknown') if hasattr(tool, 'annotations') and tool.annotations else 'unknown'
                tool_key = f"{app_name}.{tool.name}"
                if tool_key in seen_keys:
                    # This is a duplicate - mark it as invalid
                    if hasattr(tool, 'annotations') and tool.annotations:
                        if isinstance(tool.annotations, dict):
                            tool.annotations["validationStatus"] = "INVALID"
                            tool.annotations["errorMessage"] = f"Duplicate tool: {app_name}.{tool.name}"
                        else:
                            # For ToolAnnotations objects, try to set the fields
                            setattr(tool.annotations, 'validationStatus', 'INVALID')
                            setattr(tool.annotations, 'errorMessage', f'Duplicate tool: {app_name}.{tool.name}')
                    logger.warning(f"⚠️ Marked duplicate tool as invalid: {app_name}.{tool.name}")
                else:
                    seen_keys.add(tool_key)

        self._cached_tools = list(tools_list) # Store a copy
        self._last_functions_dir_mtime = current_mtime
        self._last_servers_dir_mtime = server_mtime
        # First identify running vs. non-running servers using proper accessor
        self._last_active_server_keys = set(await self.server_manager.get_running_servers()) # Store active server keys

        return tools_list

    async def _get_prompts_list(self) -> list:
        """Core logic to return a list of available prompts (empty stub)"""
        # Currently no prompts supported
        return []

    async def _get_resources_list(self) -> list:
        """Core logic to return a list of available resources (empty stub)"""
        # Currently no resources supported
        return []

    async def send_client_log(self,
                              level: str,
                              data: Any,
                              logger_name: str = None,
                              request_id: str = None,
                              client_id: str = None,
                              seq_num: Optional[int] = None,
                              entry_point_name: Optional[str] = None,
                              message_type: str = "text", # Message content type
                              stream_id: Optional[str] = None # Stream identifier
                              ):
        """Send a log message notification to connected clients using direct WebSocket communication.

        Args:
            level: The log level ("debug", "info", "warning", "error")
            data: The log message content (can be string or structured data)
            logger_name: Optional name to identify the specific function emitting the log
            request_id: Optional ID of the original request for client-side correlation
            client_id: Optional client identifier for routing the message
            seq_num: Optional sequence number for client-side ordering
            entry_point_name: Optional name of the top-level function originally called
            message_type: Type of message content ("text", "json", "image/png", etc.). Default is "text"
            stream_id: Optional stream identifier for the message
        """
        try:
            # Normalize level to uppercase for consistency
            level = level.upper()

            # Create a simple JSON-RPC notification structure
            params = {
                "level": level,
                "data": data,
                "logger": logger_name or "unknown_caller", # The immediate caller
                "requestId": request_id,
                "entryPoint": entry_point_name or "unknown_entry_point", # The original entry point
                "messageType": message_type # Type of content (text, json, image, etc.)
            }

            # Add seqNum if provided
            if seq_num is not None:
                params["seqNum"] = seq_num
            else:
                # Log an error if seq_num is None - this shouldn't happen
                logger.error(f"❌ Missing sequence number in send_client_log for message type '{message_type}', client {client_id}, request {request_id}")
                # We continue without seqNum for backward compatibility, but this is an error condition

            # Add streamId if provided
            if stream_id is not None:
                params["streamId"] = stream_id

            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": params # Use the params dict built above
            }

            # Get the global tracking collections
            global active_websockets, client_connections, current_request_client_id

            # If no specific client_id was provided, try to use the one from the current request
            if client_id is None and 'current_request_client_id' in globals():
                client_id = current_request_client_id

            # ONLY send to the specific client that made the request - NO broadcasting
            if client_id and client_id in client_connections:
                # Convert to JSON string
                notification_json = json.dumps(notification)


                client_info = client_connections[client_id]
                client_type = client_info.get("type")
                connection = client_info.get("connection")

                if client_type == "websocket" and connection:
                    try:
                        await connection.send_text(notification_json)
                        logger.debug(f"📢 Sent notification to specific client: {client_id}")
                    except Exception as e:
                        logger.warning(f"Failed to send to client {client_id}: {e}")

                elif client_type == "cloud" and connection and hasattr(connection, 'is_connected') and connection.is_connected:
                    try:
                        await connection.send_message('mcp_notification', notification)
                        #logger.debug(f"☁️ Sent notification to cloud client: {client_id}")
                    except Exception as e:
                        logger.warning(f"Failed to send to cloud client {client_id}: {e}")
            else:
                logger.warning(f"Cannot send client log: no valid client_id provided or client not found: {client_id}")
                # Log the client connections we know about for debugging
                logger.debug(f"Known client connections: {list(client_connections.keys())}")

        except Exception as e:
            # Don't let logging errors affect the main operation
            logger.error(f"❌ Error sending direct client log notification: {str(e)}")
            logger.debug(f"Log notification error details: {traceback.format_exc()}")
            # We intentionally don't re-raise here

    async def _execute_tool(self, name: str, args: dict, client_id: str = None, request_id: str = None, user: str = None, session_id: str = None, command_seq: int = None) -> list[TextContent]:
        """Core logic to handle a tool call. Ensures result is List[TextContent(type='text')]"""
        logger.info(f"🔧 EXECUTING TOOL: {name}")
        logger.debug(f"WITH ARGUMENTS: {args}")
        if user:
            logger.debug(f"CALLED BY USER: {user}")
        if session_id:
            logger.debug(f"SESSION ID: {session_id}")
        # ---> ADDED: Log entry and raw args
        logger.debug(f"---> _execute_tool ENTERED. Name: '{name}', Raw Args:\n{format_json_log(args) if isinstance(args, dict) else args!r}") # <-- ADD THIS LINE

        # Parse compound tool names by splitting on '*'
        # Format: "remote_owner*remote_name*app*location*function" (empty fields allowed)
        actual_function_name = name
        parsed_app_name = None
        parsed_location_name = None
        parsed_owner_name = None
        parsed_remote_name = None

        if "*" in name:
            parts = name.split("*")
            if len(parts) >= 5:
                # Full format: remote_owner*remote_name*app*location*function
                parsed_owner_name = parts[0] if parts[0] else None
                parsed_remote_name = parts[1] if parts[1] else None
                parsed_app_name = parts[2] if parts[2] else None
                parsed_location_name = parts[3] if parts[3] else None
                actual_function_name = parts[4]

                logger.info(f"🔍 PARSED TOOL NAME: owner='{parsed_owner_name}', remote='{parsed_remote_name}', app='{parsed_app_name}', location='{parsed_location_name}', function='{actual_function_name}'")

        logger.debug(f"Actual function name to route: '{actual_function_name}'")

        # --- BEGIN TOOL CALL LOGGING ---
        # Note: We now log on SUCCESS (after execution) or ERROR (in exception handler)
        # This avoids duplicate entries for failed calls
        # Store call start time for logging later
        call_start_datetime = datetime.datetime.now(datetime.timezone.utc)
        call_start_time = call_start_datetime.isoformat()
        should_log_call = not (actual_function_name.startswith('_function') or actual_function_name.startswith('_server') or actual_function_name.startswith('_admin'))
        # --- END TOOL CALL LOGGING ---

        try:
            result_raw = None # Initialize raw result variable

            # Security check: Only owner can call internal functions
            if (actual_function_name.startswith('_function') or
                actual_function_name.startswith('_server') or
                actual_function_name.startswith('_admin')):

                caller = user or client_id or "unknown"
                owner = atlantis.get_owner()

                # Treat localhost websocket connections as the owner
                if caller.startswith("ws_127.0.0.1_") and owner:
                    caller = owner

                if owner and caller != owner:
                    logger.warning(f"🚨 SECURITY: Internal function '{actual_function_name}' called by '{caller}' but owner is '{owner}' - ACCESS DENIED")
                    raise ValueError(f"Access denied: Internal functions can only be accessed by owner")

                logger.debug(f"✅ Internal function '{actual_function_name}' authorized for owner: {caller}")

            # Handle built-in tool calls
            if actual_function_name == "_function_set":
                logger.debug(f"---> Calling built-in: function_set") # <-- ADD THIS LINE
                # function_set now returns (extracted_name, result_messages)
                extracted_name, result_messages = await self.function_manager.function_set(args, self)
                result_raw = result_messages # Use the messages returned by function_set
                if extracted_name:
                    # Notify only if function_set successfully extracted a name
                    await self._notify_tool_list_changed(change_type="updated", tool_name=extracted_name)
            elif actual_function_name == "_function_get":
                logger.debug(f"---> Calling built-in: get_function_code") # <-- ADD THIS LINE
                result_raw = await self.function_manager.get_function_code(args, self)
            elif actual_function_name == "_function_remove":
                # Remove function
                func_name = args.get("name")
                app_name = args.get("app")  # Optional app name for disambiguation
                if not func_name:
                    raise ValueError("Missing required parameter: name")

                logger.debug(f"---> Calling built-in: function_remove for '{func_name}'" + (f" (app: {app_name})" if app_name else ""))

                # Check if function exists using the function mapping (supports subfolders and app-specific lookup)
                function_file = await self.function_manager._find_file_containing_function(func_name, app_name)

                if not function_file:
                    if app_name:
                        error_message = f"Function '{func_name}' does not exist in app '{app_name}'."
                    else:
                        error_message = f"Function '{func_name}' does not exist."
                    error_annotations = {
                        "tool_error": {"tool_name": name, "message": error_message}
                    }
                    result_raw = [TextContent(type="text", text=error_message, annotations=error_annotations)]
                else:
                    # Function exists, continue with removing it
                    await self.function_manager.function_remove(func_name, app_name)
                    try:
                        await self._notify_tool_list_changed(change_type="removed", tool_name=func_name) # Pass params
                    except Exception as e:
                        logger.error(f"Error sending tool notification after removing {func_name}: {str(e)}")
                    result_raw = [TextContent(type="text", text=f"Function '{func_name}' removed successfully.")] # <-- Success message

            elif actual_function_name == "_function_add":
                # Add empty function
                logger.debug(f"---> Calling built-in: function_add with args:\n{format_json_log(args)}")

                func_name = args.get("name")
                app_name = args.get("app")  # Optional app name for disambiguation
                if not func_name:
                    raise ValueError("Missing required parameter: name")

                # Check if function already exists using the function mapping (supports subfolders and app-specific lookup)
                function_file = await self.function_manager._find_file_containing_function(func_name, app_name)

                if function_file:
                    # Function already exists - find all locations containing this function
                    existing_locations = []
                    if not app_name:
                        # Find ALL apps that contain this function, with filenames
                        for existing_app_path, app_functions in self.function_manager._function_file_mapping_by_app.items():
                            if func_name in app_functions:
                                filename = app_functions[func_name]
                                if existing_app_path == "unknown":
                                    location = f"root {filename}"
                                else:
                                    # Convert slash path to dot notation for display to user
                                    existing_app_display = self.function_manager._path_to_app_name(existing_app_path) if existing_app_path else "unknown"
                                    location = f"{existing_app_display} {filename}"
                                existing_locations.append(location)

                    # Create detailed error message
                    if app_name:
                        error_message = f"Function '{func_name}' already exists in app '{app_name}'."
                    elif existing_locations:
                        if len(existing_locations) == 1:
                            error_message = f"Function '{func_name}' already exists in {existing_locations[0]}."
                        else:
                            locations_list = ", ".join(existing_locations)
                            error_message = f"Function '{func_name}' already exists in: {locations_list}. Specify an app parameter."
                    else:
                        error_message = f"Function '{func_name}' already exists."
                    error_annotations = {
                        "tool_error": {"tool_name": name, "message": error_message}
                    }
                    result_raw = [TextContent(type="text", text=error_message, annotations=error_annotations)]
                else:
                    # Function doesn't exist, create it
                    await self.function_manager.function_add(func_name, None, app_name)
                    try:
                        await self._notify_tool_list_changed(change_type="added", tool_name=func_name) # Pass params
                    except Exception as e:
                        logger.error(f"Error sending tool notification after adding {func_name}: {str(e)}")
                    result_raw = [TextContent(type="text", text=f"Empty function '{func_name}' created successfully.")] # <-- Success message

            elif actual_function_name == "_server_get":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_get for '{svc_name}'")
                result_raw = await self.server_manager.server_get(svc_name)
            elif actual_function_name == "_server_add":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: 'name' must be a string")
                logger.debug(f"---> Calling built-in: server_add for '{svc_name}'")
                # Now server_add only requires the name parameter and creates a template
                success = await self.server_manager.server_add(svc_name)
                if success:
                    try:
                        await self._notify_tool_list_changed(change_type="added", tool_name=svc_name)
                    except Exception as e:
                        logger.error(f"Error sending tool notification after adding server {svc_name}: {str(e)}")
                    result_raw = [TextContent(type="text", text=f"MCP '{svc_name}' added successfully.")]
                else:
                    result_raw = [TextContent(type="text", text=f"Failed to add MCP '{svc_name}'.")]
            elif actual_function_name == "_server_remove":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_remove for '{svc_name}'")
                success = await self.server_manager.server_remove(svc_name)
                if success:
                    try:
                        await self._notify_tool_list_changed(change_type="removed", tool_name=svc_name)
                    except Exception as e:
                        logger.error(f"Error sending tool notification after removing server {svc_name}: {str(e)}")
                    result_raw = [TextContent(type="text", text=f"Server '{svc_name}' removed successfully.")]
                else:
                    result_raw = [TextContent(type="text", text=f"Failed to remove server '{svc_name}'.")]
            elif actual_function_name == "_server_set":
                logger.debug(f"---> Calling built-in: server_set with args:\n{format_json_log(args)}")
                # Extract the config from the args dictionary
                config = args.get("config")
                if not config:
                    raise ValueError("Missing required parameter: config")

                # Try to extract the server name from the config JSON, but allow non-JSON content
                try:
                    server_name = self.server_manager.extract_server_name(config)
                    if not server_name:
                        raise ValueError("Failed to get server name")
                except Exception as e:
                    # If parsing fails completely, we need an explicit name
                    logger.warning(f"Could not parse config as JSON: {e}")
                    # Check if name was provided directly
                    server_name = args.get('name')
                    if not server_name:
                        raise ValueError("Unable to resolve server name")

                # Call server_set with the correct parameters
                result_raw = await self.server_manager.server_set(server_name, config)
                # Notify clients that server configuration changed (could affect available tools)
                if result_raw:  # Only notify on success
                    try:
                        await self._notify_tool_list_changed(change_type="updated", tool_name=server_name)
                    except Exception as e:
                        logger.error(f"Error sending tool notification after updating server {server_name}: {str(e)}")
            elif actual_function_name == "_server_validate":
                svc_name = args.get("name")
                if not svc_name:
                    raise ValueError("Missing required parameter: name")
                logger.debug(f"---> Calling built-in: server_validate for '{svc_name}'")
                result_raw = await self.server_manager.server_validate(svc_name)
            elif actual_function_name == "_server_start":
                logger.debug(f"---> Calling built-in: server_start with args:\n{format_json_log(args)}")
                result_raw = await self.server_manager.server_start(args, self)
                # Wait for server to be ready before notifying clients
                server_name = args.get('name')
                if server_name:
                    try:
                        # Wait for the server to finish initializing (with timeout)
                        task_info = self.server_manager.server_tasks.get(server_name)
                        if task_info and 'ready_event' in task_info:
                            logger.debug(f"Waiting for server '{server_name}' to be ready before sending notification...")
                            await asyncio.wait_for(task_info['ready_event'].wait(), timeout=30.0)
                            logger.info(f"Server '{server_name}' is ready, invalidating cache and regenerating tool list")
                            # Invalidate cache to force regeneration with new server tools
                            self._cached_tools = None
                            # Regenerate tool list (this will log all tools including new ones from the server)
                            updated_tools = await self._get_tools_list(caller_context=f"after_server_start:{server_name}")
                            # Report the updated tool list to console
                            logger.info(f"📊 Reporting updated tool list after MCP server '{server_name}' started...")
                            await self.cloud_client._report_tools_to_console(tools_list=updated_tools)
                            # Now send notification to clients
                            logger.info(f"Sending tool list change notification for '{server_name}'")
                            await self._notify_tool_list_changed(change_type="updated", tool_name=server_name)
                        else:
                            logger.warning(f"No ready_event found for server '{server_name}', skipping notification wait")
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for server '{server_name}' to be ready, sending notification anyway")
                        await self._notify_tool_list_changed(change_type="updated", tool_name=server_name)
                    except Exception as e:
                        logger.error(f"Error sending tool notification after starting server {server_name}: {str(e)}")
            elif actual_function_name == "_server_stop":
                logger.debug(f"---> Calling built-in: server_stop with args:\n{format_json_log(args)}")
                result_raw = await self.server_manager.server_stop(args, self)
                # Notify clients that tool list has changed (server stopped = tools removed)
                server_name = args.get('name')
                if server_name:
                    try:
                        logger.info(f"Server '{server_name}' stopped, invalidating cache and regenerating tool list")
                        # Invalidate cache to force regeneration without stopped server tools
                        self._cached_tools = None
                        # Regenerate tool list (this will log tools without the stopped server)
                        updated_tools = await self._get_tools_list(caller_context=f"after_server_stop:{server_name}")
                        # Report the updated tool list to console
                        logger.info(f"📊 Reporting updated tool list after MCP server '{server_name}' stopped...")
                        await self.cloud_client._report_tools_to_console(tools_list=updated_tools)
                        # Now send notification to clients
                        logger.info(f"Sending tool list change notification for '{server_name}'")
                        await self._notify_tool_list_changed(change_type="updated", tool_name=server_name)
                    except Exception as e:
                        logger.error(f"Error sending tool notification after stopping server {server_name}: {str(e)}")
            elif actual_function_name == "_server_get_tools":
                server_name = args.get('name')
                if not server_name or not isinstance(server_name, str):
                     raise ValueError("Missing or invalid 'name' argument for _server_get_tools")
                # Check if server was running before the call (to detect auto-start)
                was_running = await self.server_manager.is_server_running(server_name)
                result_raw = await self.server_manager.get_server_tools(server_name) # Pass only the name string (may auto-start)
                # If server was auto-started, send notification
                if not was_running:
                    is_running_now = await self.server_manager.is_server_running(server_name)
                    if is_running_now:
                        logger.info(f"Server '{server_name}' was auto-started by get_server_tools")
                        try:
                            logger.info(f"Invalidating cache and regenerating tool list after auto-start")
                            # Invalidate cache to force regeneration with new server tools
                            self._cached_tools = None
                            # Regenerate tool list (this will log all tools including new ones from the server)
                            updated_tools = await self._get_tools_list(caller_context=f"after_autostart:{server_name}")
                            # Report the updated tool list to console
                            logger.info(f"📊 Reporting updated tool list after MCP server '{server_name}' auto-started...")
                            await self.cloud_client._report_tools_to_console(tools_list=updated_tools)
                            # Now send notification to clients
                            logger.info(f"Sending tool list change notification for '{server_name}'")
                            await self._notify_tool_list_changed(change_type="updated", tool_name=server_name)
                        except Exception as e:
                            logger.error(f"Error sending tool notification after auto-starting server {server_name}: {str(e)}")
                # Convert Tool objects to dictionaries for JSON serialization
                if result_raw and isinstance(result_raw, list):
                    result_raw = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        for tool in result_raw
                    ]
            elif actual_function_name == "_function_history":
                app_name = args.get("app")  # Required app name
                function_name = args.get("name")  # Required function name
                logger.debug(f"---> Calling built-in: _function_history (app: {app_name}, function: {function_name})")

                # First, validate that the function exists
                try:
                    function_file = await self.function_manager._find_file_containing_function(function_name, app_name)
                    if not function_file:
                        raise ValueError(f"Function '{function_name}' not found in app '{app_name or '(root)'}'")
                except Exception as e:
                    raise ValueError(f"Function '{function_name}' not found in app '{app_name or '(root)'}': {e}")

                # Now get the history for this function
                if not os.path.exists(TOOL_CALL_LOG_PATH):
                    # No log file yet means no history (but function exists)
                    result_raw = []
                else:
                    try:
                        with open(TOOL_CALL_LOG_PATH, "r", encoding="utf-8") as f:
                            # Read each line and parse as JSON, skipping empty lines
                            log_entries = [json.loads(line) for line in f if line.strip()]

                        # Filter by BOTH app_name and function name
                        # NOTE: History is tracked separately per app. If you move a function
                        # from one app to another (e.g., from root "" to "myapp" or vice versa),
                        # a new separate history will be created for that app. Previous history
                        # from the old app location will remain but won't be shown in the new location.
                        filtered_entries = [
                            entry for entry in log_entries
                            if (entry.get("tool_name") == function_name and
                                entry.get("app_name") == (app_name if app_name else ""))
                        ]

                        result_raw = filtered_entries
                    except (json.JSONDecodeError, IOError) as e:
                        logger.error(f"Error reading or parsing tool_call_log.json: {e}")
                        raise ValueError(f"Error accessing function history: {e}")

            elif actual_function_name == "_function_log":
                app_name = args.get("app")  # Optional app name for filtering
                logger.debug("---> Calling built-in: _function_log" + (f" (app: {app_name})" if app_name else ""))
                if not os.path.exists(OWNER_LOG_PATH):
                    # Return an empty history array
                    result_raw = []
                else:
                    try:
                        with open(OWNER_LOG_PATH, "r", encoding="utf-8") as f:
                            file_content = f.read()
                            # Handle empty file or file with only whitespace
                            if not file_content.strip():
                                log_entries = [] # Return empty list if file is effectively empty
                            else:
                                log_entries = json.loads(file_content) # Parse the string content

                        # If app filter is provided, we could potentially filter here
                        # For now, return all entries as the current log format may not track app info
                        # TODO: Implement app-based filtering when log format includes app information
                        # Return the list of log entries directly
                        result_raw = log_entries
                    except (json.JSONDecodeError, IOError) as e:
                        logger.error(f"Error reading or parsing owner log: {e}")
                        # Return an error message inside the tool response
                        raise ValueError(f"Error accessing function history: {e}")

            elif actual_function_name == "_admin_restart":
                # Restart the server by terminating the process (wrapper script should restart it)
                logger.info(f"🔄 ADMIN RESTART requested by owner: {user or 'unknown'}")

                # Send response first before terminating
                result_raw = [TextContent(type="text", text="Server restart initiated. The process will terminate and should be restarted by wrapper script.")]

                # Schedule termination after a brief delay to allow response to be sent
                async def delayed_shutdown():
                    await asyncio.sleep(0.1)  # Brief delay to send response
                    logger.info("🛑 Sending SIGINT for graceful restart...")
                    os.kill(os.getpid(), signal.SIGINT)  # Same as Ctrl-C - triggers graceful shutdown

                # Schedule the shutdown task
                asyncio.create_task(delayed_shutdown())

            elif actual_function_name == "_admin_pip_install":
                # Install Python packages using pip
                package = args.get("package")
                force = False  # Hardcoded to false for now

                if not package:
                    raise ValueError("Missing required parameter: package")

                logger.info(f"📦 ADMIN PIP INSTALL requested by owner: {user or 'unknown'} - Package: {package}")

                # Build pip command
                pip_cmd = [sys.executable, "-m", "pip", "install", package]
                if force:
                    pip_cmd.append("--force-reinstall")

                try:
                    # Run pip install command
                    result = subprocess.run(
                        pip_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )

                    if result.returncode == 0:
                        success_msg = f"✅ Successfully installed package: {package}"
                        if result.stdout:
                            success_msg += f"\n\nOutput:\n{result.stdout}"
                        logger.info(f"📦 Package '{package}' installed successfully")
                        result_raw = [TextContent(type="text", text=success_msg)]
                    else:
                        error_msg = f"❌ Failed to install package: {package}"
                        if result.stderr:
                            error_msg += f"\n\nError output:\n{result.stderr}"
                        if result.stdout:
                            error_msg += f"\n\nStdout:\n{result.stdout}"
                        logger.error(f"📦 Failed to install package '{package}': {result.stderr}")
                        result_raw = [TextContent(type="text", text=error_msg)]

                except subprocess.TimeoutExpired:
                    timeout_msg = f"⏰ Pip install timed out after 5 minutes for package: {package}"
                    logger.error(f"📦 Pip install timeout for package '{package}'")
                    result_raw = [TextContent(type="text", text=timeout_msg)]
                except Exception as e:
                    error_msg = f"💥 Error during pip install for package '{package}': {str(e)}"
                    logger.error(f"📦 Pip install error for package '{package}': {str(e)}")
                    result_raw = [TextContent(type="text", text=error_msg)]

            elif actual_function_name == "_admin_click":
                # Handle click events by invoking stored callbacks as dynamic functions
                key = args.get("key")
                if not key:
                    raise ValueError("Missing required parameter: key")

                logger.info(f"🖱️ ADMIN CLICK: {user} clicked key: {key}")

                # Check if we have a callback for this key using the global atlantis
                callback = atlantis._click_callbacks.get(key)
                logger.info(f"🖱️ Available callback keys: {list(atlantis._click_callbacks.keys())}")

                if callback:
                    logger.info(f"🖱️ Found callback for key '{key}', invoking as dynamic function...")

                    # Create a wrapper function that can be called as a dynamic function
                    # We'll inject it temporarily into atlantis and call it through function_manager
                    wrapper_name = f"_click_callback_{key.replace('-', '_').replace('.', '_')}"

                    # Inject the callback into the existing atlantis context temporarily
                    setattr(atlantis, wrapper_name, callback)

                    try:
                        # Invoke through function manager to get full dynamic function context
                        result_raw = await self.function_manager.function_call(
                            name=wrapper_name,
                            client_id=client_id,
                            request_id=request_id,
                            user=user,
                            session_id=session_id,
                            app=None,  # No specific app
                            args={}    # No additional args
                        )
                    finally:
                        # Clean up the temporary wrapper
                        if hasattr(atlantis, wrapper_name):
                            delattr(atlantis, wrapper_name)
                else:
                    logger.info(f"🖱️ No callback found for key '{key}'")
                    click_msg = f"🖱️ Click received for key '{key}' but no callback registered"
                    result_raw = [TextContent(type="text", text=click_msg)]

            elif actual_function_name == "_admin_upload":
                # Handle uploads by invoking stored callbacks as dynamic functions
                key = args.get("key")
                filename = args.get("filename")
                filetype = args.get("filetype")
                base64Content = args.get("base64Content")

                if not key:
                    raise ValueError("Missing required parameter: key")
                if not filename:
                    raise ValueError("Missing required parameter: filename")
                if not filetype:
                    raise ValueError("Missing required parameter: filetype")
                if not base64Content:
                    raise ValueError("Missing required parameter: base64Content")

                logger.info(f"🖱️ ADMIN UPLOAD: {user} uploading id: {key}, filename: {filename}, filetype: {filetype}")

                # Check if we have a callback for this key using the global atlantis
                callback = atlantis._upload_callbacks.get(key)
                logger.info(f"🖱️ Available callback keys: {list(atlantis._upload_callbacks.keys())}")

                if callback:
                    logger.info(f"🖱️ Found callback for key '{key}', invoking as dynamic function...")

                    # Create a wrapper function that can be called as a dynamic function
                    # We'll inject it temporarily into atlantis and call it through function_manager
                    wrapper_name = f"_upload_callback_{key.replace('-', '_').replace('.', '_')}"

                    # Inject the callback into the existing atlantis context temporarily
                    setattr(atlantis, wrapper_name, callback)

                    try:
                        # Invoke through function manager to get full dynamic function context
                        result_raw = await self.function_manager.function_call(
                            name=wrapper_name,
                            client_id=client_id,
                            request_id=request_id,
                            user=user,
                            session_id=session_id,
                            app=None,  # No specific app
                            args={"filename": filename, "filetype": filetype, "base64Content": base64Content}
                        )
                    finally:
                        # Clean up the temporary wrapper
                        if hasattr(atlantis, wrapper_name):
                            delattr(atlantis, wrapper_name)
                else:
                    logger.info(f"🖱️ No callback found for key '{key}'")
                    upload_msg = f"🖱️ Upload received for key '{key}' but no callback registered"
                    result_raw = [TextContent(type="text", text=upload_msg)]

            elif actual_function_name == "_admin_app_create":
                # Create a new app directory with main.py containing empty index() function
                app_name = args.get("appName")
                if not app_name:
                    raise ValueError("Missing required parameter: appName")

                logger.info(f"📁 ADMIN APP CREATE requested by owner: {user or 'unknown'} - App: {app_name}")

                # Create main.py with index() function stub using the proper stub generator
                filename = "main"
                function_name = "index"
                # Use the DynamicFunctionManager's stub generator for consistency
                main_stub = self.function_manager._code_generate_stub(function_name)
                await self.function_manager.function_add(filename, main_stub, app_name)
                try:
                    await self._notify_tool_list_changed(change_type="added", tool_name=function_name)
                except Exception as e:
                    logger.error(f"Error sending tool notification after adding {function_name}: {str(e)}")
                result_raw = [TextContent(type="text", text=f"✅ Successfully created app '{app_name}' with {filename}.py containing {function_name}() function")]

            elif actual_function_name == "_admin_git_update":
                # Update server code by running git fetch and git merge
                remote = args.get("remote", "origin")
                branch = args.get("branch", "main")

                logger.info(f"🔄 ADMIN GIT UPDATE requested by owner: {user or 'unknown'} - Remote: {remote}, Branch: {branch}")

                try:
                    # Run git fetch
                    fetch_result = subprocess.run(
                        ["git", "fetch", remote],
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )

                    if fetch_result.returncode != 0:
                        error_msg = f"❌ Git fetch failed: {fetch_result.stderr.strip()}"
                        logger.error(f"🔄 Git fetch error: {error_msg}")
                        result_raw = [TextContent(type="text", text=error_msg)]
                    else:
                        # Run git merge
                        merge_result = subprocess.run(
                            ["git", "merge", f"{remote}/{branch}"],
                            capture_output=True,
                            text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))
                        )

                        if merge_result.returncode != 0:
                            error_msg = f"❌ Git merge failed: {merge_result.stderr.strip()}"
                            logger.error(f"🔄 Git merge error: {error_msg}")
                            result_raw = [TextContent(type="text", text=error_msg)]
                        else:
                            success_msg = f"✅ Successfully updated from {remote}/{branch}"
                            if merge_result.stdout.strip():
                                success_msg += f"\n{merge_result.stdout.strip()}"
                            logger.info(f"🔄 Git update successful: {success_msg}")
                            result_raw = [TextContent(type="text", text=success_msg)]

                except Exception as e:
                    error_msg = f"❌ Git update failed with exception: {str(e)}"
                    logger.error(f"🔄 Git update exception: {error_msg}")
                    result_raw = [TextContent(type="text", text=error_msg)]

            elif actual_function_name.startswith('_function') or actual_function_name.startswith('_server') or actual_function_name.startswith('_admin'):
                # Catch-all for invalid internal functions (only _function* and _server* are internal)
                error_message = f"Invalid internal function: '{actual_function_name}'. Check available internal functions."
                error_annotations = {
                    "tool_error": {"tool_name": actual_function_name, "message": error_message}
                }
                result_raw = [TextContent(type="text", text=error_message, annotations=error_annotations)]

            # Handle MCP tool calls (check actual_function_name after asterisk parsing to avoid false positives from dots in app names)
            elif '.' in actual_function_name or ' ' in actual_function_name:

                # --- Handle MCP tool call ---

                logger.info(f"🌐 MCP TOOL CALL: {actual_function_name}")
                # Split on the first occurrence of '.' or ' '
                server_alias, tool_name_on_server = re.split('[. ]', actual_function_name, 1)
                logger.debug(f"Parsed: Server Alias='{server_alias}', Remote Tool='{tool_name_on_server}'")

                # Check if MCP server config exists and is running
                if server_alias not in self._server_configs: # Access instance variable
                       raise ValueError(f"Unknown server alias: '{server_alias}'")
                if not await self.server_manager.is_server_running(server_alias):
                    raise ValueError(f"Server '{server_alias}' is not running.")

                # Get the session for the target server
                task_info = self.server_manager.server_tasks.get(server_alias)
                if not task_info:
                     # This shouldn't happen if the check above passed, but safety first
                    raise ValueError(f"Could not find task info for running server '{server_alias}'.")

                session = task_info.get('session')
                ready_event = task_info.get('ready_event')

                if not session and ready_event:
                    session_ready_timeout = 5.0 # Allow a bit more time for proxy calls
                    logger.debug(f"Session for '{server_alias}' not immediately ready for proxy call. Waiting up to {session_ready_timeout}s...")
                    try:
                        await asyncio.wait_for(ready_event.wait(), timeout=session_ready_timeout)
                        session = task_info.get('session') # Re-fetch session after wait
                        if not session:
                            raise ValueError(f"Server '{server_alias}' session not available even after waiting.")
                        logger.debug(f"Session for '{server_alias}' became ready.")
                    except asyncio.TimeoutError:
                         raise ValueError(f"Timeout waiting for server '{server_alias}' session to become ready for proxy call.")
                elif not session:
                     # Session not available and no ready_event to wait for
                     raise ValueError(f"Server '{server_alias}' is running but its session is not available and cannot wait (no ready_event).")

                # Proxy the call using the retrieved session
                try:
                    logger.info(f"🌐 PROXYING tool call '{tool_name_on_server}' to server '{server_alias}' with args: {args}")
                    # Use the standard request timeout defined elsewhere
                    proxy_response: CallToolResult = await asyncio.wait_for(
                        session.call_tool(tool_name_on_server, args or {}),
                        timeout=SERVER_REQUEST_TIMEOUT
                    )
                    logger.info(f"✅ PROXY response received from '{server_alias}'")
                    logger.debug(f"Raw Proxy Response: {proxy_response}") # proxy_response is CallToolResult

                    # Directly use the content from the proxied result if it's a list.
                    # The isError flag from proxy_response is implicitly handled by the caller
                    # receiving our server's CallToolResult (which we don't explicitly build here,
                    # the framework does it based on the returned content list and exceptions).
                    if proxy_response.content and isinstance(proxy_response.content, list):
                         final_result = proxy_response.content
                         logger.debug(f"Using content list directly from proxied response: {final_result}")
                         logger.debug(f"<--- _execute_tool RETURNING proxied result directly.")
                         return final_result
                    else:
                        # Fallback: If content is missing or not a list - THIS is an error in proxy process.
                        error_message = f"Proxied server '{server_alias}' returned unexpected content format (expected list): {proxy_response.content}"
                        logger.error(error_message)
                        # Raise an exception, which the framework will turn into an error response
                        raise ValueError(error_message)

                except McpError as mcp_err:
                    logger.error(f"❌ MCPError proxying tool call '{name}' to '{server_alias}': {mcp_err}", exc_info=True)
                    # Format the MCPError into a user-friendly error message
                    error_message = f"Error calling '{tool_name_on_server}' on server '{server_alias}': {mcp_err.message} (Code: {mcp_err.code})"
                    raise ValueError(error_message) from mcp_err
                except asyncio.TimeoutError:
                    logger.error(f"❌ Timeout proxying tool call '{name}' to '{server_alias}'.")
                    raise ValueError(f"Timeout calling '{tool_name_on_server}' on server '{server_alias}'.")
                except Exception as proxy_err:
                    logger.error(f"❌ Unexpected error proxying tool call '{name}' to '{server_alias}': {proxy_err}", exc_info=True)
                    raise ValueError(f"Unexpected error calling '{tool_name_on_server}' on server '{server_alias}': {proxy_err}") from proxy_err

            elif not (actual_function_name.startswith('_function') or actual_function_name.startswith('_server') or actual_function_name.startswith('_admin')):

                # --- Handle Local Dynamic Function Call ---
                logger.info(f"🔧 CALLING LOCAL DYNAMIC FUNCTION: {name}")

                # warn if cached load error
                if name in _runtime_errors:
                     load_error_info = _runtime_errors[name]
                     logger.warning(f"❌ Function '{name}' has a cached load error: {load_error_info['error']}")
                     # Maybe return a specific error message here instead of raising ValueError?
                     # Creating an error TextContent for consistency

                # NEW: Function calling now uses function-to-file mapping internally
                # No need to check if {name}.py exists since function_call handles that

                # Call the dynamic function
                try:
                    # Dynamic functions are directly handled by name matching
                    # Add detailed logging to show exactly what we're receiving from the cloud
                    logger.info(f"RECEIVED FROM CLOUD: Tool: '{name}', Type: {type(args)}, Raw Args:\n{format_json_log(args) if isinstance(args, dict) else args!r}")
                    if user:
                        logger.info(f"Call made by user: {user}")
                    logger.debug(f"---> Calling dynamic: function_call for '{actual_function_name}' with args: {args} and client_id: {client_id} and request_id: {request_id}, user: {user}") # Log args and client_id separately
                    # Pass arguments, client_id, user, and session_id distinctly
                    # Pass parsed app name for proper function routing
                    final_args = args.copy() if args else {}
                    result_raw = await self.function_manager.function_call(name=actual_function_name, client_id=client_id, request_id=request_id, user=user, session_id=session_id, command_seq=command_seq, app=parsed_app_name, args=final_args)
                    logger.debug(f"<--- Dynamic function '{name}' RAW result: {result_raw} (type: {type(result_raw)})")
                except Exception as e:
                    # Error already enhanced with command context at source, just re-raise
                    raise


            else:
                # Handle unknown tool names starting with '_', or if no branch matched
                logger.error(f"❓ Unknown or unhandled tool name: {name}")
                raise ValueError(f"Unknown or unhandled tool name: {name}")

            # ---> ADDED: Process raw result into final_result format (List[TextContent]) with June 2025 MCP spec support
            if isinstance(result_raw, str):
                final_result = [TextContent(type="text", text=result_raw)]
            elif isinstance(result_raw, dict):
                # Support new June 2025 MCP spec: provide both serialized JSON and structuredContent
                logger.debug(f"<--- Creating structured content result for tool '{name}' (June 2025 MCP spec).")
                try:
                    # Use pretty JSON formatting for dict results
                    json_string = format_json_log(result_raw, colored=True)
                    # Create TextContent with both text and structured content for backwards compatibility
                    annotations = {
                        "sourceType": "json",
                        "structuredContent": result_raw  # New June 2025 spec feature
                    }
                    result_content = [TextContent(type="text", text=json_string, annotations=annotations)]
                except TypeError as e:
                    logger.error(f"Error serializing dictionary result to JSON for tool '{name}': {e}")
                    result_content = [TextContent(type="error", text=f"Error serializing result: {e}")]
                final_result = result_content
            elif isinstance(result_raw, list) and all(isinstance(item, TextContent) for item in result_raw):
                final_result = result_raw
            elif result_raw is None:
                final_result = [] # Assign a default empty list
            else:
                # Convert any other result to string with structured content support
                try:
                    # Try to serialize as JSON first for structured content
                    # Check if result_raw is a list of dicts (like history entries)
                    if isinstance(result_raw, list) and result_raw and all(isinstance(item, dict) for item in result_raw):
                        # Pretty format list of dicts
                        result_str = json.dumps(result_raw, indent=2, default=str)
                    elif isinstance(result_raw, dict):
                        # Pretty format dict
                        result_str = format_json_log(result_raw, colored=True)
                    else:
                        # Use standard compact formatting for other types
                        result_str = json.dumps(result_raw)
                    annotations = {
                        "sourceType": "json",
                        "structuredContent": result_raw  # New June 2025 spec feature
                    }
                    final_result = [TextContent(type="text", text=result_str, annotations=annotations)]
                except TypeError:
                    result_str = str(result_raw) # Fallback to plain string conversion
                    logger.warning(f"⚠️ Tool '{name}' returned non-standard type {type(result_raw)}. Converting to string: {result_str}")
                    final_result = [TextContent(type="text", text=result_str)]

            # ---> ADDED: Log final result before returning
            # Pretty print the actual text content for better readability in logs
            if final_result and isinstance(final_result, list) and len(final_result) > 0:
                first_item = final_result[0]
                if isinstance(first_item, TextContent) and hasattr(first_item, 'text'):
                    # Try to parse and colorize the JSON if it's valid JSON
                    try:
                        parsed_json = json.loads(first_item.text)
                        colored_output = format_json_log(parsed_json, colored=True)
                        logger.debug(f"<--- _execute_tool RETURNING final result with text:\n{colored_output}")
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON or can't parse, just log as-is
                        logger.debug(f"<--- _execute_tool RETURNING final result with text:\n{first_item.text}")
                else:
                    logger.debug(f"<--- _execute_tool RETURNING final result:\n{format_json_log(final_result) if isinstance(final_result, dict) else final_result!r}")
            else:
                logger.debug(f"<--- _execute_tool RETURNING final result:\n{format_json_log(final_result) if isinstance(final_result, dict) else final_result!r}")

            # --- Log successful tool call ---
            if should_log_call:
                try:
                    caller_identity = "unknown_caller"
                    if user:
                        caller_identity = user
                    elif client_id:
                        caller_identity = f"client:{client_id}"

                    # Calculate elapsed time in milliseconds
                    call_end_datetime = datetime.datetime.now(datetime.timezone.utc)
                    elapsed_ms = round((call_end_datetime - call_start_datetime).total_seconds() * 1000, 2)

                    success_log_entry = {
                        "caller": caller_identity,
                        "tool_name": actual_function_name,
                        "app_name": parsed_app_name if parsed_app_name else "",
                        "timestamp": call_start_time,
                        "status": "success",
                        "elapsed_ms": elapsed_ms
                    }

                    os.makedirs(LOG_DIR, exist_ok=True)
                    with open(TOOL_CALL_LOG_PATH, "a", encoding="utf-8") as f:
                        json.dump(success_log_entry, f)
                        f.write("\n")
                except Exception as log_e:
                    logger.error(f"Failed to write success log to {TOOL_CALL_LOG_PATH}: {log_e}")

            return final_result

        except Exception as e:
            # Error already logged with full context at source, just continue with error logging to file
            # --- BEGIN TOOL ERROR LOGGING ---
            if should_log_call:
                try:
                    caller_identity = "unknown_caller"
                    if user:
                        caller_identity = user
                    elif client_id:
                        caller_identity = f"client:{client_id}"

                    # Calculate elapsed time in milliseconds
                    call_end_datetime = datetime.datetime.now(datetime.timezone.utc)
                    elapsed_ms = round((call_end_datetime - call_start_datetime).total_seconds() * 1000, 2)

                    error_log_entry = {
                        "caller": caller_identity,
                        "tool_name": actual_function_name, # Use parsed function name
                        "app_name": parsed_app_name if parsed_app_name else "", # Include app name
                        "timestamp": call_start_time,  # Use the call start time, not a new timestamp
                        "status": "error",
                        "error_message": str(e),
                        "elapsed_ms": elapsed_ms
                    }

                    os.makedirs(LOG_DIR, exist_ok=True)
                    with open(TOOL_CALL_LOG_PATH, "a", encoding="utf-8") as f:
                        json.dump(error_log_entry, f)
                        f.write("\n")

                    # --- ADDED: Log error to owner ---
                    try:
                        owner_to_log = atlantis._owner # Get current owner
                        if owner_to_log:
                            error_message_for_owner = f"Error in tool '{name}': {str(e)}"
                            # Construct a unique reference for this error instance if possible
                            error_ref = request_id if request_id else str(uuid.uuid4())[:8]
                            await atlantis.owner_log(error_message_for_owner)
                            logger.debug(f"Logged error for tool '{name}' to owner '{owner_to_log}'.")
                        else:
                            logger.warning(f"Could not log error for tool '{name}' to owner: Owner not set.")
                    except Exception as owner_log_e:
                        logger.error(f"CRITICAL: Failed to log ERROR to owner: {owner_log_e}")
                    # --- END Owner Log ---

                except Exception as log_e:
                    logger.error(f"CRITICAL: Failed to write ERROR to {TOOL_CALL_LOG_PATH}: {log_e}")
            # --- END TOOL ERROR LOGGING ---

            # Re-raise the exception so that ServiceClient._process_mcp_request can handle it
            # and construct a proper JSON-RPC error response.
            raise

    async def _handle_tools_call(self, params: dict, client_id: str, request_id: str, for_cloud: bool = False) -> dict:
        """Consolidated tools/call handler that works for both cloud and standard MCP clients"""

        # Extract required parameters
        tool_name = params.get("name")
        tool_args = params.get("arguments") # MCP spec uses 'arguments'

        # Extract optional context fields
        user = params.get("user", None)
        session_id = params.get("session_id", None)
        command_seq = params.get("command_seq", None)

        # Validate required parameters
        if tool_name is None or tool_args is None:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "Invalid params: missing tool name or arguments"}
            }

        # Log the call
        logger.info(f"🔧 Processing 'tools/call' for tool '{tool_name}' with args: {tool_args}")
        logger.debug(f"Tool name: '{tool_name}', Arguments:\n{format_json_log(tool_args)}")
        if user:
            logger.debug(f"Call made by user: {user}")
        if session_id:
            logger.debug(f"Call made with session_id: {session_id}")

        # Log the tool execution (don't re-register connections here)
        if for_cloud:
            logger.debug(f"☁️ Calling _execute_tool for: {tool_name}")
        else:
            logger.debug(f"🔧 Calling _execute_tool for: {tool_name}")

        # Verify client connection exists (should be registered by WebSocket or ServiceClient)
        global client_connections
        if client_id not in client_connections:
            logger.warning(f"⚠️ Client {client_id} not found in client_connections during tool call")
        else:
            connection_info = client_connections[client_id]
            logger.debug(f"✅ Found client {client_id} with type: {connection_info.get('type')}")

        try:
            # Execute the tool
            result_list = await self._execute_tool(
                name=tool_name,
                args=tool_args,
                client_id=client_id,
                request_id=request_id,
                user=user,
                session_id=session_id,
                command_seq=command_seq
            )

            logger.info(f"🎯 Tool '{tool_name}' execution completed with {len(result_list)} content items")

            # Format response based on client type
            if for_cloud:
                # Cloud client expects "contents" format
                try:
                    contents_list = []
                    for content in result_list:
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

                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"contents": contents_list}
                    }

                except Exception as e:
                    logger.error(f"❌ Error constructing cloud response: {e}")
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": "Internal server error during response formatting"}
                    }
            else:
                # Standard MCP client expects direct content format
                try:
                    contents_list = []
                    for item in result_list:
                        if hasattr(item, 'model_dump'):
                            contents_list.append(item.model_dump())
                        else:
                            logger.warning(f"⚠️ Non-pydantic result item: {type(item)}")
                            contents_list.append({"type": "text", "text": str(item)})

                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": contents_list}
                    }

                except Exception as e:
                    logger.error(f"❌ Error constructing MCP response: {e}")
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": "Internal server error during response formatting"}
                    }

        except Exception as e:
            logger.error(f"❌ Error executing tool '{tool_name}': {str(e)}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": f"Tool execution failed: {str(e)}"}
            }

    async def _notify_tool_list_changed(self, change_type: str, tool_name: str):
        """Send a 'notifications/tools/list_changed' notification with details to all connected clients."""
        logger.info(f"🔔 Notifying clients about tool list change ({change_type}: {tool_name})...")
        notification_params = {
            "changeType": change_type,
            "toolName": tool_name
        }
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed", # MCP spec format
            "params": notification_params # Include details in params
        }
        notification_json = json.dumps(notification)

        # Access global connection tracking
        global client_connections

        if not client_connections:
            logger.debug("No clients connected, skipping tool list change notification.")
            return

        # Deduplicate cloud connections to ensure we only notify each unique cloud connection once
        unique_connections = {}
        for client_id, info in client_connections.items():
            client_type = info.get("type")
            connection = info.get("connection")

            if client_type == "cloud":
                # Use connection object ID as key to deduplicate
                conn_id = id(connection)
                if conn_id not in unique_connections:
                    unique_connections[conn_id] = client_id
                    logger.debug(f"Using client_id {client_id} for cloud connection {conn_id}")
            else:
                # For non-cloud connections, just use the client_id directly
                unique_connections[client_id] = client_id

        logger.debug(f"Found {len(unique_connections)} unique connections from {len(client_connections)} client entries")

        # Iterate through the deduplicated client IDs
        client_ids = list(unique_connections.values())
        for client_id in client_ids:
            if client_id not in client_connections: # Check if client disconnected during iteration
                continue

            client_info = client_connections[client_id]
            client_type = client_info.get("type")
            connection = client_info.get("connection")

            if not connection:
                logger.warning(f"No connection object found for client {client_id}, skipping notification.")
                continue

            try:
                if client_type == "websocket":
                    await connection.send_text(notification_json)
                    logger.debug(f"📢 Sent notifications/tools/list_changed to WebSocket client: {client_id}")
                elif client_type == "cloud" and connection and hasattr(connection, 'is_connected') and connection.is_connected:
                    await connection.send_message('mcp_notification', notification)
                    logger.debug(f"☁️ Sent notifications/tools/list_changed to Cloud client: {client_id}")
                else:
                    logger.warning(f"Unknown or disconnected client type for {client_id}, skipping notification.")

            except Exception as e:
                logger.warning(f"Failed to send notifications/tools/list_changed notification to client {client_id}: {e}")
                # Consider removing the client connection if sending fails repeatedly?

async def get_all_tools_for_response(server: 'DynamicAdditionServer', caller_context: str) -> List[Dict[str, Any]]:
    """
    Fetches all tools from the server and prepares them as dictionaries for a JSON response.
    """
    logger.debug(f"Helper: Calling _get_tools_list for all tools from {caller_context}")
    raw_tool_list: List[Tool] = await server._get_tools_list(caller_context=caller_context)
    tools_dict_list: List[Dict[str, Any]] = []
    for tool in raw_tool_list:
        try:
            # Debug log to see all tools and their annotations BEFORE serialization
            #logger.debug(f"🔍 SERIALIZING TOOL '{tool.name}' with annotations: {getattr(tool, 'annotations', None)}")
            if hasattr(tool, 'annotations') and isinstance(tool.annotations, dict):
                source_file = tool.annotations.get('sourceFile', 'NOT_FOUND')
            elif hasattr(tool, 'annotations') and hasattr(tool.annotations, 'sourceFile'):
                source_file = getattr(tool.annotations, 'sourceFile', 'NOT_FOUND')

            # Ensure model_dump is called correctly for each tool
            tool_dict = tool.model_dump(mode='json') # Use mode='json' for better serialization

            # Debug log for all tools AFTER serialization
            annotations = tool_dict.get('annotations') if tool_dict else None
            if annotations and isinstance(annotations, dict):
                source_file = annotations.get('sourceFile', 'NOT_FOUND')

            if tool_dict and annotations and isinstance(annotations, dict) and annotations.get('type') == 'server':
                logger.debug(f"🔍 SERIALIZED SERVER TOOL '{tool_dict.get('name')}' to dict:\n{format_json_log(tool_dict)}")
                # Check if started_at is in annotations
                if annotations and 'started_at' in annotations:
                    logger.debug(f"✅ Started_at preserved in serialized tool dict for '{tool_dict.get('name')}': {annotations['started_at']}")

                    # If started_at is in annotations but not at the top level, add it to the top level
                    started_at_val = annotations['started_at']
                    if 'started_at' not in tool_dict:
                        logger.debug(f"🔎 Adding started_at to TOP LEVEL for '{tool_dict.get('name')}': {started_at_val}")
                        tool_dict['started_at'] = started_at_val
                else:
                    # This is normal for stopped servers
                    running_status = annotations.get('runningStatus', 'unknown')
                    logger.debug(f"ℹ️ MCP server '{tool_dict.get('name')}' not started (status: {running_status})")

            tools_dict_list.append(tool_dict)
        except Exception as e:
            error_msg = f"Error serializing tool: {e}"
            logger.error(f"❌ Error dumping tool model '{tool.name}' to dict: {e}", exc_info=True)

            # Create a placeholder that preserves as much original tool info as possible
            tool_error_dict = {
                "name": tool.name if hasattr(tool, 'name') else "unknown_tool",
                "description": tool.description if hasattr(tool, 'description') else "",
                "parameters": tool.parameters.model_dump() if hasattr(tool, 'parameters') and hasattr(tool.parameters, 'model_dump') else {"type": "object", "properties": {}},
            }

            # Add error information to annotations without changing other fields
            annotations = {}
            if hasattr(tool, 'annotations') and tool.annotations:
                # Try to preserve original annotations if possible
                try:
                    if hasattr(tool.annotations, 'model_dump'):
                        annotations = tool.annotations.model_dump()
                    elif isinstance(tool.annotations, dict):
                        annotations = tool.annotations.copy()
                except Exception:
                    pass  # If we can't get original annotations, use empty dict

            # Add error info to annotations
            annotations["error"] = True
            annotations["error_message"] = error_msg
            tool_error_dict["annotations"] = annotations
            tools_dict_list.append(tool_error_dict)  # Include the error info instead of skipping
    logger.debug(f"Helper: Prepared {len(tools_dict_list)} tool dictionaries.")
    return tools_dict_list

async def get_filtered_tools_for_response(server: 'DynamicAdditionServer', caller_context: str) -> List[Dict[str, Any]]:
    """
    Fetches tools, filters out server-type tools, and prepares them for a JSON response.
    """
    logger.debug(f"Helper: Calling get_all_tools_for_response for filtering from {caller_context}")
    all_tools_dict_list = await get_all_tools_for_response(server, caller_context)

    filtered_tools_dict_list: List[Dict[str, Any]] = []
    filtered_out_names: List[str] = []

    for tool_dict in all_tools_dict_list:
        # Check annotations safely
        annotations = tool_dict.get('annotations')
        if isinstance(annotations, dict) and annotations.get('type') == 'server':
            filtered_out_names.append(tool_dict.get('name', '<Unnamed Tool>'))
        else:
            filtered_tools_dict_list.append(tool_dict)

    if filtered_out_names:
        logger.info(f"🐾 Helper: Filtered out {len(filtered_out_names)} server-type tools from list requested by {caller_context}: {', '.join(filtered_out_names)}")

    logger.debug(f"Helper: Returning {len(filtered_tools_dict_list)} filtered tool dictionaries.")
    return filtered_tools_dict_list





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
        self.owner = None
        self.connection_active = True
        # Store creation time for stable client ID
        self._creation_time = int(time.time())

    # THIS IS THE BIG REPORT
    async def _report_tools_to_console(self, tools_list=None):
        """Generate and log a formatted report of all available tools

        Args:
            tools_list: Optional pre-fetched list of tools. If not provided, will fetch from mcp_server.
        """
        # Get the list of tools to log them
        if tools_list is None:
            tools_list = await self.mcp_server._get_tools_list(caller_context="_report_tools_to_console")
        # Create list with app names and source files for easier inspection of duplicates

        tool_info_list = []
        protected_info_list = []
        hidden_info_list = []
        server_info_list = []
        mcp_tools_list = []
        internal_info_list = []
        internal_count = 0
        for tool in tools_list:
            app_name = getattr(tool.annotations, 'app_name', None) if hasattr(tool, 'annotations') else None
            source_file = getattr(tool.annotations, 'sourceFile', 'unknown') if hasattr(tool, 'annotations') else 'unknown'
            app_source = getattr(tool.annotations, 'app_source', 'unknown') if hasattr(tool, 'annotations') else 'unknown'
            last_modified = getattr(tool.annotations, 'lastModified', None) if hasattr(tool, 'annotations') else None
            decorators = getattr(tool.annotations, 'decorators', []) if hasattr(tool, 'annotations') else []
            protection_name = getattr(tool.annotations, 'protection_name', None) if hasattr(tool, 'annotations') else None
            is_index = getattr(tool.annotations, 'is_index', False) if hasattr(tool, 'annotations') else False
            price_per_call = getattr(tool.annotations, 'price_per_call', None) if hasattr(tool, 'annotations') else None
            price_per_sec = getattr(tool.annotations, 'price_per_sec', None) if hasattr(tool, 'annotations') else None

            # Check if tool is hidden
            is_hidden = getattr(tool.annotations, 'temporarilyVisible', False) if hasattr(tool, 'annotations') else False
            tool_type = getattr(tool.annotations, 'type', None) if hasattr(tool, 'annotations') else None
            is_server = tool_type == 'server'  # Server entries are not callable tools
            is_mcp_tool = tool_type == 'server_tool'  # MCP tools from external servers

            # Check if this is an internal tool - check the tool name regardless of app_source
            is_internal = tool.name.startswith('_admin') or tool.name.startswith('_function') or tool.name.startswith('_server')

            if is_internal:
                internal_count += 1
                app_source = 'internal'
                app_display = 'internal'
            else:
                app_display = app_name if app_name else 'top-level'

            # Replace NA with [Nyaa] for top-level functions
            app_source_display = app_source
            if app_source == 'NA':
                app_source_display = 'Nyaa'

            # Color code based on app source
            if app_source == 'decorator':
                source_color = CYAN_COLOR
            elif app_source == 'directory':
                source_color = YELLOW
            elif app_source == 'internal':
                source_color = PINK
            else:
                source_color = GREY_COLOR

            # Format timestamp as relative time if available
            timestamp_str = ""
            if last_modified:
                try:
                    modified_dt = datetime.datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    relative_time = humanize.naturaltime(modified_dt)
                    timestamp_str = f" {GREY_COLOR}[{relative_time}]{RESET_COLOR}"
                except Exception as e:
                    timestamp_str = f" {GREY_COLOR}[{last_modified}]{RESET_COLOR}"

            # Detect if function is protected
            is_protected = 'protected' in decorators if decorators else False

            # Determine visibility indicator for non-internal functions
            visibility_str = ""
            if not is_internal and not is_server:
                if 'public' in decorators:
                    visibility_str = f" {CYAN_COLOR}[@public]{RESET_COLOR}"
                elif 'protected' in decorators:
                    # For protected functions, just show the group name without @protected wrapper
                    # since it's already in the "Protected" section
                    if protection_name:
                        visibility_str = f" {YELLOW}[{protection_name}]{RESET_COLOR}"
                    else:
                        visibility_str = f" {YELLOW}[@protected]{RESET_COLOR}"
                elif 'tick' in decorators:
                    visibility_str = f" {MAGENTA}[@tick]{RESET_COLOR}"
                elif 'visible' in decorators:
                    visibility_str = f" {GREY_COLOR}[@visible]{RESET_COLOR}"

            # Add index indicator if function is marked as index
            if is_index:
                visibility_str += f" {BRIGHT_WHITE}[@index]{RESET_COLOR}"

            # Add pricing information if available
            if price_per_call is not None and price_per_sec is not None:
                visibility_str += f" {CYAN_COLOR}[${price_per_call:.4f}/call ${price_per_sec:.4f}/sec]{RESET_COLOR}"

            # Format the line with colors
            # Format differently for servers (no app name column), MCP tools, internal tools, hidden, protected, or regular
            if is_server:
                formatted_line = f"{tool.name:40} {GREY_COLOR}{source_file:50}{RESET_COLOR}{timestamp_str}"
                server_info_list.append((app_display, tool.name, formatted_line))
            elif is_mcp_tool:
                # MCP tools from external servers - extract server name from tool name
                origin_server = getattr(tool.annotations, 'originServer', 'unknown') if hasattr(tool, 'annotations') else 'unknown'
                formatted_line = f"{CYAN_COLOR}{origin_server:30}{RESET_COLOR} {tool.name:40} {GREY_COLOR}{source_file:50}{RESET_COLOR}{timestamp_str}"
                mcp_tools_list.append((origin_server, tool.name, formatted_line))
            elif is_internal:
                formatted_line = f"{BOLD_COLOR}{app_display:30}{RESET_COLOR} {tool.name:40} {GREY_COLOR}{source_file:50}{RESET_COLOR} {source_color}[{app_source_display}]{RESET_COLOR}{timestamp_str}"
                internal_info_list.append((app_display, tool.name, formatted_line))
            elif is_hidden:
                formatted_line = f"{BOLD_COLOR}{app_display:30}{RESET_COLOR} {tool.name:40} {GREY_COLOR}{source_file:50}{RESET_COLOR} {source_color}[{app_source_display}]{RESET_COLOR}{visibility_str}{timestamp_str}"
                hidden_info_list.append((app_display, tool.name, formatted_line))
            elif is_protected:
                formatted_line = f"{BOLD_COLOR}{app_display:30}{RESET_COLOR} {tool.name:40} {GREY_COLOR}{source_file:50}{RESET_COLOR} {source_color}[{app_source_display}]{RESET_COLOR}{visibility_str}{timestamp_str}"
                protected_info_list.append((app_display, tool.name, formatted_line))
            else:
                formatted_line = f"{BOLD_COLOR}{app_display:30}{RESET_COLOR} {tool.name:40} {GREY_COLOR}{source_file:50}{RESET_COLOR} {source_color}[{app_source_display}]{RESET_COLOR}{visibility_str}{timestamp_str}"
                tool_info_list.append((app_display, tool.name, formatted_line))

        # Sort by app name then tool name (case-insensitive)
        tool_info_list.sort(key=lambda x: (x[0].lower(), x[1].lower()))
        protected_info_list.sort(key=lambda x: (x[0].lower(), x[1].lower()))
        hidden_info_list.sort(key=lambda x: (x[0].lower(), x[1].lower()))
        server_info_list.sort(key=lambda x: (x[0].lower(), x[1].lower()))
        mcp_tools_list.sort(key=lambda x: (x[0].lower(), x[1].lower()))
        internal_info_list.sort(key=lambda x: (x[0].lower(), x[1].lower()))

        # Calculate counts
        external_count = len(tools_list) - internal_count
        logger.info(f"📊 REGISTERING {len(tools_list)} TOOLS WITH CLOUD ({external_count} external + {internal_count} internal):")
        logger.info(f"")
        logger.info(f"  {BOLD_COLOR}Functions: {len(tool_info_list)}{RESET_COLOR}")
        for _, _, formatted_line in tool_info_list:
            logger.info(f"    {formatted_line}")

        if protected_info_list:
            logger.info(f"")
            logger.info(f"  {BOLD_COLOR}Protected: {len(protected_info_list)}{RESET_COLOR}")
            for _, _, formatted_line in protected_info_list:
                logger.info(f"    {formatted_line}")

        if mcp_tools_list:
            logger.info(f"")
            logger.info(f"  {BOLD_COLOR}MCP Tools: {len(mcp_tools_list)}{RESET_COLOR}")
            for _, _, formatted_line in mcp_tools_list:
                logger.info(f"    {formatted_line}")

        if internal_info_list:
            logger.info(f"")
            logger.info(f"  {BOLD_COLOR}Internal: {len(internal_info_list)}{RESET_COLOR}")
            for _, _, formatted_line in internal_info_list:
                logger.info(f"    {formatted_line}")

        if hidden_info_list:
            logger.info(f"")
            logger.info(f"  {BOLD_COLOR}Hidden Functions (temporarily visible): {len(hidden_info_list)}{RESET_COLOR}")
            for _, _, formatted_line in hidden_info_list:
                logger.info(f"    {formatted_line}")

        # Report skipped hidden functions (split into Hidden and Invalid/Errors)
        if hasattr(self.mcp_server.function_manager, '_skipped_hidden_functions') and self.mcp_server.function_manager._skipped_hidden_functions:
            # Separate hidden functions from invalid/error functions
            hidden_functions = []
            invalid_functions = []

            for item in self.mcp_server.function_manager._skipped_hidden_functions:
                # Check if this is an error condition vs. intentional hiding using explicit flag
                if item.get('is_error', False):
                    invalid_functions.append(item)
                else:
                    hidden_functions.append(item)

            # Report intentionally hidden functions
            if hidden_functions:
                logger.info(f"")
                logger.info(f"  {BOLD_COLOR}Hidden: {len(hidden_functions)}{RESET_COLOR}")
                for item in sorted(hidden_functions, key=lambda x: ((x['app'] or 'top-level').lower(), x['name'].lower())):
                    # Convert slash path to dot notation for display
                    app_display = self.mcp_server.function_manager._path_to_app_name(item['app']) if item['app'] else 'top-level'
                    logger.info(f"    {BOLD_COLOR}{app_display:30}{RESET_COLOR} {item['name']:40} {GREY_COLOR}{item['file']:50}{RESET_COLOR}")

            # Report invalid/error functions
            if invalid_functions:
                logger.info(f"")
                logger.error(f"  {BOLD_COLOR}❌ INVALID FUNCTIONS: {len(invalid_functions)}{RESET_COLOR}")
                for item in sorted(invalid_functions, key=lambda x: ((x['app'] or 'top-level').lower(), x['name'].lower())):
                    # Convert slash path to dot notation for display
                    app_display = self.mcp_server.function_manager._path_to_app_name(item['app']) if item['app'] else 'top-level'
                    reason = item.get('reason', 'unknown error')
                    logger.error(f"    {BOLD_COLOR}{app_display:30}{RESET_COLOR} {item['name']:40} {GREY_COLOR}{item['file']:50}{RESET_COLOR} {RED}[{reason}]{RESET_COLOR}")

            # Report duplicate functions
            duplicate_functions = self.mcp_server.function_manager._duplicate_functions
            if duplicate_functions:
                logger.info(f"")
                logger.error(f"  {BOLD_COLOR}❌ DUPLICATE FUNCTIONS (REMOVED): {len(duplicate_functions)}{RESET_COLOR}")
                for app_path, func_name, file_paths in sorted(duplicate_functions, key=lambda x: ((x[0] or 'top-level').lower(), x[1].lower())):
                    # Convert slash path to dot notation for display
                    app_display = self.mcp_server.function_manager._path_to_app_name(app_path) if app_path else 'top-level'
                    logger.error(f"    {BOLD_COLOR}{app_display:30}{RESET_COLOR} {func_name:40}")
                    for i, file_path in enumerate(file_paths, 1):
                        logger.error(f"      {GREY_COLOR}Occurrence {i}: {file_path}{RESET_COLOR}")

        if server_info_list:
            logger.info(f"")
            logger.info(f"  {BOLD_COLOR}MCP Servers: {len(server_info_list)}{RESET_COLOR}")
            for _, _, formatted_line in server_info_list:
                logger.info(f"    {formatted_line}")

        # App summary section
        app_counts = defaultdict(int)
        for app_display, _, _ in tool_info_list:
            app_counts[app_display] += 1
        for app_display, _, _ in protected_info_list:
            app_counts[app_display] += 1
        for app_display, _, _ in hidden_info_list:
            app_counts[app_display] += 1

        if app_counts:
            logger.info(f"")
            logger.info(f"  {BOLD_COLOR}Apps Summary:{RESET_COLOR}")
            for app_name in sorted(app_counts.keys(), key=str.lower):
                count = app_counts[app_name]
                logger.info(f"    {BOLD_COLOR}{app_name:30}{RESET_COLOR} {count} function(s)")

        logger.info(f" ")
        logger.info(f"- tools report done -")

    async def connect(self):
        """Establish a Socket.IO connection to the cloud server"""
        logger.info(f"☁️ CONNECTING TO CLOUD SERVER: {self.server_url} (namespace: {self.namespace})")
        self.connection_active = True
        self.connection_task = asyncio.create_task(self._maintain_connection())
        return self.connection_task

    async def _maintain_connection(self):
        """Maintains the connection to the cloud server with retries"""
        while self.connection_active and not is_shutting_down:
            logger.info("☁️ Starting _maintain_connection loop iteration") # DEBUG ADDED
            # logger.info(f"☁️ Socket.IO version: {socketio.__version__}") # Commented out for debugging freeze
            try:
                # Create a new Socket.IO client instance with increased buffer size
                # Engine.IO uses aiohttp for WebSocket connections, which has a default
                # max_msg_size of ~4MB. We need to increase this to handle large payloads
                # from the cloud server (e.g., file uploads, large tool responses).
                MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

                # Configure websocket options for aiohttp
                # This controls the maximum size of messages we can RECEIVE from the cloud
                websocket_options = {
                    'max_msg_size': MAX_MESSAGE_SIZE,  # Maximum message size in bytes (aiohttp parameter)
                    'timeout': 30.0,  # WebSocket timeout in seconds
                }

                self.sio = socketio.AsyncClient(
                    logger=False,  # Disable Socket.IO's own logger to avoid conflicts
                    engineio_logger=False,  # Disable engine.io logger
                    request_timeout=30,  # Increase timeout for large payloads (HTTP requests)
                    websocket_extra_options=websocket_options  # Pass aiohttp WebSocket options
                )

                logger.info(f"☁️ Created Socket.IO client with max_msg_size={MAX_MESSAGE_SIZE:,} bytes ({MAX_MESSAGE_SIZE/1024/1024:.1f} MB)")

                # Register event handlers
                self._register_event_handlers()

                logger.info(f"☁️ Attempting connection to cloud server (attempt {self.retry_count + 1})")

                # Connect with authentication data including hostname

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
                        "port": self.server_port, # Send the stored port
                        "serverVersion": SERVER_VERSION
                    },
                    retry=False # We handle retries manually with backoff
                )

                # Verify configuration after connection
                if hasattr(self.sio, 'eio') and self.sio.eio:
                    websocket_options_actual = getattr(self.sio.eio, 'websocket_extra_options', {})
                    logger.info(f"☁️ VERIFIED: websocket_extra_options = {websocket_options_actual}")
                else:
                    logger.warning(f"⚠️ Could not verify websocket configuration - eio attribute not found after connect")

                # Wait for disconnection
                await self.sio.wait()
                logger.info("☁️ Socket.IO connection closed (wait() returned)") # DEBUG ADDED

            except Exception as e:
                if is_shutting_down:
                    logger.info("☁️ Shutting down, stopping cloud connection attempts")
                    break

                self.is_connected = False
                self.sio = None

                detailed_error_info_message = "Check DEBUG logs for a full traceback."
                error_message = str(e)


                self.retry_count += 1

                # Enhanced error message logic
                if error_message == "One or more namespaces failed to connect":
                    if self.email and self.api_key:
                        specific_error_log = f"Authentication failed. Please make sure credentials are valid (email: {self.email}, API key, service name: {self.serviceName})"
                    else:
                        specific_error_log = f"Namespace '{self.namespace}' failed to connect. This could be due to server-side issues, incorrect namespace configuration, or missing authentication details if required by the server."
                elif isinstance(e, socketio.exceptions.ConnectionError):
                    specific_error_log = f"{error_message}. This often indicates a network issue, the cloud server at {self.server_url} being down, or a firewall blocking the connection."
                else:
                    specific_error_log = f"{error_message}. An unexpected error occurred during connection. {detailed_error_info_message}"

                logger.error(f"❌ EGAD!! ATLANTIS CLOUD SERVER CONNECTION ERROR (attempt {self.retry_count}): {specific_error_log}")

                # Print a more detailed stack trace for debugging
                logger.debug(f"Traceback: {traceback.format_exc()}")

                # Calculate exponential backoff delay with jitter
                backoff_delay = 5
                jitter = 5 * random.uniform(0, 1) # Add random jitter (0-1 seconds)
                actual_delay = min(backoff_delay + jitter, CLOUD_CONNECTION_MAX_BACKOFF_SECONDS)

                # Wait before retrying
                logger.info(f"☁️ RETRYING {self.email} CLOUD CONNECTION IN {actual_delay:.2f} SECONDS...")
                await asyncio.sleep(actual_delay)

        logger.info("☁️ Cloud connection giving up")

    def print_ascii_art(self, filepath):
        """
        Reads an ASCII art file and prints its content to the console.

        Args:
            filepath (str): The path to the ASCII art file.
        """
        try:
            with open(filepath, 'r') as file:
                ascii_art = file.read()
                print(ascii_art)
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _register_event_handlers(self):
        """Register Socket.IO event handlers"""
        if not self.sio:
            return

        # Catch-all handler to log ALL incoming events (fires before specific handlers)
        @self.sio.on('*', namespace=self.namespace)
        async def catch_all_events(event, *args):
            """Log all incoming Socket.IO events before they reach specific handlers"""
            try:
                # Calculate total size of all args
                total_size = sum(sys.getsizeof(arg) for arg in args)
                logger.info(f"☁️ SOCKETIO EVENT RECEIVED: '{event}' - Args count: {len(args)}, Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

                # Log size of each argument if large
                if total_size > 1024 * 1024:  # > 1MB
                    logger.warning(f"⚠️ LARGE SOCKETIO PAYLOAD DETECTED: {total_size/1024/1024:.2f} MB")
                    for i, arg in enumerate(args):
                        arg_size = sys.getsizeof(arg)
                        logger.info(f"  - Arg[{i}] type: {type(arg).__name__}, size: {arg_size:,} bytes ({arg_size/1024/1024:.2f} MB)")
            except Exception as e:
                logger.error(f"❌ Error in catch_all_events logger: {e}")

        @self.sio.event(namespace=self.namespace)
        async def connect():
            logger.info("Connected to server, waiting for welcome")
            self.is_connected = True
            # Emit the client event upon successful connection
            await self.send_message('client', {'status': 'connected'})

        # Connection established event
        @self.sio.event(namespace=self.namespace)
        async def welcome(username): # Ensure handler is async
            atlantis._set_owner(username) # Update atlantis module
            self.retry_count = 0  # Reset retry counter on successful connection

            self.print_ascii_art("../kitty.txt")

            logger.info("") # Blank line before
            logger.info(f"{BOLD}{CYAN}=================================================={RESET}")
            logger.info(f"{BOLD}{BRIGHT_WHITE}🚀✨🎉 CONNECTED TO ATLANTIS CLOUD SERVER! 🎉✨🚀{RESET}")
            logger.info(f"{BOLD}{CYAN}=================================================={RESET}")
            logger.info("") # Blank line after
            logger.info(f"{BOLD}{BRIGHT_WHITE}VERSION     : {SERVER_VERSION}{RESET}")
            logger.info(f"{BOLD}{BRIGHT_WHITE}CLOUD URL   : {self.server_url}{RESET}")
            logger.info(f"{BOLD}{BRIGHT_WHITE}REMOTE NAME : {self.serviceName}{RESET}")
            logger.info(f"{BOLD}{BRIGHT_WHITE}LOGIN       : {self.email}{RESET}")
            logger.info(f"{BOLD}{BRIGHT_WHITE}OWNER       : {atlantis._owner}{RESET}")
            logger.info(f"{BOLD}{BRIGHT_WHITE}LOCAL PORT  : {self.server_port}{RESET}")
            logger.info("") # Blank line after
            logger.info("") # Blank line after

            # --- ADDED: Register this connection with the MCP server ---
            cloud_sid = self.sio.sid if self.sio else 'unknown_sid' # Get Socket.IO session ID if available
            connection_id = f"service_{cloud_sid}"
            self.mcp_server.service_connections[connection_id] = {
                "type": "service",
                "connection": self.sio,
                "id": connection_id
            }
            logger.info(f"✅ Registered cloud service connection: {connection_id}")
            # -------------------------------------------------------------

            # Report tools to console
            await self._report_tools_to_console()


        # Connection error event
        @self.sio.event(namespace=self.namespace)
        def connect_error(data):
            logger.error(f"❌ DETAILED CLOUD CONNECTION FAILURE INFO: {data}") # Made log distinct

        # Disconnection event
        @self.sio.event(namespace=self.namespace)
        async def disconnect(): # Ensure handler is async
            logger.warning("⚠️ DISCONNECTED FROM CLOUD SERVER! (disconnect event)") # DEBUG ADDED
            self.is_connected = False

            # --- ADDED: Unregister this connection from the MCP server ---
            cloud_sid = self.sio.sid if self.sio else 'unknown_sid'
            connection_id = f"service_{cloud_sid}"
            removed = self.mcp_server.service_connections.pop(connection_id, None)
            if removed:
                logger.info(f"✅ Removed cloud service connection: {connection_id}")
            else:
                logger.warning(f"⚠️ Tried to remove dead cloud service connection: {connection_id}")
            # --------------------------------------------------------------

            # If disconnection was not intentional (e.g., server shutdown), try reconnecting
            if self.connection_active and not is_shutting_down:
                logger.info("☁️ Attempting to reconnect to cloud server...")
                # The _maintain_connection loop will handle the retry logic
            else:
                logger.info("☁️ Disconnection was expected or shutdown initiated, not reconnecting.")


        # Service message event
        @self.sio.event(namespace=self.namespace)
        async def service_message(data):
            logger.info(f"☁️ RAW RECEIVED SERVICE MESSAGE")
            #logger.debug(f"☁️ RAW RECEIVED SERVICE MESSAGE:\n{format_json_log(data) if isinstance(data, dict) else data}")

            # --- Handle Awaitable Command Responses from Cloud Client ---
            if isinstance(data, dict) and \
               data.get("method") == "atlantis/commandResult" and \
               isinstance(data.get("params"), dict):

                params = data["params"]
                correlation_id = params.get("correlationId")

                # Ensure self.mcp_server and awaitable_requests exist
                if hasattr(self.mcp_server, 'awaitable_requests') and \
                   correlation_id and correlation_id in self.mcp_server.awaitable_requests:

                    future = self.mcp_server.awaitable_requests.pop(correlation_id, None)
                    if future and not future.done():
                        if "result" in params:
                            logger.info(f"✅☁️ Received cloud result for awaitable command (correlationId: {correlation_id})")
                            future.set_result(params["result"])
                        elif "error" in params:
                            client_error_details = params["error"] # This could be a string from the cloud
                            logger.error(f"❌☁️ Received cloud error for awaitable command (correlationId: {correlation_id}): {client_error_details}")
                            if isinstance(client_error_details, Exception):
                                future.set_exception(McpError(client_error_details))
                            elif isinstance(client_error_details, dict) and "message" in client_error_details:
                                future.set_exception(McpError(client_error_details.get("message", "Unknown cloud error")))
                            else: # Handle string error or other non-Exception types
                                future.set_exception(Exception(f"{str(client_error_details)}"))
                        else:
                            # treat as result
                            future.set_result(None)
                        logger.debug(f"📥☁️ Handled atlantis/commandResult for {correlation_id} from cloud. Returning from service_message.")
                        return # IMPORTANT: Return early, this message is handled.
                    elif future and future.done():
                        logger.warning(f"⚠️☁️ Received cloud commandResult for {correlation_id}, but future was already done. Ignoring.")
                        return
                    else: # Future not found in pop (e.g. already timed out and removed)
                        logger.warning(f"⚠️☁️ Received cloud commandResult for {correlation_id}, but no active future found (pop returned None). Might have timed out. Ignoring.")
                        return
                else:
                    logger.warning(f"⚠️☁️ Received cloud atlantis/commandResult with missing, invalid, or non-pending correlationId: '{correlation_id}'. It will be passed to standard MCP processing if not caught by other logic.")
            # --- End Awaitable Command Response Handling ---

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
        """Process an MCP JSON-RPC request from the cloud server by manually routing
        to the appropriate logic method in the DynamicAdditionServer.
        """
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        logger.warning(f"🔴 CLOUD PATH: {method}")

        # Use persistent cloud client ID for this connection
        client_id = f"cloud_{self._creation_time}_{id(self)}"
        logger.debug(f"Created persistent cloud client ID: {client_id}")

        # Register this cloud client in client_connections (missing registration!)
        global client_connections
        client_connections[client_id] = {"type": "cloud", "connection": self}
        logger.debug(f"☁️ Registered cloud client in client_connections: {client_id}")

        logger.info(f"☁️ Processing MCP request via manual routing: {method} (ID: {request_id})")

        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }

        try:

            if method == "tools/list":
                logger.info(f"🧰 Processing 'tools/list' request via helper")
                filtered_tools_dict_list = await get_filtered_tools_for_response(self.mcp_server, caller_context="process_mcp_request_websocket")
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": filtered_tools_dict_list}
                }
                logger.debug(f"📦 Prepared tools/list response (ID: {request_id}) with {len(filtered_tools_dict_list)} tools.")
                return response
            elif method == "tools/list_all": # Handling for list_all in direct connections
                # get all tools including internal
                # get servers including those not running
                # Call the core logic method directly (pass client_id)
                logger.info(f"🧰 Processing 'tools/list_all' request via helper")
                all_tools_dict_list = await get_all_tools_for_response(self.mcp_server, caller_context="process_mcp_request_websocket")
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": all_tools_dict_list}
                }
                return response

            elif method == "tools/call":
                # Use consolidated handler for cloud clients
                return await self.mcp_server._handle_tools_call(
                    params=params,
                    client_id=client_id,
                    request_id=request_id,
                    for_cloud=True
                )

            else:
                # Unknown method
                response["error"] = {"code": -32601, "message": f"Method not found: {method}"}

            return response

        except Exception as e:
            logger.error(f"❌ ERROR PROCESSING MCP REQUEST: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            response["error"] = {"code": -32000, "message": str(e)}
            return response

    async def send_message(self, event: str, data: dict):
        """Send a message to the cloud server via a named event.
        Raises McpError if sending fails.
        """
        if not self.sio or not self.is_connected:
            # This check is also present in the caller (send_awaitable_client_command for cloud),
            # but good to have defense in depth. More importantly, this ensures this method's
            # contract is to raise on failure, not return False.
            logger.warning(f"⚠️ ServiceClient: Attempted to send event '{event}' but not connected.")
            error_data = ErrorData(
                code=-32000,
                message=f"ServiceClient not connected. Cannot send event '{event}'."
            )
            raise McpError(error_data)

        try:
            # Special handling for 'notifications/message' which is how client logs/commands are sent to cloud
            if event == 'mcp_notification' and data.get('method') == 'notifications/message':
                # The actual Socket.IO event for the cloud service is 'mcp_notification'
                # The 'data' payload already contains the 'notifications/message' structure.
                pass # emit_event is already 'mcp_notification'

            if data.get('method') == 'notifications/message':
                pass
                #logger.info(f"☁️ SENDING CLIENT LOG/COMMAND via {event}: {data.get('params', {}).get('command', data.get('method'))}")
            else:
                logger.debug(f"☁️ SENDING MCP MESSAGE via {event}: {data.get('method')}")

            await self.sio.emit(event, data, namespace=self.namespace)
            # If emit succeeds, we don't return True anymore; success is implied by no exception.

        except socketio.exceptions.SocketIOError as e:
            # Catch specific Socket.IO errors during emit (e.g., BadNamespaceError, ConnectionError if not caught by 'is_connected')
            logger.error(f"❌ ServiceClient: Socket.IO error sending event '{event}': {str(e)}")
            error_data = ErrorData(
                code=-32000,
                message=f"ServiceClient: Socket.IO error sending event '{event}': {str(e)}"
            )
            raise McpError(error_data) from e
        except Exception as e:
            # Catch any other unexpected errors during emit
            logger.error(f"❌ ServiceClient: Unexpected error sending event '{event}': {str(e)}")
            error_data = ErrorData(
                code=-32000,
                message=f"ServiceClient: Unexpected error sending event '{event}': {str(e)}"
            )
            raise McpError(error_data) from e

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

# Global collection to track active websocket connections
active_websockets = set()

# Global dictionary to track client connections by ID
client_connections = {}

# Global dictionary to store dynamically registered clients (loaded from file)
REGISTERED_CLIENTS: Dict[str, Dict[str, Any]] = {}
CLIENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registered_clients.json")

# --- ADDED Persistence Functions ---
def _load_registered_clients():
    """Load registered clients from the JSON file into memory."""
    global REGISTERED_CLIENTS
    if os.path.exists(CLIENTS_FILE):
        try:
            with open(CLIENTS_FILE, 'r') as f:
                REGISTERED_CLIENTS = json.load(f)
                logger.info(f"💾 Loaded {len(REGISTERED_CLIENTS)} registered clients from {CLIENTS_FILE}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"❌ Error loading {CLIENTS_FILE}: {e}. Starting with empty registrations.")
            REGISTERED_CLIENTS = {}
    else:
        logger.info(f"ℹ️ Client registration file not found ({CLIENTS_FILE}). Starting fresh.")
        REGISTERED_CLIENTS = {}

def _save_registered_clients():
    """Save the current registered clients dictionary to the JSON file."""
    try:
        with open(CLIENTS_FILE, 'w') as f:
            json.dump(REGISTERED_CLIENTS, f, indent=2) # Use indent for readability
            logger.debug(f"💾 Saved {len(REGISTERED_CLIENTS)} registered clients to {CLIENTS_FILE}") # DEBUG level
    except IOError as e:
        logger.error(f"❌ Error saving registered clients to {CLIENTS_FILE}: {e}")
# --- End Persistence Functions ---

# --- ADDED BACK Global MCP Server Instantiation ---
mcp_server = DynamicAdditionServer()
# Set the server instance in utils immediately so client_log() works
utils.set_server_instance(mcp_server)

# Custom WebSocket handler for the MCP server
async def handle_websocket(websocket: WebSocket):
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
                request_data = json.loads(message)

                # --- Handle Awaitable Command Responses ---
                if isinstance(request_data, dict) and \
                   request_data.get("method") == "atlantis/commandResult" and \
                   "params" in request_data:

                    params = request_data["params"]
                    correlation_id = params.get("correlationId")

                    # Ensure self.mcp_server and awaitable_requests exist
                    if hasattr(mcp_server, 'awaitable_requests') and \
                       correlation_id and correlation_id in mcp_server.awaitable_requests:

                        future = mcp_server.awaitable_requests.pop(correlation_id, None)
                        if future and not future.done():
                            if "result" in params:
                                logger.info(f"✅ Received result for awaitable command (correlationId: {correlation_id})")
                                future.set_result(params["result"])
                            elif "error" in params:
                                client_error_details = params["error"]
                                logger.error(f"❌ Received error from client for awaitable command (correlationId: {correlation_id}): {client_error_details}")
                                future.set_exception(McpError(f"Client error for command (correlationId: {correlation_id}): {client_error_details}"))
                            else:
                                # treat as result
                                future.set_result(None)
                            # This message is handled, continue to next message in the loop
                            logger.debug(f"📥 Handled atlantis/commandResult for {correlation_id}, continuing WebSocket loop.")
                            continue
                        elif future and future.done():
                            # Future was already done (e.g., timed out and handled by send_awaitable_client_command)
                            logger.warning(f"⚠️ Received commandResult for {correlation_id}, but future was already done. Ignoring.")
                            continue
                        else:
                            # Future not found in pop, might have timed out and been removed by send_awaitable_client_command's timeout logic
                            logger.warning(f"⚠️ Received commandResult for {correlation_id}, but no active future found (pop returned None). Might have timed out. Ignoring.")
                            continue
                    else:
                        # No correlationId or not in awaitable_requests, could be a stray message or an issue.
                        logger.warning(f"⚠️ Received atlantis/commandResult without a valid/pending correlationId: '{correlation_id}'. Passing to standard processing just in case, but this is unusual.")
                # --- End Awaitable Command Response Handling ---

                logger.debug(f"📥 Received (for MCP processing): {request_data}")
                logger.warning(f"🟡 WEBSOCKET→process_mcp_request: {request_data.get('method')}")
                # Process the request using our MCP server (include client_id)
                response = await process_mcp_request(mcp_server, request_data, client_id)

                # Send the response back to the client
                #logger.debug(f"📤 Sending: {response}")
                logger.debug(f"📤 Sending response")
                await websocket.send_text(json.dumps(response))

            except json.JSONDecodeError:
                logger.error(f"🚫 Invalid JSON received: {message}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                logger.error(f"🚫 Error processing request: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect as e:
        logger.info(f"☑️ WebSocket client disconnected normally: code={e.code}, reason={e.reason}")
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

    method = request.get("method")
    logger.warning(f"🟢 WS/HTTP PATH: {method}")

    if "id" not in request:
        return {"error": "Missing request ID"}

    req_id = request.get("id")
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
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        elif method == "tools/list":
            logger.info(f"🧰 Processing 'tools/list' request via helper")
            filtered_tools_dict_list = await get_filtered_tools_for_response(server, caller_context="process_mcp_request_websocket")
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": filtered_tools_dict_list}
            }
            logger.debug(f"📦 Prepared tools/list response (ID: {req_id}) with {len(filtered_tools_dict_list)} tools.")
            return response
        elif method == "tools/list_all": # Handling for list_all in direct connections
            # get all tools including internal
            # get servers including those not running
            # Call the core logic method directly (pass client_id)
            logger.info(f"🧰 Processing 'tools/list_all' request via helper")
            all_tools_dict_list = await get_all_tools_for_response(server, caller_context="process_mcp_request_websocket")
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": all_tools_dict_list}
            }
            return response

        elif method == "prompts/list":
            result = await server._get_prompts_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": result}}
        elif method == "resources/list":
            result = await server._get_resources_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": result}}
        elif method == "tools/call":
            # Use consolidated handler for standard MCP clients
            return await server._handle_tools_call(
                params=params,
                client_id=client_id,
                request_id=req_id,
                for_cloud=False
            )
        else:
            logger.warning(f"⚠️ Unknown method requested: {method}")
            return {"jsonrpc": "2.0", "id": req_id, "error": f"Unknown method: {method}"}
    except Exception as e:

        logger.error(f"🚫 Error processing request '{method}': {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {"jsonrpc": "2.0", "id": req_id, "error": f"Error processing request: {e}"}

# Set up the Starlette application with routes
async def handle_registration(request: Request) -> JSONResponse:
    """Handle dynamic client registration requests (POST /register).

    Expects JSON body with at least 'client_name' and 'redirect_uris'.
    Generates client_id and client_secret.
    Persists client data to file.
    Based on RFC 7591.
    """
    try:
        client_metadata = await request.json()
        logger.info(f"🔑 Received registration request: {client_metadata}")

        # --- Enhanced Validation ---
        client_name = client_metadata.get("client_name")
        redirect_uris = client_metadata.get("redirect_uris")

        if not client_name:
            return JSONResponse({"error": "invalid_client_metadata", "error_description": "Missing 'client_name'"}, status_code=400)
        if not redirect_uris or not isinstance(redirect_uris, list) or not all(isinstance(uri, str) for uri in redirect_uris):
            return JSONResponse({"error": "invalid_redirect_uri", "error_description": "'redirect_uris' must be a non-empty array of strings"}, status_code=400)
        # Add more validation for other DCR params (grant_types, response_types, scope etc.) if needed

        # --- Client Creation ---
        client_id = str(uuid.uuid4())
        issued_at = int(datetime.datetime.utcnow().timestamp()) # Use timestamp
        client_secret = secrets.token_urlsafe(32) # Generate client secret

        # --- Store Client Details ---
        # Store all provided valid metadata
        registered_data = {
            "client_id": client_id,
            "client_name": client_name,
            "redirect_uris": redirect_uris,
            "client_id_issued_at": issued_at,
            "client_secret": client_secret, # Store client secret
            "client_secret_expires_at": 0, # 0 means never expires, or set a timestamp
            # Store other validated DCR fields here (grant_types, response_types, scope etc.)
            # "token_endpoint_auth_method": client_metadata.get("token_endpoint_auth_method", "client_secret_basic") # Default or from request
        }
        REGISTERED_CLIENTS[client_id] = registered_data
        _save_registered_clients() # Save after modification

        logger.info(f"✅ Registered new client: ID={client_id}, Name='{client_name}', URIs={redirect_uris}")


        # Return the registered client metadata (INCLUDING secret for M2M simplicity for now)
        response_data = registered_data.copy()
        # Consider *not* returning the secret in production for higher security

        return JSONResponse(response_data, status_code=201) # 201 Created

    except json.JSONDecodeError:
        logger.error("❌ Registration failed: Invalid JSON in request body")
        return JSONResponse({"error": "invalid_client_metadata", "error_description": "Invalid JSON format"}, status_code=400)
    except Exception as e:
        logger.error(f"❌ Registration failed: {str(e)}", exc_info=True)
        return JSONResponse({"error": "internal_server_error", "error_description": "Internal server error during registration"}, status_code=500)

async def handle_mcp_http(request: Request) -> Response:
    """Handle MCP requests over HTTP (POST /mcp).

    This endpoint accepts MCP JSON-RPC requests and processes them using the same
    logic as the WebSocket handler, but returns HTTP responses.
    """
    try:
        # Only accept POST requests
        if request.method != "POST":
            return JSONResponse(
                {"error": "method_not_allowed", "message": "Only POST method is supported for MCP HTTP endpoint"},
                status_code=405
            )

        # Parse the JSON-RPC request
        try:
            request_data = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": "invalid_json", "message": "Invalid JSON in request body"},
                status_code=400
            )

        # Validate JSON-RPC format
        if not isinstance(request_data, dict):
            return JSONResponse(
                {"error": "invalid_request", "message": "Request must be a JSON object"},
                status_code=400
            )

        if request_data.get("jsonrpc") != "2.0":
            return JSONResponse(
                {"error": "invalid_request", "message": "Invalid JSON-RPC version"},
                status_code=400
            )

        # Generate a client ID for HTTP requests (using request IP and user agent)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        client_id = f"http_{client_ip}_{hash(user_agent) % 10000}"

        logger.info(f"🌐 HTTP MCP request from {client_id}: {request_data.get('method', 'unknown')}")
        logger.warning(f"🟠 HTTP→process_mcp_request: {request_data.get('method')}")

        # Process the MCP request using the same logic as WebSocket
        response_data = await process_mcp_request(mcp_server, request_data, client_id)

        if response_data:
            return JSONResponse(response_data)
        else:
            # For notifications (no response expected)
            return Response(status_code=204)

    except Exception as e:
        logger.error(f"❌ Error handling HTTP MCP request: {str(e)}", exc_info=True)
        return JSONResponse(
            {"error": "internal_server_error", "message": "Internal server error"},
            status_code=500
        )

async def handle_health_check(request: Request) -> JSONResponse:
    """Simple health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "version": SERVER_VERSION,
        "endpoints": {
            "websocket": "/mcp",
            "http": "/mcp",
            "health": "/health",
            "register": "/register"
        }
    })

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Override the max body size for large file uploads
        original_receive = request.receive

        async def custom_receive():
            message = await original_receive()
            # Allow up to 100MB body size
            if message['type'] == 'http.request' and 'body' in message:
                return message
            return message

        request.receive = custom_receive
        return await call_next(request)

app = Starlette(
    routes=[
        WebSocketRoute("/mcp", endpoint=handle_websocket),
        Route("/register", endpoint=handle_registration, methods=["POST"]),
        Route("/mcp", endpoint=handle_mcp_http, methods=["POST"]),
        Route("/health", endpoint=handle_health_check, methods=["GET"])
    ],
    middleware=[
        Middleware(RequestSizeLimitMiddleware)
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
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Set the logging level")
    args = parser.parse_args()

    # Update host and port from command line arguments
    HOST = args.host
    PORT = args.port

    # Update the logging level from command line argument
    from state import update_log_level
    update_log_level(args.log_level)

    # Initialize PID Manager with service name if provided
    pid_manager = PIDManager(service_name=args.service_name)

    # Check if server is already running
    existing_pid = pid_manager.check_server_running()
    if existing_pid:
        logger.error(f"❌ Server already running with PID {existing_pid}! Exiting...")
        sys.exit(1)

    # Create PID file
    if not pid_manager.create_pid_file():
        logger.error("❌ Failed to create PID file! Exiting...")
        sys.exit(1)

    _load_registered_clients() # Load clients at startup

    # Update cloud server settings if provided
    if args.cloud_host != CLOUD_SERVER_HOST or args.cloud_port != CLOUD_SERVER_PORT:
        CLOUD_SERVER_HOST = args.cloud_host
        CLOUD_SERVER_PORT = args.cloud_port
        CLOUD_SERVER_URL = f"{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"

    # Set up the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    logger.info(f"")
    logger.info(f"{BRIGHT_WHITE}================{RESET}")
    logger.info(f"{BRIGHT_WHITE}REMOTE MCP {SERVER_VERSION}{RESET}")
    logger.info(f"{BRIGHT_WHITE}================{RESET}")
    logger.info(f"")

    # Initialize the MCP server
    logger.info(f"{BRIGHT_WHITE}🔧 === CALLING SERVER INITIALIZE FROM MAIN ==={RESET}")
    loop.run_until_complete(mcp_server.initialize())

    # Ensure dynamic directories exist
    os.makedirs(FUNCTIONS_DIR, exist_ok=True)
    logger.info(f"📁 Dynamic functions directory: {FUNCTIONS_DIR}")
    os.makedirs(SERVERS_DIR, exist_ok=True)
    logger.info(f"📁 Dynamic servers directory: {SERVERS_DIR}")

    # Start the file watcher - collect all watched directories first
    watched_function_dirs = [FUNCTIONS_DIR]
    watched_server_dirs = [SERVERS_DIR]

    # Find all symlinked directories within FUNCTIONS_DIR (recursively search for all symlinks)
    try:
        for root, dirs, files in os.walk(FUNCTIONS_DIR):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if os.path.islink(dir_path):
                    resolved_path = os.path.realpath(dir_path)
                    if os.path.isdir(resolved_path):
                        watched_function_dirs.append(resolved_path)
                        relative_path = os.path.relpath(dir_path, FUNCTIONS_DIR)
                        logger.info(f"👁️ Found symlinked directory: {relative_path} -> {resolved_path}")
    except Exception as e:
        logger.warning(f"⚠️ Error finding symlink directories: {e}")

    # Create event handler with all watched directories
    event_handler = DynamicConfigEventHandler(mcp_server, loop, watched_function_dirs, watched_server_dirs)
    observer = Observer()

    # Schedule watchers for all directories
    observer.schedule(event_handler, FUNCTIONS_DIR, recursive=True)
    for resolved_path in watched_function_dirs[1:]:  # Skip FUNCTIONS_DIR since already scheduled
        observer.schedule(event_handler, resolved_path, recursive=True)
        logger.info(f"👁️ Watching symlinked directory: {resolved_path}")

    observer.schedule(event_handler, SERVERS_DIR, recursive=False)
    observer.start()
    logger.info(f"👁️ Watching for changes in {FUNCTIONS_DIR} and {SERVERS_DIR}...")

    # Start the Uvicorn server with increased websocket message size limits
    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",
        ws_max_size=100 * 1024 * 1024,  # 100MB max websocket message size
        ws_ping_interval=60,           # Ping interval in seconds
        ws_ping_timeout=120,           # Ping timeout in seconds
        limit_max_requests=1000,       # Max concurrent requests
        limit_concurrency=1000,        # Max concurrent connections
        h11_max_incomplete_event_size=100 * 1024 * 1024  # 100MB max HTTP body size
    )
    server = uvicorn.Server(config)

    # Create tasks for the server and cloud connection
    server_task = loop.create_task(server.serve())
    cloud_task = None

    try:
        # Start the server
        logger.info(f"🌟 STARTING LOCAL MCP SERVER AT {HOST}:{PORT}")
        logger.info(f"   📡 WebSocket endpoint: ws://{HOST}:{PORT}/mcp")
        logger.info(f"   🌐 HTTP endpoint: http://{HOST}:{PORT}/mcp")
        logger.info(f"   ❤️  Health check: http://{HOST}:{PORT}/health")

        # Connect to cloud server if enabled
        if not args.no_cloud:
            if not args.email or not args.api_key or not args.service_name:
                logger.error("❌ CLOUD SERVER CONNECTION REQUIRES EMAIL, API KEY AND SERVICE NAME")
                logger.error("❌ Use --email and --api_key to specify credentials, --service-name to specify desired service name")
                logger.info("☁️ CLOUD SERVER CONNECTION DISABLED")
            else:
                logger.info(f"{PINK}☁️ CLOUD SERVER CONNECTION ENABLED: {CLOUD_SERVER_URL}{RESET}")
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
                # Store cloud client reference on server for tool reporting
                mcp_server.cloud_client = cloud_connection
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
        pid_manager.remove_pid_file() # Ensure PID file is removed on exit
        # Use the service_name from pid_manager, or default to 'MCP' if not available
        service_name = pid_manager.service_name if pid_manager.service_name else 'MCP'
        logger.info(f"👋 SERVER '{service_name}' SHUTDOWN COMPLETE")
