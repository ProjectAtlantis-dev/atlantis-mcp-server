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
from utils import check_server_running, create_pid_file, remove_pid_file
from typing import Any, Callable, Dict, List, Optional, Union
import datetime

from mcp.server import Server
from mcp.server.websocket import websocket_server
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
    dynamic_functions, tools, tasks
)

# Import dynamic functions
from dynamic import (
    discover_functions, extract_metadata_from_file, load_function_from_file,
    function_register, get_function_code, function_remove, function_add
)

# Import task functions
from task import task_add, task_run, task_remove, task_peek

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

# Create an MCP server with proper MCP protocol handling
class DynamicAdditionServer(Server):
    """MCP server that provides an addition tool and supports dynamic function registration"""

    def __init__(self):
        super().__init__("Dynamic Function Server")
        
        # Register tool handlers using SDK decorators
        # These now wrap the actual logic methods defined below

        @self.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """SDK Handler for tools/list"""
            return await self._get_tools_list()

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
    async def initialize(self):
        """Initialize the server by discovering available functions"""
        # Scan the functions directory to discover available functions
        await discover_functions(self)

    # --- Core Logic Methods (callable directly) ---

    async def _get_tools_list(self) -> list[Tool]:
        """Core logic to return a list of available tools"""
        logger.info("📋 TOOLS LIST LOGIC EXECUTED")

        # Start with our built-in tools
        tools_list = [
            Tool(
                name="_function_register",
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
            # --- Task Management Tools --- #
            Tool(
                name="_task_add",
                description="Adds a task with the provided payload data.",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "payload": {
                            "type": "object",
                            "description": "The task payload object. Must include 'functionName' and 'arguments'."
                        }
                    },
                    "required": ["payload"]
                }
            ),
            Tool(
                name="_task_run",
                description="Runs a task with the specified ID and stores the result.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "The ID of the task to run"}
                    },
                    "required": ["id"]
                }
            ),
            Tool(
                name="_task_remove",
                description="Removes a task by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "The ID of the task to remove"}
                    },
                    "required": ["id"]
                }
            ),
            Tool(
                name="_task_peek",
                description="Gets the details of a task by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "The ID of the task to get details for"}
                    },
                    "required": ["id"]
                }
            ),
        ]

        # Add dynamically registered tools
        for name, tool_obj in tools.items():
            tools_list.append(tool_obj)

        logger.info(f"📋 RETURNING {len(tools_list)} TOOLS")
        return tools_list

    async def _get_prompts_list(self) -> list:
        """Core logic to return a list of available prompts (empty stub)"""
        # Currently no prompts supported
        return []

    async def _get_resources_list(self) -> list:
        """Core logic to return a list of available resources (empty stub)"""
        # Currently no resources supported
        return []

    async def _execute_tool(self, name: str, args: dict) -> list[TextContent]:
        """Core logic to handle a tool call. Ensures result is List[TextContent(type='text')]"""
        logger.info(f"🔧 EXECUTING TOOL: {name}")
        logger.debug(f"WITH ARGUMENTS: {args}")

        try:
            # Handle built-in tool calls
            if name == "_function_register":
                return await function_register(args, self)
            elif name == "_function_get":
                return await get_function_code(args, self)
            elif name == "_function_remove":
                return await function_remove(args, self)
            elif name == "_function_add":
                return await function_add(args, self)
            # Task management
            elif name == "_task_add":
                return await task_add(args)
            elif name == "_task_run":
                return await task_run(args)
            elif name == "_task_remove":
                return await task_remove(args)
            elif name == "_task_peek":
                return await task_peek(args)
            # Handle dynamic function calls
            elif name in dynamic_functions:
                # Call the dynamic function with the arguments
                logger.info(f"🔧 CALLING DYNAMIC FUNCTION: {name}")
                result = await dynamic_functions[name](args)
                return result
            else:
                # Unknown tool
                error_message = f"Unknown tool: {name}"
                logger.error(f"❌ {error_message}")
                raise ValueError(error_message)

        except Exception as e:
            logger.error(f"❌ TOOL EXECUTION ERROR: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Re-raise the exception with its original message
            raise type(e)(str(e))

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
            tools_list = await self.mcp_server._get_tools_list()
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
                # Call the core logic method directly
                result_list = await self.mcp_server._get_tools_list()
                # Convert Tool objects to dictionaries for JSON serialization using model_dump()
                response["result"] = {"tools": [tool.model_dump() for tool in result_list]} # Use model_dump()

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments") # MCP spec uses 'arguments'

                if tool_name is None or tool_args is None:
                    response["error"] = {"code": -32602, "message": "Invalid params: missing tool name or arguments"}
                else:
                    logger.debug(f"☁️ Calling _execute_tool for: {tool_name}")
                    # Call the core logic method directly
                    call_result_list = await self.mcp_server._execute_tool(name=tool_name, args=tool_args)
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
            logger.debug(f"☁️ SENDING MESSAGE: {event} - {data}")
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

# WebSocket handler for the MCP server
async def handle_websocket(websocket):
    await websocket_server(websocket, mcp_server)

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
