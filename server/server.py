#!/usr/bin/env python3
import logging
import json
import inspect
import asyncio
import os
import importlib
import importlib.util
import re
import signal
import sys
import psutil
import time
from typing import Any, Callable, Dict, List, Optional, Union

# NOTE: This server uses two different socket protocols:
# 1. Standard WebSockets: When acting as a SERVER to accept connections from node-mcp-client
# 2. Socket.IO: When acting as a CLIENT to connect to the cloud Node.js server
#
# Each server dictates its own protocol, and clients must adapt accordingly.
# - The node-mcp-client connects via standard WebSockets to our server.py
# - Our server.py connects via Socket.IO to the cloud server
# - Both ultimately route to the same MCP handlers in the DynamicAdditionServer class
import socketio
from mcp.server import Server
from mcp.server.websocket import websocket_server
from mcp.client.websocket import websocket_client
from mcp.types import Tool, TextContent, CallToolResult
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
import uvicorn
import argparse

# Configure logging - focus on our application logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
# Set our app logger to DEBUG
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.DEBUG)

# Directory to store dynamic function files
FUNCTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_functions")

# Path for the PID file
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.pid")

# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces by default
PORT = 8000

# Cloud server configuration
CLOUD_SERVER_HOST = "localhost"
CLOUD_SERVER_PORT = 3010
CLOUD_SERVER_URL = f"http://{CLOUD_SERVER_HOST}:{CLOUD_SERVER_PORT}"
CLOUD_SERVICE_NAMESPACE = "/service"  # Socket.IO namespace for service-to-service communication
CLOUD_CONNECTION_RETRY_SECONDS = 5
CLOUD_CONNECTION_MAX_RETRIES = 5  # Set to None for infinite retries


# Create functions directory if it doesn't exist
os.makedirs(FUNCTIONS_DIR, exist_ok=True)

# Flags to track server state
is_shutting_down = False
cloud_connection_active = False

# Check if a server is already running
def check_server_running():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())

            # Check if the process with this PID exists
            if psutil.pid_exists(pid):
                # Get process name to confirm it's our server
                process = psutil.Process(pid)
                if "python" in process.name().lower() and any("server.py" in cmd.lower() for cmd in process.cmdline()):
                    return pid

            # If we get here, the PID exists but it's not our server process
            logger.warning(f"🧹 Removing stale PID file from previous server instance")
            os.remove(PID_FILE)
            return None
        except (ValueError, ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"🧹 Removing stale PID file: {str(e)}")
            os.remove(PID_FILE)
            return None
    return None

# Create PID file
def create_pid_file():
    pid = os.getpid()
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(pid))
        logger.info(f"📝 Created PID file with server process ID: {pid}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create PID file: {str(e)}")
        return False

# Remove PID file
def remove_pid_file():
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            logger.info(f"🧹 Removed PID file")
    except Exception as e:
        logger.error(f"❌ Failed to remove PID file: {str(e)}")

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
if check_server_running():
    logger.error("❌ Server is already running! Exiting...")
    sys.exit(1)

# Create PID file
if not create_pid_file():
    logger.error("❌ Failed to create PID file! Exiting...")
    sys.exit(1)

# Create an MCP server with proper MCP protocol handling
class DynamicAdditionServer(Server):
    """MCP server that provides an addition tool and supports dynamic function registration"""

    def __init__(self):
        super().__init__("Dynamic Function Server")
        # No need to load a registry at startup - we'll scan files directly

        # Register tool handlers using SDK decorators

        # Define the tools list handler
        @self.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Return a list of available tools"""
            logger.info("📋 TOOLS LIST REQUESTED")

            # Start with our built-in tools
            tools = [
                Tool(
                    name="register_function",
                    description="Install or update a Python function",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "code": {"type": "string"},
                        },
                        "required": ["name", "code"]
                    }
                ),
                Tool( # Add definition for get_tool_code
                    name="get_function",
                    description="Get the source code for a Python function",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The name of the function to get code for"}
                        },
                        "required": ["name"]
                    }
                ),
                Tool( # Add definition for remove_dynamic_tool
                    name="remove_function",
                    description="Remove a dynamically registered Python function",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The name of the function to remove"}
                        },
                        "required": ["name"]
                    }
                ),
            ]

            # Scan the functions directory for .py files
            dynamic_functions = self._discover_functions()

            # Add all discovered functions to the list
            tools.extend(dynamic_functions)

            return tools

        # Define stub for list_prompts handler
        @self.list_prompts()
        async def handle_list_prompts() -> list:
            """Return a list of available prompts (empty stub)"""
            logger.info("📋 PROMPTS LIST REQUESTED")
            return []

        # Define stub for list_resources handler
        @self.list_resources()
        async def handle_list_resources() -> list:
            """Return a list of available resources (empty stub)"""
            logger.info("📋 RESOURCES LIST REQUESTED")
            return []

        # Define the tool call handler
        @self.call_tool()
        async def handle_call_tool(name: str, args: dict) -> list[TextContent]:
            """Handle a tool call"""
            logger.info(f"🧰 TOOL CALLED: {name}")

            if name == "register_function":
                return await self._register_function(args)

            elif name == "get_function": # Corrected name check
                return await self._get_function_code(args)

            elif name == "remove_function": # Add routing for remove_function
                return await self._remove_function(args)

            # Check if this is a dynamically registered function
            else:
                # Check if we have a file for this function
                function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")
                if os.path.exists(function_path):
                    try:
                        # Load metadata from the file
                        metadata = self._extract_metadata_from_file(function_path)
                        if not metadata or "func_name" not in metadata:
                            raise ValueError(f"Could not extract metadata from {function_path}")

                        # Load the function dynamically
                        func = self._load_function_from_file(name, function_path, metadata["func_name"])

                        logger.info(f"🔧 EXECUTING DYNAMIC FUNCTION: {name}")

                        # Execute the function with the provided arguments
                        if inspect.iscoroutinefunction(func):
                            result = await func(**args)
                        else:
                            # Run non-async function in an executor to avoid blocking
                            result = await asyncio.to_thread(func, **args)

                        logger.info(f"✅ DYNAMIC FUNCTION RESULT: {result}")

                        # Return the result as text
                        return [TextContent(type="text", text=str(result))]
                    except Exception as e:
                        logger.error(f"❌ ERROR EXECUTING DYNAMIC FUNCTION {name}: {str(e)}")
                        import traceback
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        return [TextContent(type="text", text=f"Error: {str(e)}")]
                else:
                    logger.warning(f"❌ UNKNOWN FUNCTION: {name}")
                    return [TextContent(type="text", text=f"Unknown function: {name}")]

    def _extract_metadata_from_file(self, file_path):
        """Extract metadata from the Python file comments"""
        metadata = {
            "func_name": None,
            "description": "Dynamically registered function",
            "input_schema": {"type": "object"}
        }

        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Extract the function name from the file name
            basename = os.path.basename(file_path)
            function_name = os.path.splitext(basename)[0]
            metadata["name"] = function_name

            # Extract description from comments
            desc_match = re.search(r'#\s*Description:\s*(.+)', content)
            if desc_match:
                metadata["description"] = desc_match.group(1).strip()

            # Extract function name from the def statement
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
            if func_match:
                metadata["func_name"] = func_match.group(1)

            # Try to extract input schema from docstring or comments
            schema_match = re.search(r'""".*?Input schema:(.+?)"""', content, re.DOTALL)
            if schema_match:
                try:
                    # Try to parse JSON from the docstring
                    schema_str = schema_match.group(1).strip()
                    metadata["input_schema"] = json.loads(schema_str)
                except:
                    pass

            return metadata
        except Exception as e:
            logger.error(f"❌ ERROR EXTRACTING METADATA FROM {file_path}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def _discover_functions(self):
        """Scan the functions directory and discover all available functions"""
        functions = []

        try:
            # List all .py files in the functions directory, excluding __init__.py
            for filename in os.listdir(FUNCTIONS_DIR):
                if filename.endswith(".py") and filename != "__init__.py":
                    file_path = os.path.join(FUNCTIONS_DIR, filename)

                    # Extract the function name from the file name
                    function_name = os.path.splitext(filename)[0]

                    # Extract metadata from the file
                    metadata = self._extract_metadata_from_file(file_path)
                    if metadata:
                        # Create a Tool object from the metadata
                        functions.append(Tool(
                            name=function_name,
                            description=metadata.get("description", f"Dynamic function: {function_name}"),
                            inputSchema=metadata.get("input_schema", {"type": "object"})
                        ))
                        logger.debug(f"🔍 DISCOVERED FUNCTION: {function_name}")

            logger.info(f"🔍 DISCOVERED {len(functions)} DYNAMIC FUNCTIONS")
            return functions
        except Exception as e:
            logger.error(f"❌ ERROR DISCOVERING FUNCTIONS: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []

    def _load_function_from_file(self, name, file_path, func_name):
        """Load a function from a Python file"""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(name, file_path)
            if not spec or not spec.loader:
                raise ValueError(f"Could not load module spec for {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the function from the module
            if not hasattr(module, func_name):
                raise ValueError(f"Function {func_name} not found in module {name}")

            func = getattr(module, func_name)

            # Get the function signature
            signature = inspect.signature(func)

            def wrapper(**kwargs):
                # Validate the input arguments against the function signature
                try:
                    bound_args = signature.bind(**kwargs)
                    bound_args.apply_defaults()
                except TypeError as e:
                    raise ValueError(f"Invalid input arguments: {str(e)}")

                # Call the function with the validated arguments
                return func(**bound_args.arguments)

            return wrapper

        except Exception as e:
            logger.error(f"❌ ERROR LOADING FUNCTION FROM {file_path}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    async def _register_function(self, args: dict) -> list[TextContent]:
        """Register a new Python function as a tool"""
        name = args.get("name")
        code = args.get("code")
        description = args.get("description", f"Dynamically registered function: {name}")

        # Validate the function name (must be a valid Python identifier)
        if not name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return [TextContent(type="text", text=f"Invalid function name: {name}. Must be a valid Python identifier.")]

        logger.info(f"🔄 REGISTERING NEW FUNCTION: {name}")

        try:
            # First, try to extract the function name from the code
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
            if not func_match:
                raise ValueError("Could not find a function definition in the code")

            func_name = func_match.group(1)

            # Create a file for the function
            file_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")

            # Create a namespace to safely test the code first
            namespace = {}
            try:
                # Test execute the code to verify it's valid Python
                exec(code, namespace)

                # Check if the function is defined
                if func_name not in namespace:
                    raise ValueError(f"Function {func_name} was not defined in the code")

                # Check if the function accepts named parameters and not just a single 'args' parameter
                func = namespace[func_name]
                sig = inspect.signature(func)
                if len(sig.parameters) < 1:
                    raise ValueError(f"Function {func_name} must have at least one parameter")
            except Exception as e:
                raise ValueError(f"Invalid function code: {str(e)}")

            # Auto-generate input schema from function signature
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                param_type = "string" # Default type if annotation is missing
                if param.annotation != inspect.Parameter.empty:
                    # Basic type mapping (can be expanded)
                    if param.annotation == str:
                        param_type = "string"
                    elif param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"
                    # Add more complex type mappings if needed (e.g., typing.List[str])

                properties[param_name] = {"type": param_type}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                else:
                    # Add default value to schema description if present
                    properties[param_name]["default"] = param.default

            input_schema = {
                "type": "object",
                "properties": properties
            }
            if required:
                input_schema["required"] = required

            # Format the generated input schema as a comment
            schema_str = json.dumps(input_schema, indent=2)

            # The code is valid, now save it to a file with proper formatting and metadata in comments
            module_code = f"""# Dynamic function: {name}
# Description: {description}
# Generated by MCP Dynamic Function Server

{code.strip()}

\"\"\"
Input schema:
{schema_str}
\"\"\"
"""

            # Save the code to the file
            with open(file_path, 'w') as f:
                f.write(module_code)

            logger.info(f"✅ SUCCESSFULLY REGISTERED FUNCTION: {name}")
            return [TextContent(type="text", text=f"Successfully registered function: {name}")]

        except Exception as e:
            logger.error(f"❌ ERROR REGISTERING FUNCTION: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return [TextContent(type="text", text=f"Error registering function: {str(e)}")]

    async def _get_function_code(self, args: dict) -> list[TextContent]:
        """Get the Python source code for a dynamically registered function"""
        name = args.get("name")
        if not name:
            return [TextContent(type="text", text="Error: Function name not provided")]

        logger.info(f"📄 GETTING CODE FOR FUNCTION: {name}")

        # Construct the expected path to the function's Python file
        function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")

        # Check if the function file exists
        if not os.path.exists(function_path):
            logger.warning(f"⚠️ Function file not found: {function_path}")
            return [TextContent(type="text", text=f"Error: Function '{name}' not found.")]

        try:
            # Read the function code from the file
            with open(function_path, 'r') as f:
                code = f.read()

            logger.info(f"✅ SUCCESSFULLY RETRIEVED CODE FOR: {name}")
            # Return the code as text content
            return [TextContent(type="text", text=code)]
        except Exception as e:
            logger.error(f"❌ ERROR READING FUNCTION FILE {function_path}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return [TextContent(type="text", text=f"Error reading function code for '{name}': {str(e)}")]

    async def _remove_function(self, args: dict) -> list[TextContent]:
        """Remove a dynamically registered function by deleting its file"""
        name = args.get("name")
        if not name:
            return [TextContent(type="text", text="Error: Function name not provided")]

        logger.info(f"🗑️ REMOVING FUNCTION: {name}")

        # Construct the expected path to the function's Python file
        function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")

        # Check if the function file exists
        if not os.path.exists(function_path):
            logger.warning(f"⚠️ Function file not found for removal: {function_path}")
            return [TextContent(type="text", text=f"Error: Function '{name}' not found.")]

        try:
            # Delete the function file
            os.remove(function_path)
            logger.info(f"✅ SUCCESSFULLY REMOVED FUNCTION: {name}")
            return [TextContent(type="text", text=f"Successfully removed function: {name}")]
        except Exception as e:
            logger.error(f"❌ ERROR REMOVING FUNCTION FILE {function_path}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return [TextContent(type="text", text=f"Error removing function '{name}': {str(e)}")]


# ServiceClient class to manage the connection to the cloud server via Socket.IO
class ServiceClient:
    """Socket.IO client for connecting to the cloud server

    This class implements a Socket.IO CLIENT to connect TO the cloud server.
    Socket.IO is different from standard WebSockets - it adds features like:
    - Automatic reconnection
    - Fallback to long polling when WebSockets aren't available
    - Built-in event system with namespaces
    - Authentication handling

    While server.py acts as a WebSocket SERVER for the node-mcp-client,
    it must act as a Socket.IO CLIENT to connect to the cloud server.
    Each server dictates which protocol clients must use to connect.

    Manages the Socket.IO connection to the cloud server's service namespace.
    """
    def __init__(self, server_url: str, namespace: str, email: str, api_key: str, mcp_server):
        self.server_url = server_url
        self.namespace = namespace
        self.email = email
        self.api_key = api_key
        self.mcp_server = mcp_server
        self.sio = None
        self.connection_task = None
        self.is_connected = False
        self.retry_count = 0
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

                # Connect to the cloud server using Socket.IO
                logger.info(f"☁️ Attempting connection to cloud server (attempt {self.retry_count + 1})")

                # Connect with authentication data
                await self.sio.connect(
                    self.server_url,
                    namespaces=[self.namespace],
                    auth={
                        "email": self.email,  # Must match exact casing expected by Node.js server
                        "apiKey": self.api_key         # Must match exact casing expected by Node.js server
                    }
                )

                self.is_connected = True
                self.retry_count = 0  # Reset retry counter on successful connection
                logger.info("✅ CONNECTED TO CLOUD SERVER!")

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

                # Wait before retrying
                logger.info(f"☁️ RETRYING CONNECTION IN {CLOUD_CONNECTION_RETRY_SECONDS} SECONDS...")
                await asyncio.sleep(CLOUD_CONNECTION_RETRY_SECONDS)

        logger.info("☁️ Cloud connection maintenance loop ended")

    def _register_event_handlers(self):
        """Register Socket.IO event handlers"""
        if not self.sio:
            return

        # Connection established event
        @self.sio.event(namespace=self.namespace)
        def connect():
            logger.info(f"✅ CONNECTED TO CLOUD SERVER NAMESPACE: {self.namespace}")

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
            logger.info(f"☁️ RECEIVED SERVICE MESSAGE: {data}")
            # Check if this is an MCP JSON-RPC request
            if isinstance(data, dict) and 'jsonrpc' in data and 'method' in data:
                # This is an MCP JSON-RPC request
                response = await self._process_mcp_request(data)
                if response:
                    await self.send_message('service_response', response)
            else:
                # Process as a regular service message
                response = await self._process_service_message(data)
                if response:
                    await self.send_message('service_response', response)


    async def _process_mcp_request(self, request: dict) -> Union[dict, None]:
        """Process an MCP JSON-RPC request from the cloud server

        This method takes requests received over Socket.IO from the cloud server
        and routes them to the same handlers that process WebSocket requests.

        Despite the different transport protocols (Socket.IO vs WebSockets),
        both connection types ultimately use the same underlying MCP handlers
        like handle_list_tools() to process requests.
        """
        request_id = request.get("id", "unknown-id")
        method = request.get("method", "")
        params = request.get("params", {})

        logger.info(f"☁️ Processing MCP request: {method} (ID: {request_id})")

        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }

        try:
            # Map MCP JSON-RPC methods to MCP server methods
            if method == "tools/list":
                logger.info(f"☁️ Forwarding tools/list request to MCP server")
                # The mcp_server already has a list_tools handler
                tools_list = await self.mcp_server.handle_list_tools()
                response["result"] = {"tools": tools_list}

            elif method == "tools/call":
                # Extract tool details from params
                tool_name = params.get("name")
                tool_args = params.get("args", {})

                if not tool_name:
                    response["error"] = {"code": -32602, "message": "Invalid params: missing tool name"}
                else:
                    logger.info(f"☁️ Executing tool from cloud: {tool_name}")
                    # Use our MCP server to handle the tool call
                    result = await self.mcp_server.handle_call_tool(tool_name, tool_args)
                    response["result"] = {"result": result}

            else:
                # Unknown method
                logger.warning(f"⚠️ Unknown MCP method: {method}")
                response["error"] = {"code": -32601, "message": f"Method not found: {method}"}

        except Exception as e:
            # Exception during processing
            logger.error(f"❌ ERROR PROCESSING MCP REQUEST: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            response["error"] = {"code": -32000, "message": f"Server error: {str(e)}"}

        return response

    async def _process_service_message(self, message: dict) -> Union[dict, None]:
        """Process a non-MCP service message from the cloud server"""
        # Handle regular service messages (non-MCP JSON-RPC)
        message_type = message.get("type", "unknown")
        logger.info(f"☁️ Processing service message of type: {message_type}")

        # Handle different message types
        if message_type == "call_tool":
            # Legacy format for tool calls
            tool_name = message.get("name")
            tool_args = message.get("args", {})
            logger.info(f"☁️ Forwarding tool call: {tool_name}")

            try:
                # Use the MCP server to handle the tool call
                call_result = await self.mcp_server.handle_call_tool(tool_name, tool_args)
                return {"type": "tool_result", "result": call_result}
            except Exception as e:
                logger.error(f"❌ ERROR EXECUTING TOOL {tool_name}: {str(e)}")
                return {"type": "error", "tool": tool_name, "error": str(e)}

        # Return None if no response is needed or message type not recognized
        return None

    async def send_message(self, event: str, data: dict) -> bool:
        """Send a message to the cloud server"""
        if not self.is_connected or not self.sio:
            logger.warning(f"⚠️ Cannot send {event}: No active cloud connection")
            return False

        try:
            await self.sio.emit(event, data, namespace=self.namespace)
            logger.debug(f"☁️ SENT {event} TO CLOUD: {data}")
            return True
        except Exception as e:
            logger.error(f"❌ ERROR SENDING {event} TO CLOUD: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    async def disconnect(self):
        """Disconnect from the cloud server"""
        logger.info("☁️ Disconnecting from cloud server")
        self.connection_active = False

        if self.sio and self.is_connected:
            try:
                await self.sio.disconnect()
            except Exception as e:
                logger.error(f"❌ ERROR DISCONNECTING: {str(e)}")

        self.is_connected = False

# Create our MCP server instance
mcp_server = DynamicAdditionServer()

# We'll create the cloud connection later after parsing arguments
cloud_connection = None

# Create a Starlette app with websocket support
async def handle_websocket(websocket):
    """Handle MCP websocket connections from node-mcp-client

    This function handles standard WebSocket connections from clients like node-mcp-client.
    We act as a WebSocket SERVER here, which is different from how we act as a Socket.IO CLIENT
    when connecting to the cloud server.

    Both connection types ultimately route to the same MCP handlers in DynamicAdditionServer.
    """
    logger.info(f"⚡ NEW CONNECTION: {websocket.client}")

    try:
        logger.info("🔄 STARTING MCP WEBSOCKET TRANSPORT")
        async with websocket_server(
            websocket.scope, websocket.receive, websocket.send
        ) as streams:
            logger.info("🚀 MCP SESSION STARTED")

            # Run the MCP server with the websocket streams
            await mcp_server.run(
                streams[0], streams[1], mcp_server.create_initialization_options()
            )
            logger.info("👋 MCP SESSION ENDED")
    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        # Print a more detailed stack trace for debugging
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("🔌 CONNECTION CLOSED")

# Create a Starlette application with our websocket route
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
        # Update the cloud connection URL
        cloud_connection.server_url = CLOUD_SERVER_URL

    # Set up the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

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
            if not args.email or not args.api_key:
                logger.error("❌ CLOUD SERVER CONNECTION REQUIRES EMAIL AND API KEY")
                logger.error("❌ Use --email and --api-key to specify credentials")
                logger.info("☁️ CLOUD SERVER CONNECTION DISABLED")
            else:
                logger.info(f"☁️ CLOUD SERVER CONNECTION ENABLED: {CLOUD_SERVER_URL}")
                # Create the cloud connection with the provided credentials
                cloud_connection = ServiceClient(
                    server_url=CLOUD_SERVER_URL,
                    namespace=CLOUD_SERVICE_NAMESPACE,
                    email=args.email,
                    api_key=args.api_key,
                    mcp_server=mcp_server
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
        logger.info("👋 SERVER SHUTDOWN COMPLETE")
