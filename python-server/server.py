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
import datetime
import random
import socketio
from mcp.server import Server
from mcp.server.websocket import websocket_server
from mcp.client.websocket import websocket_client
from mcp.types import Tool, TextContent, CallToolResult, ToolListChangedNotification, NotificationParams, Annotations
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
import uvicorn
import argparse
from werkzeug.utils import secure_filename


# NOTE: This server uses two different socket protocols:
# 1. Standard WebSockets: When acting as a SERVER to accept connections from node-mcp-client
# 2. Socket.IO: When acting as a CLIENT to connect to the cloud Node.js server
#
# Each server dictates its own protocol, and clients must adapt accordingly.
# - The node-mcp-client connects via standard WebSockets to our server.py
# - Our server.py connects via Socket.IO to the cloud server
# - Both ultimately route to the same MCP handlers in the DynamicAdditionServer class

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
CLOUD_CONNECTION_RETRY_SECONDS = 5  # Initial delay in seconds
CLOUD_CONNECTION_MAX_RETRIES = 10 # Maximum number of retries before giving up (None for infinite)
CLOUD_CONNECTION_MAX_BACKOFF_SECONDS = 60 # Maximum delay for exponential backoff

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
        self.tasks = {} # Store tasks in a dictionary
        self.next_task_id = 1 # Initialize next task ID

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

    # --- Core Logic Methods (callable directly) ---

    async def _get_tools_list(self) -> list[Tool]:
        """Core logic to return a list of available tools"""
        logger.info("📋 TOOLS LIST LOGIC EXECUTED")

        # Start with our built-in tools
        tools = [
            Tool(
                name="function_register",
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
                name="function_get",
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
                name="function_remove",
                description="Remove a dynamically registered Python function",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the function to remove"}
                    },
                    "required": ["name"]
                }
            ),
            # --- Task Management Tools --- #
            Tool(
                name="task_add",
                description="(Stub) Add a new task via a JSON payload", # Updated description
                inputSchema={
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "description": "The JSON object containing the task details."
                        }
                    },
                    "required": ["payload"]
                }
            ),
            Tool(
                name="task_run",
                description="(Stub) Run an existing task",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "ID of the task to run"}
                    },
                    "required": ["id"]
                }
            ),
            Tool(
                name="task_remove",
                description="(Stub) Remove a task",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "ID of the task to remove"}
                    },
                    "required": ["id"]
                }
            ),
            Tool(
                name="task_peek",
                description="(Stub) Retrieve the stored details for a specific task ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "ID of the task to peek"}
                    },
                    "required": ["id"]
                }
            ),
        ]

        # Scan the functions directory for .py files
        dynamic_functions = self._discover_functions()

        # Add all discovered functions to the list
        tools.extend(dynamic_functions)

        return tools

    async def _get_prompts_list(self) -> list:
        """Core logic to return a list of available prompts (empty stub)"""
        logger.info("📋 PROMPTS LIST LOGIC EXECUTED")
        return []

    async def _get_resources_list(self) -> list:
        """Core logic to return a list of available resources (empty stub)"""
        logger.info("📋 RESOURCES LIST LOGIC EXECUTED")
        return []

    async def _execute_tool(self, name: str, args: dict) -> list[TextContent]:
        """Core logic to handle a tool call. Ensures result is List[TextContent(type='text')]"""
        logger.info(f"🧰 TOOL EXECUTION LOGIC: {name}")

        try:
            result_value: Any = None # Variable to hold the raw result before wrapping

            if name == "function_register":
                result_value = await self._register_function(args)

            elif name == "function_get": # Corrected name check
                result_value = await self._get_function_code(args)

            elif name == "function_remove": # Add routing for remove_function
                result_value = await self._remove_function(args)

            # --- Handle Task Management Stubs ---
            elif name == "task_add":
                result_value = await self._task_add(args)
            elif name == "task_run":
                result_value = await self._task_run(args)
            elif name == "task_remove":
                result_value = await self._task_remove(args)
            elif name == "task_peek":
                result_value = await self._task_peek(args)

            # Check if this is a dynamically registered function
            else:
                function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")
                if os.path.exists(function_path):
                    metadata = self._extract_metadata_from_file(function_path)
                    if not metadata or "func_name" not in metadata:
                        raise ValueError(f"Could not extract metadata from {function_path}")

                    func = self._load_function_from_file(name, function_path, metadata["func_name"])
                    logger.info(f"🔧 EXECUTING DYNAMIC FUNCTION: {name}")

                    if inspect.iscoroutinefunction(func):
                        raw_dynamic_result = await func(**args)
                    else:
                        raw_dynamic_result = await asyncio.to_thread(func, **args)

                    logger.info(f"✅ DYNAMIC FUNCTION RESULT: {raw_dynamic_result}")
                    result_value = raw_dynamic_result # Assign raw result to be wrapped later
                else:
                    raise ValueError(f"Unknown tool name: {name}")

            # --- Wrap the result_value in List[TextContent] --- VITAL STEP
            if isinstance(result_value, list) and all(isinstance(item, TextContent) for item in result_value):
                 # Already in correct format (e.g., from _register_function etc.)
                 # Ensure type field is present if SDK helpers didn't add it
                 # (This is defensive, assuming helpers might also return just {'text':...})
                 for item in result_value:
                      if not hasattr(item, 'type') or item.type != 'text':
                           # If type is missing or wrong, force it (might lose other fields if TextContent model changes)
                           logger.warning(f"Tool {name} helper returned TextContent without type='text', correcting.")
                           item.type = 'text' # Modify in place if possible, or reconstruct
                 logger.debug(f"🚀 Returning correctly formatted result: {result_value}") # Add detailed log
                 return result_value # Should return the list with annotations intact
            elif isinstance(result_value, str):
                 logger.debug(f"🚀 Wrapping string result: {result_value}")
                 return [TextContent(type="text", text=result_value)] # Wrap string
            else:
                 # Attempt to convert other types to string representation
                 logger.warning(f"Tool {name} returned non-standard type {type(result_value)}, converting to string.")
                 return [TextContent(type="text", text=str(result_value))] # Wrap converted string

        except Exception as e:
             logger.error(f"❌ Error executing tool '{name}': {e}")
             import traceback
             logger.debug(f"Traceback: {traceback.format_exc()}")
             # Re-raise the exception so the framework can handle it as a JSON-RPC error
             raise e

    # --- Helper Methods for Dynamic Functions ---
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
                        tool_instance = Tool(
                            name=function_name,
                            description=metadata.get("description", f"Dynamic function: {function_name}"),
                            inputSchema=metadata.get("input_schema", {"type": "object"})
                        )

                        # Get and add timestamp
                        try:
                            mtime = os.path.getmtime(file_path)
                            # Use ISO format with timezone Z for UTC
                            timestamp_str = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
                            # Dynamically add the attribute to the instance
                            tool_instance.lastUpdated = timestamp_str
                            logger.debug(f"📄 Processing {filename}, last updated: {timestamp_str}")
                        except OSError as e:
                            logger.warning(f"⚠️ Could not get mtime for {filename}: {e}")
                            tool_instance.lastUpdated = None # Assign None if timestamp fetch fails

                        functions.append(tool_instance)
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
            raise ValueError(f"Invalid function name: {name}. Must be a valid Python identifier.")

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
            exec(code, namespace)

            # Check if the function is defined
            if func_name not in namespace:
                raise ValueError(f"Function {func_name} was not defined in the code")

            # Check if the function accepts named parameters and not just a single 'args' parameter
            func = namespace[func_name]
            sig = inspect.signature(func)
            if len(sig.parameters) < 1:
                raise ValueError(f"Function {func_name} must have at least one parameter")

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

            # Send notification that tools list has changed
            try:
                # Get the current session
                ctx = self.request_context
                # Create and send a ToolListChangedNotification
                notification = ToolListChangedNotification(
                    method="notifications/tools/list_changed",
                    params=NotificationParams()
                )
                await ctx.session.send_notification(notification)
                logger.info(f"📢 SENT TOOL LIST CHANGED NOTIFICATION")
            except Exception as e:
                # Log a warning instead of returning an error, registration succeeded.
                logger.warning(f"⚠️ COULD NOT SEND TOOL LIST CHANGED NOTIFICATION: {str(e)}")

            return [TextContent(type="text", text=f"Successfully registered function: {name}")]
        except Exception as e:
            logger.error(f"❌ ERROR REGISTERING FUNCTION: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise e

    async def _get_function_code(self, args: dict) -> list[TextContent]:
        """Get the Python source code and description for a dynamically registered function"""
        name = args.get("name")
        if not name:
            raise ValueError("Missing 'name' in arguments")

        logger.info(f"📄 GETTING CODE AND DESC FOR FUNCTION: {name}")

        # Construct the expected path to the function's Python file
        function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")

        # Check if the function file exists
        if not os.path.exists(function_path):
            raise ValueError(f"Function '{name}' not found.")

        try:
            # Read the function code from the file
            with open(function_path, 'r') as f:
                code = f.read()

            # Extract metadata to get the description
            metadata = self._extract_metadata_from_file(function_path)
            description = metadata.get("description", None) # Get description or None

            # Prepare the result as a dictionary
            result_data = {
                "name": name,
                "code": code,
                "description": description
            }

            logger.info(f"✅ SUCCESSFULLY RETRIEVED CODE AND DESC FOR: {name}")
            # Return the result as a JSON string
            return [TextContent(type="text", text=json.dumps(result_data))]
        except Exception as e:
            logger.error(f"❌ ERROR READING FUNCTION FILE OR METADATA {function_path}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise e

    async def _remove_function(self, args: dict) -> list[TextContent]:
        """Remove a dynamically registered function by deleting its file"""
        name = args.get("name")
        if not name:
            raise ValueError("Missing 'name' in arguments")

        logger.info(f"🗑️ REMOVING FUNCTION: {name}")

        # Construct the expected path to the function's Python file
        function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")

        # Check if the function file exists
        if not os.path.exists(function_path):
            raise ValueError(f"Function '{name}' not found.")

        try:
            # Delete the function file
            os.remove(function_path)
            logger.info(f"✅ SUCCESSFULLY REMOVED FUNCTION: {name}")

            # Send notification that tools list has changed
            try:
                # Get the current session
                ctx = self.request_context
                # Create and send a ToolListChangedNotification
                notification = ToolListChangedNotification(
                    method="notifications/tools/list_changed",
                    params=NotificationParams()
                )
                await ctx.session.send_notification(notification)
                logger.info(f"📢 SENT TOOL LIST CHANGED NOTIFICATION")
            except Exception as e:
                # Log a warning instead of returning an error, removal succeeded.
                logger.warning(f"⚠️ COULD NOT SEND TOOL LIST CHANGED NOTIFICATION: {str(e)}")

            return [TextContent(type="text", text=f"Successfully removed function: {name}")]
        except Exception as e:
            logger.error(f"❌ ERROR REMOVING FUNCTION FILE {function_path}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise e

    # --- Task Management Stubs --- #

    async def _task_add(self, args: dict) -> list[TextContent]:
        """Adds a new task using the provided payload."""
        logger.info(f"⚙️ TASK ADD CALLED with args: {args}")
        try:
            # Extract the payload from the arguments
            task_payload = args.get('payload')
            if task_payload is None:
                raise ValueError("Missing 'payload' in arguments")
            if not isinstance(task_payload, dict):
                raise ValueError("'payload' must be a JSON object (dictionary)")

            # Generate a new task ID
            task_id = self.next_task_id
            self.next_task_id += 1

            # Store the task details (the extracted payload dictionary)
            self.tasks[task_id] = task_payload # Store the payload, not the whole args

            logger.info(f"✅ Task added with ID: {task_id}, Details: {task_payload}")
            # Return the new task ID as string in 'text' and int in annotations.task_id_int
            return [
                TextContent(
                    type="text",
                    text=str(task_id),
                    annotations=Annotations(task_id_int=task_id) # Add ID to annotations
                )
            ]
        except Exception as e:
            logger.error(f"❌ Error adding task: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise e

    async def _task_run(self, args: dict) -> list[TextContent]:
        """Stub for running an existing task"""
        logger.info("▶️ TASK RUN CALLED (STUB)")
        # Placeholder logic: Extract args if needed
        task_id = args.get("id", "unknown_id")
        return [TextContent(type="text", text=f"Task run for ID '{task_id}' called (stub)")]

    async def _task_remove(self, args: dict) -> list[TextContent]:
        """Stub for removing a task"""
        logger.info("🗑️ TASK REMOVE CALLED (STUB)")
        # Placeholder logic: Extract args if needed
        task_id = args.get("id", "unknown_id")
        return [TextContent(type="text", text=f"Task remove for ID '{task_id}' called (stub)")]

    async def _task_peek(self, args: dict) -> list[TextContent]:
        """Retrieve the stored details for a specific task ID."""
        logger.info(f"👀 TASK PEEK CALLED with args: {args}")
        task_id_str = args.get('id')
        if task_id_str is None:
            raise ValueError("Missing 'id' in arguments")

        try:
            task_id = int(task_id_str) # Ensure ID is an integer
        except ValueError:
            raise ValueError("'id' must be an integer")

        # Retrieve the task details from the dictionary
        task_details = self.tasks.get(task_id)

        if task_details is not None:
            logger.info(f"✅ Task {task_id} details found: {task_details}")
            # Return the stored task details as JSON string in 'text' and raw dict in annotations.task_payload_json
            return [
                TextContent(
                    type="text",
                    text=json.dumps(task_details),
                    annotations=Annotations(task_payload_json=task_details) # Add payload to annotations
                )
            ]
        else:
            logger.warning(f"❓ Task ID {task_id} not found.")
            raise ValueError(f"Task ID {task_id} not found")


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
    def __init__(self, server_url: str, namespace: str, email: str, api_key: str, serviceName: str, mcp_server):
        self.server_url = server_url

        self.namespace = namespace

        self.email = email
        self.api_key = api_key
        self.serviceName = serviceName

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

                logger.info(f"☁️ Attempting connection to cloud server (attempt {self.retry_count + 1})")

                # Connect with authentication data including hostname
                import socket
                hostname = socket.gethostname()
                await self.sio.connect(
                    self.server_url,
                    namespaces=[self.namespace],
                    auth={
                        "email": self.email,
                        "apiKey": self.api_key,
                        "hostname": hostname,
                        "serviceName": self.serviceName
                    }
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
                    await self.send_message('service_response', response)
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
                    response["result"] = {"contents": [content.model_dump(include={'type', 'text'}) for content in call_result_list]} # Use model_dump() & include

            elif method == "prompts/list":
                 # Call the core logic method directly
                result_list = await self.mcp_server._get_prompts_list()
                response["result"] = {"prompts": result_list} # Assuming prompts are already serializable

            elif method == "resources/list":
                 # Call the core logic method directly
                result_list = await self.mcp_server._get_resources_list()
                response["result"] = {"resources": result_list} # Assuming resources are already serializable

            else:
                # Unknown method
                logger.warning(f"⚠️ Unknown MCP method: {method}")
                response["error"] = {"code": -32601, "message": f"Method not found: {method}"}

        except Exception as e:
            # General exception during processing
            logger.error(f"❌ ERROR PROCESSING MCP REQUEST: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            response["error"] = {"code": -32000, "message": f"Server error: {str(e)}"}

        return response

    async def send_message(self, event: str, data: dict) -> bool:
        """Send a message to the cloud server"""
        if not self.is_connected or not self.sio:
            logger.warning(f"⚠️ Cannot send {event}: No active cloud connection")
            return False

        try:
            await self.sio.emit(event, data, namespace=self.namespace)
            # logger.debug(f"☁️ SENT {event} TO CLOUD: {data}")
            logger.debug(f"☁️ SENT {event} TO CLOUD")
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
        remove_pid_file() # Ensure PID file is removed on exit
        logger.info("👋 SERVER SHUTDOWN COMPLETE")
