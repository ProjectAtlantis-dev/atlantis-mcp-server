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
from typing import Any, Callable, Dict, List, Optional
from mcp.server import Server
from mcp.server.websocket import websocket_server
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

# Directory to store dynamic tool files
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_tools")

# Path for the PID file
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.pid")

# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces by default
PORT = 8000

# Create tools directory if it doesn't exist
os.makedirs(TOOLS_DIR, exist_ok=True)

# Flag to track if we're in shutdown process
is_shutting_down = False

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
                    name="add",
                    description="Add two numbers together",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    }
                ),
                Tool(
                    name="register_function",
                    description="Register a new Python function as a tool",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "code": {"type": "string"},
                            "input_schema": {"type": "object"}
                        },
                        "required": ["name", "code", "input_schema"]
                    }
                )
            ]
            
            # Scan the tools directory for .py files
            dynamic_tools = self._discover_tools()
            
            # Add all discovered tools to the list
            tools.extend(dynamic_tools)
            
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
            
            if name == "add":
                a = args.get("a", 0)
                b = args.get("b", 0)
                
                logger.info(f"📊 ADDING: {a} + {b}")
                result = a + b
                logger.info(f"✅ RESULT: {result}")
                
                # Return a text content item with the result
                return [TextContent(type="text", text=str(result))]
            
            elif name == "register_function":
                return await self._register_function(args)
            
            # Check if this is a dynamically registered tool
            else:
                # Check if we have a file for this tool
                tool_path = os.path.join(TOOLS_DIR, f"{name}.py")
                if os.path.exists(tool_path):
                    try:
                        # Load metadata from the file
                        metadata = self._extract_metadata_from_file(tool_path)
                        if not metadata or "func_name" not in metadata:
                            raise ValueError(f"Could not extract metadata from {tool_path}")
                        
                        # Load the function dynamically
                        func = self._load_function_from_file(name, tool_path, metadata["func_name"])
                        
                        logger.info(f"🔧 EXECUTING DYNAMIC TOOL: {name}")
                        
                        # Execute the function with the provided arguments
                        if inspect.iscoroutinefunction(func):
                            result = await func(args)
                        else:
                            # Run non-async function in an executor to avoid blocking
                            result = await asyncio.to_thread(func, args)
                            
                        logger.info(f"✅ DYNAMIC TOOL RESULT: {result}")
                        
                        # Return the result as text
                        return [TextContent(type="text", text=str(result))]
                    except Exception as e:
                        logger.error(f"❌ ERROR EXECUTING DYNAMIC TOOL {name}: {str(e)}")
                        import traceback
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        return [TextContent(type="text", text=f"Error: {str(e)}")]
                else:
                    logger.warning(f"❌ UNKNOWN TOOL: {name}")
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    def _extract_metadata_from_file(self, file_path):
        """Extract metadata from the Python file comments"""
        metadata = {
            "func_name": None,
            "description": "Dynamically registered tool",
            "input_schema": {"type": "object"}
        }
        
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Extract the tool name from the file name
            basename = os.path.basename(file_path)
            tool_name = os.path.splitext(basename)[0]
            metadata["name"] = tool_name
            
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
    
    def _discover_tools(self):
        """Scan the tools directory and discover all available tools"""
        tools = []
        
        try:
            # List all .py files in the tools directory, excluding __init__.py
            for filename in os.listdir(TOOLS_DIR):
                if filename.endswith(".py") and filename != "__init__.py":
                    file_path = os.path.join(TOOLS_DIR, filename)
                    
                    # Extract the tool name from the file name
                    tool_name = os.path.splitext(filename)[0]
                    
                    # Extract metadata from the file
                    metadata = self._extract_metadata_from_file(file_path)
                    if metadata:
                        # Create a Tool object from the metadata
                        tools.append(Tool(
                            name=tool_name,
                            description=metadata.get("description", f"Dynamic tool: {tool_name}"),
                            inputSchema=metadata.get("input_schema", {"type": "object"})
                        ))
                        logger.debug(f"🔍 DISCOVERED TOOL: {tool_name}")
            
            logger.info(f"🔍 DISCOVERED {len(tools)} DYNAMIC TOOLS")
            return tools
        except Exception as e:
            logger.error(f"❌ ERROR DISCOVERING TOOLS: {str(e)}")
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
                
            return getattr(module, func_name)
            
        except Exception as e:
            logger.error(f"❌ ERROR LOADING FUNCTION FROM {file_path}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def _register_function(self, args: dict) -> list[TextContent]:
        """Register a new Python function as a tool"""
        name = args.get("name")
        code = args.get("code")
        description = args.get("description", f"Dynamically registered tool: {name}")
        input_schema = args.get("input_schema", {"type": "object"})
        
        # Validate the tool name (must be a valid Python identifier)
        if not name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return [TextContent(type="text", text=f"Invalid tool name: {name}. Must be a valid Python identifier.")]
        
        logger.info(f"🔄 REGISTERING NEW FUNCTION: {name}")
        
        try:
            # First, try to extract the function name from the code
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
            if not func_match:
                raise ValueError("Could not find a function definition in the code")
            
            func_name = func_match.group(1)
            
            # Create a file for the function
            file_path = os.path.join(TOOLS_DIR, f"{name}.py")
            
            # Create a namespace to safely test the code first
            namespace = {}
            try:
                # Test execute the code to verify it's valid Python
                exec(code, namespace)
                
                # Check if the function is defined
                if func_name not in namespace:
                    raise ValueError(f"Function {func_name} was not defined in the code")
            except Exception as e:
                raise ValueError(f"Invalid function code: {str(e)}")
            
            # Format the input schema as a comment
            schema_str = json.dumps(input_schema, indent=2)
            
            # The code is valid, now save it to a file with proper formatting and metadata in comments
            module_code = f"""# Dynamic tool: {name}
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

# Create our MCP server instance
mcp_server = DynamicAdditionServer()

# Create a Starlette app with websocket support
async def handle_websocket(websocket):
    """Handle MCP websocket connections"""
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
    args = parser.parse_args()
    
    # Update host and port from command line arguments
    HOST = args.host
    PORT = args.port
    
    # Run the server
    logger.info(f"🌟 STARTING MCP WEBSOCKET SERVER AT ws://{HOST}:{PORT}/mcp")
    # Set Uvicorn log level to warning to reduce noise
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
