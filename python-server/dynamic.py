#!/usr/bin/env python3
import logging
import json
import inspect
import importlib
import importlib.util
import re
import os
import ast
import functools
import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from werkzeug.utils import secure_filename
from mcp.types import Tool, TextContent, CallToolResult, ToolListChangedNotification, NotificationParams, Annotations

# Import shared state
from state import logger, FUNCTIONS_DIR, dynamic_functions, tools

async def discover_functions(mcp_server):
    """Scan the functions directory and discover all available functions"""
    logger.info(f"🔍 SCANNING FOR FUNCTIONS IN: {FUNCTIONS_DIR}")

    # Reset the dynamic functions dictionary
    dynamic_functions.clear()

    # Ensure functions directory exists
    os.makedirs(FUNCTIONS_DIR, exist_ok=True)

    # Get a list of Python files in the functions directory
    function_files = [f for f in os.listdir(FUNCTIONS_DIR) if f.endswith('.py')]

    # No functions found
    if not function_files:
        logger.info("📭 NO FUNCTION FILES FOUND")
        return

    logger.info(f"📚 FOUND {len(function_files)} FUNCTION FILES")

    # Process each function file
    for file_name in function_files:
        file_path = os.path.join(FUNCTIONS_DIR, file_name)

        try:
            # Extract metadata from the file
            metadata = extract_metadata_from_file(file_path)
            if not metadata:
                logger.warning(f"⚠️ NO METADATA FOUND IN {file_name}, SKIPPING")
                continue

            # Get the function name from the file name
            name = os.path.splitext(file_name)[0]
            func_name = metadata.get('func_name', name)

            # Load the function from the file
            load_function_from_file(name, file_path, func_name, mcp_server)

        except Exception as e:
            logger.error(f"❌ ERROR LOADING FUNCTION FROM {file_name}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

def extract_metadata_from_file(file_path):
    """Extract metadata from the Python file comments"""
    logger.debug(f"📋 EXTRACTING METADATA FROM: {file_path}")
    metadata = {}

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the AST to get function definitions
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Store the function name
                    metadata['func_name'] = node.name

                    # Extract docstring if available
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Str)):
                        metadata['description'] = node.body[0].value.s.strip()

                    # Found a function definition, no need to continue parsing
                    break
        except Exception as e:
            logger.warning(f"⚠️ AST PARSING ERROR FOR {file_path}: {str(e)}")

        # Fall back to regex parsing if AST method doesn't yield results
        if 'func_name' not in metadata:
            # Extract function name from the code using regex
            func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
            if func_match:
                metadata['func_name'] = func_match.group(1)

        if 'description' not in metadata:
            # Look for docstring comments as fallback
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                metadata['description'] = docstring_match.group(1).strip()

        return metadata

    except Exception as e:
        logger.error(f"❌ ERROR EXTRACTING METADATA FROM {file_path}: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {}

def load_function_from_file(name, file_path, func_name, mcp_server):
    """Load a function from a Python file"""
    logger.info(f"📂 LOADING FUNCTION {name} FROM {file_path}")

    try:
        # Generate a unique module name
        module_name = f"dynamic_function_{name}_{os.path.getmtime(file_path)}"

        # Load the module from the file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function from the module
        func = getattr(module, func_name)

        # Create a wrapper for the function
        @functools.wraps(func)
        async def wrapper(tool_args):
            logger.info(f"🚀 EXECUTING DYNAMIC FUNCTION: {name}")
            logger.debug(f"WITH ARGUMENTS: {tool_args}")
            try:
                return await _call_func(func, tool_args)
            except Exception as e:
                logger.error(f"❌ ERROR EXECUTING DYNAMIC FUNCTION {name}: {str(e)}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise e

        # Extract metadata for creating tool schema
        sig = inspect.signature(func)
        params = sig.parameters

        # Get the docstring as the description
        description = func.__doc__ or f"Dynamic function: {name}"

        # Create JSON schema for the function based on its signature
        properties = {}
        required = []

        for param_name, param in params.items():
            # Skip 'self' parameter if present
            if param_name == 'self':
                continue

            # Determine the parameter type
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict or param.annotation == Dict:
                    param_type = "object"
                elif param.annotation == list or param.annotation == List:
                    param_type = "array"

            # Add the parameter to the properties
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}"
            }

            # If the parameter has no default value, it's required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        # Create the tool schema
        schema = {
            "type": "object",
            "properties": properties
        }

        if required:
            schema["required"] = required

        # Get file modification time
        mtime = os.path.getmtime(file_path)
        last_modified = datetime.datetime.fromtimestamp(mtime).isoformat()

        # Create a Tool object for the function with validation status
        tool = Tool(
            name=name,
            description=description.strip(),
            inputSchema=schema,
            annotations={
                "lastModified": last_modified,
                "validationStatus": "VALID"  # If we got this far, it's valid
            }
        )

        # Store the tool in the tools dictionary
        tools[name] = tool

        logger.info(f"✅ FUNCTION {name} LOADED SUCCESSFULLY")
        return wrapper

    except Exception as e:
        logger.error(f"❌ ERROR LOADING FUNCTION {name}: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return

async def _call_func(func, tool_args):
    result = func(**tool_args)
    if inspect.isawaitable(result):
        result = await result
    # Convert result to TextContent if needed
    if isinstance(result, str):
        return [TextContent(type="text", text=result)]
    elif isinstance(result, dict):
        return [TextContent(type="text", text=json.dumps(result))]
    elif isinstance(result, list) and all(isinstance(item, TextContent) for item in result):
        return result
    else:
        return [TextContent(type="text", text=str(result))]


def extract_function_metadata_from_code(code, provided_description=None):
    """
    Extract function name and description from code string.
    Returns a tuple of (name, description)
    """
    # Try to extract function name using regex (more resilient to syntax errors)
    function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    if not function_match:
        raise ValueError("No function definition found in the provided code. Make sure it starts with 'def name(...)'")

    # Extract the name
    name = function_match.group(1)
    logger.debug(f"✅ EXTRACTED FUNCTION NAME: {name}")

    # Use provided description if available
    description = provided_description

    # Otherwise try to extract docstring/comment if present
    if not description:
        docstring_match = re.search(r'def\s+' + re.escape(name) + r'\s*\([^)]*\)\s*:(?:[^"\']|\n)*?"""(.*?)"""', code, re.DOTALL)
        if docstring_match:
            description = docstring_match.group(1).strip()
            logger.debug(f"📄 EXTRACTED DESCRIPTION FROM DOCSTRING")

    return name, description



async def function_register(args, mcp_server):
    """Register a new Python function by inspecting its code and generating metadata."""
    logger.info(f"📝 FUNCTION REGISTER CALLED with args: {args}")

    # Get parameters - only require code
    code = args.get("code")

    # Validate parameters
    if not code:
        raise ValueError("Missing required parameter: code must be provided")

    # IMPORTANT: Do basic validation of the code
    name, extracted_description = extract_function_metadata_from_code(code, "")

    # Create the function file
    function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")

    try:
        # Save the file first - this must always succeed unless there's a disk error
        with open(function_path, 'w') as f:
            f.write(code)
        logger.info(f"✅ WROTE FUNCTION CODE TO: {function_path}")

        # Try to load but expect failure - just capture status
        tool_status = "INVALID"
        load_error = None
        try:
            # Attempt to load/validate
            load_function_from_file(name, function_path, name, mcp_server)
            tool_status = "VALID"
            logger.info(f"✅ FUNCTION {name} VALIDATED SUCCESSFULLY")
        except Exception as e:
            # Just capture the error, don't prevent registration
            load_error = str(e)
            logger.warning(f"⚠️ FUNCTION VALIDATION FAILED: {load_error}")
            
            # Even though load_function_from_file failed, we need to register an invalid tool
            # so it shows up in tool/list with the INVALID status
            try:
                # Create a simple placeholder Tool with INVALID status
                invalid_tool = Tool(
                    name=name,
                    description=f"Invalid function: {name}",
                    inputSchema={"type": "object", "properties": {}},
                    annotations={
                        "lastModified": datetime.datetime.now().isoformat(),
                        "validationStatus": "INVALID",
                        "errorMessage": load_error
                    }
                )
                # Store in tools dictionary so it shows up in tool/list
                tools[name] = invalid_tool
                logger.info(f"⚠️ REGISTERED FUNCTION {name} WITH INVALID STATUS")
            except Exception as tool_error:
                logger.warning(f"⚠️ COULD NOT REGISTER INVALID TOOL: {str(tool_error)}")

        # Send tool list changed notification if possible
        try:
            if hasattr(mcp_server, 'request_context'):
                # Get the current session
                ctx = mcp_server.request_context
                # Create and send a ToolListChangedNotification
                notification = ToolListChangedNotification(
                    method="notifications/tools/list_changed",
                    params=NotificationParams()
                )
                await ctx.session.send_notification(notification)
                logger.info(f"📢 SENT TOOL LIST CHANGED NOTIFICATION")
        except Exception as e:
            # Log a warning but don't fail the function registration
            logger.warning(f"⚠️ COULD NOT SEND TOOL LIST CHANGED NOTIFICATION: {str(e)}")

        logger.info(f"✅ FUNCTION {name} REGISTERED SUCCESSFULLY WITH STATUS: {tool_status}")

        # Return success with validation info
        result_message = f"Function '{name}' saved successfully. Status: {tool_status}"
        if load_error:
            result_message += f" (Error: {load_error})"
        return [TextContent(type="text", text=result_message)]

    except Exception as e:
        logger.error(f"❌ ERROR REGISTERING FUNCTION {name}: {str(e)}")


        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise e

async def get_function_code(args, mcp_server):
    """Get the Python source code and description for a dynamically registered function"""
    logger.info(f"📋 GET FUNCTION CODE CALLED with args: {args}")

    # Get function name
    name = args.get("name")

    # Validate parameters
    if not name:
        raise ValueError("Missing required parameter: name")

    # Get file path for the function
    function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")

    try:
        # Check if the function file exists
        if not os.path.exists(function_path):
            raise ValueError(f"Function {name} not found")

        # Read the code from the file
        with open(function_path, 'r') as f:
            code = f.read()

        name, description = extract_function_metadata_from_code(code, "")

        logger.info(f"✅ RETRIEVED CODE FOR FUNCTION: {name}")

        # Return the code as text content
        return [TextContent(
            type="text",
            text=code,
            annotations=Annotations(
                function_name=name,
                function_description=description
            )
        )]

    except Exception as e:
        logger.error(f"❌ ERROR GETTING FUNCTION CODE FOR {name}: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise e

async def function_remove(args, mcp_server):
    """Remove a dynamically registered function by deleting its file"""
    logger.info(f"🗑️ FUNCTION REMOVE CALLED with args: {args}")

    # Get function name
    name = args.get("name")

    # Validate parameters
    if not name:
        raise ValueError("Missing required parameter: name")

    # Get file path for the function
    function_path = os.path.join(FUNCTIONS_DIR, f"{name}.py")

    try:
        # Check if the function file exists
        if not os.path.exists(function_path):
            raise ValueError(f"Function {name} not found")

        # Remove the function file
        os.remove(function_path)
        logger.info(f"✅ REMOVED FUNCTION FILE: {function_path}")

        # Remove from memory
        if name in dynamic_functions:
            del dynamic_functions[name]
        if name in tools:
            del tools[name]

        # Send tool list changed notification if possible
        try:
            if hasattr(mcp_server, 'request_context'):
                # Get the current session
                ctx = mcp_server.request_context
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

async def function_add(args, mcp_server):
    """Creates a new placeholder function file and registers it."""
    name = args.get("name")
    if not name:
        raise ValueError("Missing required argument for function_add: name")

    logger.info(f"➕ ADDING NEW PLACEHOLDER FUNCTION: {name}")

    # Define the placeholder content with the provided name
    placeholder_code = (
        "# Placeholder function created by function_add\n"
        f"def {name}():\n"
        "    \"\"\"\n"
        "    This is an empty placeholder function. \n"
        "    Replace this with your actual logic.\n"
        "    \"\"\"\n"
        "    print(f\"Executing placeholder function...\")\n"
        "    return \"Placeholder function executed successfully.\"\n"
    )
    placeholder_description = "A newly added placeholder function. Implement your logic here."

    # Prepare arguments for the existing function_register method
    registration_args = {
        "name": name,
        "code": placeholder_code,
        "description": placeholder_description
    }

    try:
        # Call the existing registration logic
        logger.debug(f"Calling internal function_register for placeholder '{name}'...")
        # Use the version for the inspecting register function
        result = await function_register(registration_args, mcp_server)
        logger.info(f"✅ Successfully added placeholder function: {name}")
        # Append a note about it being a placeholder to the success message
        original_message = result[0].text
        result[0].text = f"{original_message} (This is a placeholder, edit {name}.py to implement logic)."
        return result
    except Exception as e:
        logger.error(f"❌ ERROR ADDING PLACEHOLDER FUNCTION {name}: {str(e)}")
        # function_register should handle cleanup, just re-raise
        raise
