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
from state import logger, FUNCTIONS_DIR

async def discover_functions_DONOTUSE(mcp_server):
    """Scan the functions directory and discover all available functions"""
    logger.info(f"🔍 SCANNING FOR FUNCTIONS IN: {FUNCTIONS_DIR}")


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

def extract_metadata_from_file_DONOTUSE(file_path):
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

def load_function_from_file_DONOTUSE(name, file_path, func_name, mcp_server):
    """Load a function from a Python file"""
    logger.info(f"📂 LOADING FUNCTION {name} FROM {file_path}")

    try:
        # Validate the function
        if not validate_function(file_path, func_name):
            logger.warning(f"⚠️ FUNCTION {name} FAILED VALIDATION")
            return

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

def validate_function_DONOTUSE(file_path: str, func_name: str) -> bool:
    try:
        # First check basic syntax
        with open(file_path, 'r') as f:
            ast.parse(f.read())

        # Then try actual import
        spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # This can still fail for runtime errors
        return hasattr(module, func_name)
    except (SyntaxError, ImportError, AttributeError) as e:
        logger.error(f"❌ VALIDATION FAILED: {str(e)}")
        return False

async def _call_func_DONOTUSE(func, tool_args):
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





def extract_function_metadata_from_code_DONOUSE(code, provided_description=None):
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



async def function_register_DONOUSE(args, mcp_server):
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

async def get_function_code_DONOUSE(args, mcp_server):
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

async def function_remove_DONOUSE(args, mcp_server):
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

async def function_add_DONOUSE(args, mcp_server):
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








# === New Utility Functions (Bottom of dynamic.py) ===

import os
import re
import ast
import logging
from typing import Optional, Dict
from werkzeug.utils import secure_filename
from state import FUNCTIONS_DIR, logger # Assuming logger and FUNCTIONS_DIR are defined/imported earlier

# --- 1. File Save/Load ---

def _fs_save_code(name: str, code: str) -> Optional[str]:
    """
    Saves the provided code string to a file named {name}.py in the FUNCTIONS_DIR.
    Uses secure_filename for basic safety. Returns the full path if successful, None otherwise.
    """
    if not name or not isinstance(name, str):
        logger.error("❌ _fs_save_code: Invalid name provided.")
        return None

    safe_name = secure_filename(f"{name}.py")
    if not safe_name.endswith(".py"): # Ensure it's still a python file after securing
         safe_name = f"{name}.py" # Fallback if secure_filename removes extension (less likely)

    file_path = os.path.join(FUNCTIONS_DIR, safe_name)

    try:
        os.makedirs(FUNCTIONS_DIR, exist_ok=True) # Ensure directory exists
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        logger.debug(f"💾 Saved code for '{name}' to {file_path}")
        return file_path
    except IOError as e:
        logger.error(f"❌ _fs_save_code: Failed to write file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ _fs_save_code: Unexpected error saving {file_path}: {e}")
        return None

def _fs_load_code(name: str) -> Optional[str]:
    """
    Loads code from {name}.py in FUNCTIONS_DIR. Returns code string or None if not found/error.
    """
    if not name or not isinstance(name, str):
        logger.error("❌ _fs_load_code: Invalid name provided.")
        return None

    safe_name = secure_filename(f"{name}.py")
    if not safe_name.endswith(".py"):
         safe_name = f"{name}.py"

    file_path = os.path.join(FUNCTIONS_DIR, safe_name)

    if not os.path.exists(file_path):
        logger.warning(f"⚠️ _fs_load_code: File not found for '{name}' at {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        logger.debug(f"💾 Loaded code for '{name}' from {file_path}")
        return code
    except IOError as e:
        logger.error(f"❌ _fs_load_code: Failed to read file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ _fs_load_code: Unexpected error loading {file_path}: {e}")
        return None

# --- 2. Basic Metadata Extraction (Regex) ---

def _code_extract_basic_metadata(code_buffer: str) -> Dict[str, Optional[str]]:
    """
    Extracts function name and description using basic regex from a code string buffer.
    Designed to be resilient to minor syntax errors. Returns {'name': ..., 'description': ...}.
    Values can be None if not found.
    """
    metadata = {'name': None, 'description': None}
    if not code_buffer or not isinstance(code_buffer, str):
        return metadata

    # Regex for function name: def optional_async space+ name space* ( ... ):
    func_match = re.search(r'^\s*(async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_buffer, re.MULTILINE)
    if func_match:
        metadata['name'] = func_match.group(2)
        logger.debug(f"⚙️ Regex extracted name: {metadata['name']}")

        # Regex for the *first* docstring after the function definition line
        # Look for triple quotes after the function signature's closing parenthesis and colon
        # This is simplified and might grab comments if docstring isn't immediate
        docstring_match = re.search(r'def\s+' + re.escape(metadata['name']) + r'\s*\(.*\):\s*"""(.*?) """', code_buffer, re.DOTALL | re.MULTILINE)
        if docstring_match:
            metadata['description'] = docstring_match.group(1).strip()
            logger.debug(f"⚙️ Regex extracted description: {metadata['description'][:50]}...")
        else:
             # Fallback: Look for any initial triple-quoted string as a potential docstring
             fallback_docstring = re.search(r'"""(.*?) """', code_buffer, re.DOTALL)
             if fallback_docstring:
                 metadata['description'] = fallback_docstring.group(1).strip()
                 logger.debug(f"⚙️ Regex extracted fallback description: {metadata['description'][:50]}...")

    else:
         logger.warning("⚠️ _code_extract_basic_metadata: Could not find function definition via regex.")


    return metadata

# --- 3. Full Syntax Validation (AST) ---

def _code_validate_syntax(code_buffer: str) -> bool:
    """
    Validates if the provided code buffer is syntactically correct Python using ast.parse.
    Returns True if valid syntax, False otherwise.
    """
    if not code_buffer or not isinstance(code_buffer, str):
        return False
    try:
        ast.parse(code_buffer)
        logger.debug("⚙️ Code validation successful (AST parse).")
        # Note: This only checks syntax. It doesn't check for runtime errors,
        # correct imports, or if the expected function actually exists or runs.
        # Full runtime validation typically requires execution context or temporary files.
        return True
    except SyntaxError as e:
        logger.warning(f"⚠️ Code validation failed (AST parse): {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during code validation (AST parse): {e}")
        return False

# --- 4. Stub Generation ---

def _code_generate_stub(name: str) -> str:
    """
    Generates a string containing a basic Python function stub with the given name.
    """
    if not name or not isinstance(name, str):
        name = "unnamed_function" # Default name if invalid

    stub = f"""\
# Placeholder function '{name}' generated by the system.
# Replace the contents of this function with your implementation.

def {name}():
    \"\"\"
    This is a placeholder function for '{name}'.
    It currently does nothing but return a message.
    - Add parameters to the function signature if needed.
    - Add type hints for parameters and return values.
    - Implement the actual logic within the function body.
    - Update this docstring to explain what the function does, its parameters, and what it returns.
    \"\"\"
    print(f"Executing placeholder function: {name}...")

    # Replace this return statement with your function's result
    return f"Placeholder function '{name}' executed successfully."

# Example of how you might add parameters and logic:
#
# from typing import Optional
#
# def {name}(input_text: str, max_length: Optional[int] = None):
#     \"\"\"
#     Processes the input text.
#
#     Args:
#         input_text: The text to process.
#         max_length: Optional maximum length for the output.
#
#     Returns:
#         The processed text, or an error message.
#     \"\"\"
#     if max_length is not None and len(input_text) > max_length:
#         return f"Error: Input text exceeds maximum length of {{max_length}}"
#     processed_text = input_text.upper() # Example processing
#     return processed_text

"""
    logger.debug(f"⚙️ Generated code stub for function: {name}")
    return stub
