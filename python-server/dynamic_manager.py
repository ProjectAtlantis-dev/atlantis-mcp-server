# /Users/reinman/work/mcp/atlantis-vm/python-server/dynamic_manager.py
"""
Manages the lifecycle and validation of dynamic functions.
This includes creating, updating, removing, and validating function code.
"""

import os
import importlib.util
import datetime
import traceback
import shutil # Using shutil for potentially more robust file operations
from typing import Optional, Any, Dict, Union, Tuple
from werkzeug.utils import secure_filename
from state import logger, FUNCTIONS_DIR, CYAN, RESET # Import FUNCTIONS_DIR, CYAN, and RESET
import utils # Import our utility module for dynamic functions
import logging
import json
import inspect
import importlib
import importlib.util
import re
import ast
import functools
import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from werkzeug.utils import secure_filename
from mcp.types import Tool, TextContent, CallToolResult, ToolListChangedNotification, NotificationParams, Annotations

# Import shared state
from state import logger, FUNCTIONS_DIR

import re
import ast
import logging
from typing import Optional, Dict
from werkzeug.utils import secure_filename
from state import FUNCTIONS_DIR, logger # Assuming logger and FUNCTIONS_DIR are defined/imported earlier

# Runtime error cache (stores last known runtime error string for a function)
_runtime_errors: Dict[str, str] = {}

# --- 1. File Save/Load ---

# Ensure FUNCTIONS_DIR exists at startup (or wherever appropriate)
os.makedirs(FUNCTIONS_DIR, exist_ok=True)
OLD_DIR = os.path.join(FUNCTIONS_DIR, "OLD")
os.makedirs(OLD_DIR, exist_ok=True)


def _fs_save_code(name: str, code: str) -> Optional[str]:
    """
    Saves the provided code string to a file named {name}.py in the FUNCTIONS_DIR.
    Uses secure_filename for basic safety. Returns the full path if successful, None otherwise.
    """
    if not name or not isinstance(name, str):
        logger.error("❌ _fs_save_code: Invalid name provided.")
        return None

    safe_name = utils.clean_filename(f"{name}.py")
    if not safe_name.endswith(".py"): # Ensure it's still a python file after securing
         safe_name = f"{name}.py" # Fallback if secure_filename removes extension (less likely)

    file_path = os.path.join(FUNCTIONS_DIR, safe_name)

    try:
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

    safe_name = utils.clean_filename(f"{name}.py")
    if not safe_name.endswith(".py"):
         safe_name = f"{name}.py"

    file_path = os.path.join(FUNCTIONS_DIR, safe_name)

    if not os.path.exists(file_path):
        logger.warning(f"⚠️ _fs_load_code: File not found for '{name}' at {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        logger.info(f"{CYAN}📋 === LOADING {name} ==={RESET}")

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

# --- AST Helper Functions ---

def _ast_node_to_string(node: Optional[ast.expr]) -> str:
    """Attempt to reconstruct a string representation of an AST node (for type hints)."""
    if node is None:
        return "Any"
    # Use ast.unparse if available (Python 3.9+) for better accuracy
    if hasattr(ast, 'unparse'):
        try:
            return ast.unparse(node)
        except Exception:
            pass # Fallback to manual reconstruction if unparse fails

    # Manual reconstruction (simplified, fallback for <3.9 or unparse errors)
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return repr(node.value) # e.g., 'None' for NoneType
    elif isinstance(node, ast.Attribute):
        value_str = _ast_node_to_string(node.value)
        return f"{value_str}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        value_str = _ast_node_to_string(node.value)
        # Handle slice difference between Python versions
        slice_node = node.slice # Corrected variable name
        if hasattr(ast, 'Index') and isinstance(slice_node, ast.Index): # Python < 3.9
             slice_inner_node = slice_node.value
        else: # Python 3.9+
             slice_inner_node = slice_node

        slice_str = _ast_node_to_string(slice_inner_node)
        return f"{value_str}[{slice_str}]"
    elif isinstance(node, ast.Tuple): # For Tuple[A, B] or Union[A, B] slices
        elements = ", ".join([_ast_node_to_string(el) for el in node.elts])
        return f"({elements})" # Representing the structure, not direct type name
    else:
        return "ComplexType"


def _map_ast_type_to_json_schema(annotation_node: Optional[ast.expr]) -> Dict[str, Any]:
    """Maps an AST annotation node to a JSON Schema type component."""
    if annotation_node is None:
        # Default to string if no type hint is provided, as it's common for text-based inputs
        # Alternatively, could use "any" or make it required implicitly if desired.
        return {"type": "string", "description": "Type hint missing, assuming string"}

    # Simple Name types (str, int, etc.)
    if isinstance(annotation_node, ast.Name):
        type_name = annotation_node.id
        if type_name == 'str':
            return {"type": "string"}
        elif type_name == 'int':
            return {"type": "integer"}
        elif type_name == 'float' or type_name == 'complex': # Treat complex as number
            return {"type": "number"}
        elif type_name == 'bool':
            return {"type": "boolean"}
        elif type_name == 'list' or type_name == 'List':
            return {"type": "array"}
        elif type_name == 'dict' or type_name == 'Dict':
            return {"type": "object"}
        elif type_name == 'Any':
             # "any" isn't a standard JSON schema type. Use object without properties? Or skip type field?
             # Let's allow anything but describe it.
             return {"description": "Any type allowed"}
        else:
            # Assume custom object or unhandled simple type
            return {"type": "object", "description": f"Assumed object type: {type_name}"}

    # Constant None (NoneType)
    elif isinstance(annotation_node, ast.Constant) and annotation_node.value is None:
        return {"type": "null"}

    # Subscript types (List[T], Optional[T], Dict[K, V], Union[A, B])
    elif isinstance(annotation_node, ast.Subscript):
        container_node = annotation_node.value
        # Handle slice difference between Python versions
        slice_node = annotation_node.slice # Corrected variable name
        if hasattr(ast, 'Index') and isinstance(slice_node, ast.Index): # Python < 3.9
             slice_inner_node = slice_node.value
        else: # Python 3.9+
             slice_inner_node = slice_node

        container_name = _ast_node_to_string(container_node) # e.g., 'List', 'Optional', 'Union', 'Dict'

        # Extract inner types from the slice (could be single type or a tuple)
        inner_nodes = []
        if isinstance(slice_inner_node, ast.Tuple):
            inner_nodes = slice_inner_node.elts
        else:
            inner_nodes = [slice_inner_node]

        # Map common container types
        if container_name in ['List', 'list', 'Sequence', 'Iterable', 'Set', 'set']:
            if inner_nodes and inner_nodes[0] is not None:
                items_schema = _map_ast_type_to_json_schema(inner_nodes[0])
                return {"type": "array", "items": items_schema}
            else:
                return {"type": "array"} # List without specified item type
        elif container_name in ['Dict', 'dict', 'Mapping']:
             if len(inner_nodes) == 2 and inner_nodes[1] is not None:
                  # JSON Schema typically uses additionalProperties for value type
                  value_schema = _map_ast_type_to_json_schema(inner_nodes[1])
                  # Key type (inner_nodes[0]) is usually string in JSON
                  return {"type": "object", "additionalProperties": value_schema}
             else:
                  return {"type": "object"} # Dict without specified types
        elif container_name == 'Optional':
             if inner_nodes and inner_nodes[0] is not None:
                  schema = _map_ast_type_to_json_schema(inner_nodes[0])
                  # Make it nullable: allow original type or null
                  existing_types = []
                  if 'type' in schema:
                      existing_types = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
                  elif 'anyOf' in schema: # If inner type was already a Union
                       # Add null to the existing anyOf if not present
                       if not any(t.get('type') == 'null' for t in schema['anyOf']):
                            schema['anyOf'].append({'type': 'null'})
                       return schema
                  else:
                       # Fallback if schema is complex (e.g., just a description)
                       return {'anyOf': [schema, {'type': 'null'}]}

                  if 'null' not in existing_types:
                      existing_types.append('null')
                  schema['type'] = existing_types
                  return schema
             else:
                  # Optional without inner type, allow anything or null
                  return {"type": ["any", "null"], "description":"Optional type specified without inner type"}
        elif container_name == 'Union':
            schemas = [_map_ast_type_to_json_schema(node) for node in inner_nodes if node is not None]
            # Simplify if it reduces to Optional[T] (Union[T, None])
            non_null_schemas = [s for s in schemas if s.get('type') != 'null']
            has_null = len(schemas) > len(non_null_schemas)

            if len(non_null_schemas) == 1:
                 final_schema = non_null_schemas[0]
                 if has_null: # Make it nullable if None was part of the Union
                      existing_types = []
                      if 'type' in final_schema:
                          existing_types = final_schema['type'] if isinstance(final_schema['type'], list) else [final_schema['type']]
                      elif 'anyOf' in final_schema:
                          if not any(t.get('type') == 'null' for t in final_schema['anyOf']):
                               final_schema['anyOf'].append({'type': 'null'})
                          return final_schema
                      else: # Fallback
                          return {'anyOf': [final_schema, {'type': 'null'}]}

                      if 'null' not in existing_types:
                           existing_types.append('null')
                      final_schema['type'] = existing_types
                 return final_schema
            elif len(non_null_schemas) > 1:
                 # True Union[A, B, ...]
                 result_schema = {"anyOf": non_null_schemas}
                 if has_null: # Add null possibility if None was in the Union
                      result_schema['anyOf'].append({'type': 'null'})
                 return result_schema
            elif has_null: # Only None was in the Union?
                 return {'type': 'null'}
            else: # Empty Union?
                 return {}

        else:
             # Unhandled subscript type (e.g., Tuple[...], custom generics)
             type_str = _ast_node_to_string(annotation_node)
             return {"description": f"Unhandled generic type: {type_str}"}

    # Fallback for other node types (e.g., ast.BinOp used in type hints?)
    else:
        type_str = _ast_node_to_string(annotation_node)
        return {"description": f"Unknown type structure: {type_str}"}


def _ast_arguments_to_json_schema(args_node: ast.arguments, docstring: Optional[str] = None) -> Dict[str, Any]:
    """Builds the JSON Schema 'properties' and 'required' fields from AST arguments."""
    properties = {}
    required = []
    parsed_doc_params = {}

    # Basic docstring parsing for parameter descriptions
    if docstring:
        lines = docstring.strip().split('\n')
        param_section = False
        current_param_desc = {}
        for line in lines:
            clean_line = line.strip()
            # Detect start of common param sections
            if clean_line.lower().startswith(('args:', 'arguments:', 'parameters:')):
                 param_section = True
                 continue
             # Stop if we hit return section
            if clean_line.lower().startswith(('returns:', 'yields:')):
                 param_section = False
                 continue

            # Detect common param formats
            # Simple: ":param name: description"
            match_param = re.match(r':param\s+(\w+)\s*:(.*)', clean_line)
            # Typed: "name (type): description" or "name: type\n    description"
            match_typed = re.match(r'(\w+)\s*(?:\(.*\))?\s*:(.*)', clean_line) # Basic check for name: desc

            if match_param:
                 name = match_param.group(1)
                 desc = match_param.group(2).strip()
                 current_param_desc[name] = desc
                 param_section = True # Assume params follow sequentially
            elif param_section and match_typed:
                 name = match_typed.group(1)
                 desc = match_typed.group(2).strip()
                 # If description is empty, it might be on the next line (indented)
                 # This simple parser doesn't handle multi-line descriptions well.
                 current_param_desc[name] = desc if desc else current_param_desc.get(name, '') # Keep previous if empty
            elif param_section and clean_line and not clean_line.startswith(':'):
                 # Assume continuation of the previous param description (basic handling)
                 last_param = next(reversed(current_param_desc), None)
                 if last_param:
                      current_param_desc[last_param] += " " + clean_line

        parsed_doc_params = current_param_desc


    # --- Process Arguments ---
    all_args = args_node.posonlyargs + args_node.args
    num_defaults = len(args_node.defaults)
    defaults_start_index = len(all_args) - num_defaults

    for i, arg in enumerate(all_args):
        name = arg.arg
        param_schema = _map_ast_type_to_json_schema(arg.annotation)
        param_schema["description"] = parsed_doc_params.get(name, param_schema.get("description", "")) # Add docstring desc

        properties[name] = param_schema

        # Check if it's required (no default value)
        has_default = i >= defaults_start_index
        if not has_default:
            required.append(name)

    # Process kwonlyargs
    for i, arg in enumerate(args_node.kwonlyargs):
        name = arg.arg
        param_schema = _map_ast_type_to_json_schema(arg.annotation)
        param_schema["description"] = parsed_doc_params.get(name, param_schema.get("description", ""))

        properties[name] = param_schema

        # Check if it's required (kw_defaults[i] is None means no default provided)
        if i < len(args_node.kw_defaults) and args_node.kw_defaults[i] is None:
            required.append(name)
        elif i >= len(args_node.kw_defaults): # Should have a default or None
            required.append(name)

    # Ignore *args (args_node.vararg) and **kwargs (args_node.kwarg)

    return {"properties": properties, "required": required}

# --- End of Helper Functions ---

def _code_validate_syntax(code_buffer: str) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates syntax using ast.parse and extracts info about the *first* function definition found.

    Returns:
        tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        - is_valid (bool): True if syntax is correct.
        - error_message (Optional[str]): Error details if invalid, None otherwise.
        - function_info (Optional[Dict[str, Any]]):
            Dict with 'name', 'description', 'inputSchema' if valid and a function is found,
            None otherwise.
    """
    if not code_buffer or not isinstance(code_buffer, str):
        return False, "Empty or invalid code buffer", None

    try:
        tree = ast.parse(code_buffer)
        logger.debug("⚙️ Code validation successful (AST parse).")

        func_def_node = None
        # Find the first top-level function definition
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_def_node = node
                break

        if func_def_node:
            logger.debug(f"⚙️ Found function definition: {func_def_node.name}")
            func_name = func_def_node.name
            docstring = ast.get_docstring(func_def_node)
            input_schema = {"type": "object"} # Default empty schema

            # Generate schema from arguments
            try:
                 schema_parts = _ast_arguments_to_json_schema(func_def_node.args, docstring)
                 input_schema["properties"] = schema_parts.get("properties", {})
                 input_schema["required"] = schema_parts.get("required", [])
            except Exception as schema_e:
                 logger.warning(f"⚠️ Could not generate input schema for {func_name}: {schema_e}")
                 input_schema["description"] = f"Schema generation error: {schema_e}"

            function_info = {
                "name": func_name,
                "description": docstring or "(No description provided)", # Provide default
                "inputSchema": input_schema
            }
            return True, None, function_info
        else:
            logger.warning("⚠️ Syntax valid, but no top-level function definition found.")
            return True, "Syntax valid, but no function definition found", None

    except SyntaxError as e:
        # Get detailed error information
        error_msg = f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
        if hasattr(e, 'text') and e.text:
            # Show the problematic line if available
            error_msg += f"\nLine content: {e.text.strip()}"
            if e.offset:
                # Add a pointer to the exact error position
                error_msg += f"\n{' ' * (e.offset-1)}^"
        logger.warning(f"⚠️ Code validation failed (AST parse): {error_msg}")
        return False, error_msg, None
    except Exception as e:
        error_msg = f"Unexpected error during validation or AST processing: {str(e)}"
        logger.error(f"❌ {error_msg}\n{traceback.format_exc()}") # Log full traceback
        return False, error_msg, None

# --- 4. Stub Generation ---

def _code_generate_stub(name: str) -> str:
    """
    Generates a string containing a basic Python function stub with the given name.
    """
    if not name or not isinstance(name, str):
        name = "unnamed_function" # Default name if invalid

    stub = f"""\
def {name}():
    \"\"\"
    This is a placeholder function for '{name}'
    \"\"\"
    print(f"Executing placeholder function: {name}...")

    # Replace this return statement with your function's result
    return f"Placeholder function '{name}' executed successfully."

"""
    logger.debug(f"⚙️ Generated code stub for function: {name}")
    return stub




def function_add(name: str, code: Optional[str] = None) -> bool:
    '''
    Creates a new function file.
    If code is provided, it saves it. Otherwise, generates and saves a stub.
    Returns True on success, False if the function already exists or on error.
    '''
    secure_name = utils.clean_filename(name)
    if not secure_name:
        logger.error(f"Create failed: Invalid function name '{name}'")
        return False
    file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")

    if os.path.exists(file_path):
        logger.warning(f"Create failed: Function '{secure_name}' already exists.")
        return False

    try:
        code_to_save = code if code is not None else _code_generate_stub(secure_name)
        if _fs_save_code(secure_name, code_to_save):
            logger.info(f"Function '{secure_name}' created successfully.")
            return True
        else:
            logger.error(f"Create failed: Could not save code for '{secure_name}'.")
            return False
    except Exception as e:
        logger.error(f"Error during function creation for '{secure_name}': {e}")
        logger.debug(traceback.format_exc())
        return False


def function_remove(name: str) -> bool:
    '''
    Removes a function file by moving it to the OLD subdirectory.
    Returns True on success, False if the function doesn't exist or on error.
    '''
    secure_name = utils.clean_filename(name)
    if not secure_name:
        logger.error(f"Remove failed: Invalid function name '{name}'")
        return False
    file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")

    if not os.path.exists(file_path):
        logger.warning(f"Remove failed: Function '{secure_name}' does not exist.")
        return False

    try:
        # First, clear any potential old runtime error cache for this name
        _runtime_errors.pop(name, None)

        # Move to OLD directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        backup_path = os.path.join(OLD_DIR, f"{secure_name}_{timestamp}.py.bak")
        shutil.move(file_path, backup_path)
        logger.info(f"Function '{secure_name}' removed (moved to '{backup_path}')")
        # Remove corresponding log file if it exists
        log_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.log")
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
                logger.info(f"Removed log file for '{secure_name}' at {log_path}")
            except Exception as e:
                logger.error(f"Failed to remove log file for '{secure_name}': {e}")

        return True
    except Exception as e:
        logger.error(f"Error during function removal for '{secure_name}': {e}")
        logger.debug(traceback.format_exc())
        return False

def _write_error_log(name: str, error_message: str) -> None:
    '''
    Write an error message to a function-specific log file in the dynamic_functions folder.
    Overwrites any existing log to only keep the latest error.
    Creates a log file named {name}.log with timestamp.
    '''
    try:
        secure_name = utils.clean_filename(name)
        log_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Open in write mode to overwrite previous content
        with open(log_path, 'w') as log_file:
            log_file.write(f"{timestamp} [ERROR] {error_message}\n")

        logger.debug(f"Wrote error log for '{secure_name}' at {log_path}")
    except Exception as e:
        # Don't let logging errors disrupt the main flow
        logger.error(f"Failed to write error log for '{name}': {e}")

def function_validate(name: str) -> Dict[str, Any]:
    '''
    Validates the syntax of a function file without executing it.
    Returns a dictionary {'valid': bool, 'error': Optional[str], 'function_info': Optional[Dict]}
    with detailed error messages and extracted function details on success.
    '''
    secure_name = utils.clean_filename(name)
    if not secure_name:
        error_msg = f"Invalid function name '{name}'"
        _write_error_log(name, error_msg)
        return {'valid': False, 'error': error_msg, 'function_info': None}

    code = _fs_load_code(secure_name)
    if code is None:
        error_msg = f"Function '{secure_name}' not found or could not be read."
        _write_error_log(name, error_msg)
        return {'valid': False, 'error': error_msg, 'function_info': None}

    # _code_validate_syntax now returns: (is_valid, error_message, function_info)
    is_valid, error_message, function_info = _code_validate_syntax(code)

    if is_valid:
        # Successful validation
        logger.info(f"Syntax validation successful for function '{secure_name}'")

        # If there was a previous error log, remove it since the function is now valid
        try:
            log_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.log")
            if os.path.exists(log_path):
                os.remove(log_path)
                logger.debug(f"Removed error log for '{secure_name}' as validation now passes")
        except Exception as e:
            logger.debug(f"Failed to remove old error log for '{secure_name}': {e}")

        # Return success and the extracted function info
        return {'valid': True, 'error': None, 'function_info': function_info}
    else:
        # Failed validation - write to the error log
        error_msg_full = f"Syntax validation failed: {error_message}"
        logger.warning(f"{error_msg_full} Function: '{secure_name}'")
        _write_error_log(secure_name, error_msg_full)

        # Return the detailed error message
        return {'valid': False, 'error': error_message, 'function_info': None}

import inspect # Add import

async def function_call(name: str, client_id: str, request_id: str, **kwargs) -> Any:
    '''
    Loads and executes a dynamic function by its name, passing kwargs.
    Returns the function's return value.
    Raises exceptions if the function doesn't exist, fails to load, or errors during execution.
    '''
    secure_name = utils.clean_filename(name)
    if not secure_name:
        raise ValueError(f"Invalid function name '{name}' for calling.")

    file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dynamic function '{secure_name}' not found at {file_path}")

    try:
        # Clear any previous runtime error before attempting execution
        _runtime_errors.pop(name, None)

        # Dynamically import the module
        # The module name should be unique, using the file name is common
        module_name = f"dynamic_functions.{secure_name}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {secure_name}")

        module = importlib.util.module_from_spec(spec)
        # Add to sys.modules *before* execution to handle relative imports within the func
        # sys.modules[module_name] = module # Be careful with namespace pollution if many functions

        # Inject our utils module into the module's globals
        # This makes client_log available to the dynamic function without explicit imports
        bound_client_log = functools.partial(utils.client_log, request_id=request_id, client_id_for_routing=client_id)
        logger.debug(f"Bound client_log with request_id: {request_id}, client_id_for_routing: {client_id}")
        module.__dict__['client_log'] = bound_client_log

        spec.loader.exec_module(module)
        logger.info(f"Dynamically loaded module for function '{secure_name}'.")

        # Assume the function name inside the file matches the filename
        func_to_call = getattr(module, secure_name, None)

        if not callable(func_to_call):
            # Maybe try to find *any* function if name doesn't match?
            # For now, strictly enforce matching name.
            raise AttributeError(f"Function '{secure_name}' not found or not callable within its module.")

        logger.info(f"Calling dynamic function '{secure_name}' with args: {kwargs}")
        # Check if the function is async and await it if necessary
        if inspect.iscoroutinefunction(func_to_call):
            result = await func_to_call(**kwargs['args'])
        else:
            result = func_to_call(**kwargs['args']) # Call the function normally

        logger.info(f"Dynamic function '{secure_name}' executed successfully.")
        return result

    except Exception as e:
        error_message = f"❌ function_call: Error executing function '{name}': {traceback.format_exc()}"
        logger.error(error_message)
        # Cache the runtime error string
        _runtime_errors[name] = str(e)
        raise # Re-raise the exception so the caller knows it failed
    finally:
        pass

async def function_set(args: Dict[str, Any], server: Any) -> Tuple[Optional[str], List[TextContent]]:
    """
    Handles the _function_set tool call.
    Extracts the function name using basic regex, saves the provided code.
    Returns the extracted function name (if successful) and a status message.
    Does *not* perform full syntax validation before saving.
    """
    logger.info("⚙️ Handling _function_set call (using basic name extraction)")
    code_buffer = args.get("code")
    extracted_function_name: Optional[str] = None # Keep track of extracted name

    if not code_buffer or not isinstance(code_buffer, str):
        logger.warning("⚠️ function_set: Missing or invalid 'code' parameter.")
        # Return None for name, and the error message
        return None, [TextContent(type="text", text="Error: Missing or invalid 'code' parameter.")]

    # 1. Extract function name using basic regex
    metadata = _code_extract_basic_metadata(code_buffer)
    extracted_function_name = metadata.get('name') # Store extracted name

    if not extracted_function_name:
        error_response = "Error: Could not extract function name from the provided code using basic parsing. Ensure it starts with 'def function_name(...):'"
        logger.warning(f"⚠️ function_set: Failed to extract name via regex.")
         # Return None for name, and the error message
        return None, [TextContent(type="text", text=error_response)]

    logger.info(f"⚙️ Extracted function name via regex: {extracted_function_name}")

    # --- Backup existing file before saving new one ---
    secure_name = utils.clean_filename(extracted_function_name)
    if secure_name: # Should always be true if extracted_function_name is valid
        file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")
        if os.path.exists(file_path):
            logger.info(f"💾 Found existing file for '{secure_name}', attempting backup...")
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                # Using .py.bak for clarity
                backup_filename = f"{secure_name}_{timestamp}.py.bak"
                backup_path = os.path.join(OLD_DIR, backup_filename)
                shutil.copy2(file_path, backup_path) # copy2 preserves metadata
                logger.info(f"🛡️ Successfully backed up '{secure_name}' to '{backup_path}'")
            except Exception as e:
                logger.error(f"❌ Failed to backup existing file '{file_path}' to OLD folder: {e}")
                # Log error but continue, saving the new file might still be desired
        else:
            logger.info(f"ⓘ No existing file found for '{secure_name}', creating new file.")
    else:
        # This case should ideally not happen if name extraction was successful
        logger.warning("⚠️ Could not create secure filename for backup check in function_set.")
    # --- End Backup ---

    # 2. Save the code (validation will happen later when tools are listed/called)
    saved_path = _fs_save_code(extracted_function_name, code_buffer)

    if not saved_path:
        error_response = f"Error saving function '{extracted_function_name}' to file."
        logger.error(f"❌ function_set: {error_response}")
         # Return extracted name (as we got this far), but with error message
        return extracted_function_name, [TextContent(type="text", text=error_response)]

    logger.info(f"💾 Function '{extracted_function_name}' code saved successfully to {saved_path}")

    # Clear any cached runtime error for this function, as it's been updated
    _runtime_errors.pop(extracted_function_name, None)

    # 3. Attempt AST parsing for immediate feedback (but save regardless)
    syntax_error = None
    try:
        ast.parse(code_buffer)
        logger.info(f"✅ Basic syntax validation (AST parse) successful for '{extracted_function_name}'.")
    except SyntaxError as e:
        syntax_error = str(e)
        logger.warning(f"⚠️ Basic syntax validation (AST parse) failed for '{extracted_function_name}': {syntax_error}")

    # 4. Clear cache (server needs to reload tools)
    logger.info(f"🧹 Clearing tool cache on server due to function_set for '{extracted_function_name}'.")
    server._cached_tools = None
    server._last_functions_dir_mtime = None # Reset mtime to force reload
    server._last_servers_dir_mtime = None # Reset mtime to force reload

    # 5. Prepare success message, including validation status
    save_status = f"Function '{extracted_function_name}' saved."
    annotations = None # Default to no annotations
    if syntax_error:
        # If validation failed, add structured error to annotations
        validation_status = f"WARNING: Validation failed."
        response_message = f"{save_status} {validation_status}" # Keep text informative
        annotations = {
            "validationStatus": "ERROR",
            "validationMessage": syntax_error
        }
        logger.warning(f"⚠️ {response_message}")
    else:
        # If validation succeeded
        response_message = f"{save_status} Validation successful."
        logger.info(f"✅ {response_message}")

    # Return TextContent with text and potentially annotations
    return extracted_function_name, [TextContent(type="text", text=response_message, annotations=annotations)]


## Dynamic Function Calling
