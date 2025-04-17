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
        slice_node = node.slice
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
        slice_node = node.slice
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
