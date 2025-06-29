#!/usr/bin/env python3
"""
Class-based manager for dynamic functions lifecycle and validation.
Implements creating, updating, removing, and validating function code.
"""

import os
import sys
import re
import ast
import json
import asyncio
import logging
import pathlib
import shutil
import datetime
import atlantis
import functools
import inspect
import importlib
import traceback

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from werkzeug.utils import secure_filename

from state import logger

from ColoredFormatter import CYAN, RESET

from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    ToolListChangedNotification,
    NotificationParams,
    Annotations,
)

import utils  # Utility module for dynamic functions


# Directory to store dynamic function files
PARENT_PACKAGE_NAME = "dynamic_functions"

# --- Identity Decorator Definition ---
def _mcp_identity_decorator(f):
    """A simple identity decorator that returns the function unchanged. Used as a placeholder for @chat, @public, etc."""
    return f

# --- App Decorator Definition ---
def app(name: str):
    """Decorator to associate a dynamic function with an application name.
    Usage: @app(name="your_app_name")
    """
    def decorator(func_to_decorate):
        setattr(func_to_decorate, '_app_name', name)
        # functools.update_wrapper(decorator, func_to_decorate) # Not strictly needed for AST parsing but good practice
        return func_to_decorate
    return decorator

# Make the app decorator available for dynamic functions to import/use.
# This is a simplified way; a more robust way might involve adding it to a shared module
# that dynamic functions can import from, or injecting it into their global scope upon loading.
# For now, this definition here allows _code_validate_syntax to recognize it by name 'app'.

# --- Shared Module Decorator Definition ---
def shared(func_or_module):
    """
    Decorator that marks a function or module as 'shared'. 
    When applied, the function/module will not be reloaded when dynamic functions are invalidated.
    
    Usage: @shared
           def my_persistent_function():
               # This function will maintain its state
               ...
    """
    # Mark the function/module as shared by setting an attribute
    setattr(func_or_module, '_is_shared', True)
    return func_or_module

# --- Hidden Decorator Definition ---
def hidden(func):
    """
    Decorator that marks a function as 'hidden'.
    When applied, the function will not be included in the tools list.
    
    Usage: @hidden
           def my_hidden_function():
               # This function will not be visible in tools/list
               ...
    """
    # Mark the function as hidden by setting an attribute
    setattr(func, '_is_hidden', True)
    return func

class DynamicFunctionManager:
    def __init__(self, functions_dir):
        # State that was previously global
        self.functions_dir = functions_dir
        self._runtime_errors = {}
        self._dynamic_functions_cache = {}
        self._dynamic_load_lock = asyncio.Lock()


        # Create directories if they don't exist
        os.makedirs(self.functions_dir, exist_ok=True)
        self.old_dir = os.path.join(self.functions_dir, "OLD")
        os.makedirs(self.old_dir, exist_ok=True)

    # File operations
    async def _fs_save_code(self, name: str, code: str) -> Optional[str]:
        """
        Saves the provided code string to a file named {name}.py in the functions directory.
        Uses clean_filename for basic safety. Returns the full path if successful, None otherwise.
        """
        if not name or not isinstance(name, str):
            logger.error("❌ _fs_save_code: Invalid name provided.")
            return None

        safe_name = utils.clean_filename(f"{name}.py")
        if not safe_name.endswith(".py"): # Ensure it's still a python file after securing
             safe_name = f"{name}.py" # Fallback if clean_filename removes extension (less likely)

        file_path = os.path.join(self.functions_dir, safe_name)

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

    async def _fs_load_code(self, name):
        """
        Loads code from {name}.py in self.functions_dir. Returns code string or None if not found/error.
        """
        if not name or not isinstance(name, str):
            logger.error("❌ _fs_load_code: Invalid name provided.")
            return None

        safe_name = utils.clean_filename(f"{name}.py")
        if not safe_name.endswith(".py"): # Ensure it's still a python file after securing
            safe_name = f"{name}.py" # Fallback if clean_filename removes extension

        file_path = os.path.join(self.functions_dir, safe_name) # Use self.functions_dir

        if not os.path.exists(file_path):
            logger.warning(f"⚠️ _fs_load_code: File not found for '{name}' at {file_path}")
            raise FileNotFoundError(f"Function '{name}' not found at {file_path}")

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


    # Metadata extraction and validation
    def _code_extract_basic_metadata(self, code_buffer):
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
            # More robust and simpler approach to find docstrings
            # First find the function definition
            fn_def_pattern = r'def\s+' + re.escape(metadata['name']) + r'\s*\(.*?\)\s*:\s*'
            fn_pos = re.search(fn_def_pattern, code_buffer, re.DOTALL)

            docstring_match = None
            if fn_pos:
                # Get the position right after the function signature
                start_pos = fn_pos.end()
                # Look for the first docstring after the function signature
                docstring_pattern = r'\s*"""(.*?)"""'
                docstring_match = re.search(docstring_pattern, code_buffer[start_pos:], re.DOTALL)

                if docstring_match:
                    metadata['description'] = docstring_match.group(1).strip()
                    logger.debug(f"⚙️ Regex extracted description: {metadata['description'][:50]}...")

            # If we couldn't find a docstring after function def, try fallback
            if not docstring_match or not metadata['description']:
                # Fallback: Look for any triple-quoted string near the function definition
                full_func_pattern = f"def\\s+{re.escape(metadata['name'])}.*?\"\"\"(.*?)\"\"\""
                fallback_docstring = re.search(full_func_pattern, code_buffer, re.DOTALL)
                if fallback_docstring:
                    metadata['description'] = fallback_docstring.group(1).strip()
                    logger.debug(f"⚙️ Regex extracted fallback description: {metadata['description'][:50]}...")
                else:
                    # Last resort fallback: any docstring-like pattern in the function
                    simple_fallback = re.search(r'"""(.*?)"""', code_buffer, re.DOTALL)
                    if simple_fallback:
                        metadata['description'] = simple_fallback.group(1).strip()
                        logger.debug(f"⚙️ Regex extracted simple fallback description: {metadata['description'][:50]}...")

        else:
            logger.warning("⚠️ _code_extract_basic_metadata: Could not find function definition via regex.")


        return metadata


    def _ast_node_to_string(self, node: Optional[ast.expr]) -> str:
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
            value_str = self._ast_node_to_string(node.value)
            return f"{value_str}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value_str = self._ast_node_to_string(node.value)
            # Handle slice difference between Python versions
            slice_node = node.slice # Corrected variable name
            if hasattr(ast, 'Index') and isinstance(slice_node, ast.Index): # Python < 3.9
                slice_inner_node = slice_node.value
            else: # Python 3.9+
                slice_inner_node = slice_node

            slice_str = self._ast_node_to_string(slice_inner_node)
            return f"{value_str}[{slice_str}]"
        elif isinstance(node, ast.Tuple): # For Tuple[A, B] or Union[A, B] slices
            elements = ", ".join([self._ast_node_to_string(el) for el in node.elts])
            return f"({elements})" # Representing the structure, not direct type name
        else:
            return "ComplexType"


    def _map_ast_type_to_json_schema(self, annotation_node: Optional[ast.expr]) -> Dict[str, Any]:
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

            container_name = self._ast_node_to_string(container_node) # e.g., 'List', 'Optional', 'Union', 'Dict'

            # Extract inner types from the slice (could be single type or a tuple)
            inner_nodes = []
            if isinstance(slice_inner_node, ast.Tuple):
                inner_nodes = slice_inner_node.elts
            else:
                inner_nodes = [slice_inner_node]

            # Map common container types
            if container_name in ['List', 'list', 'Sequence', 'Iterable', 'Set', 'set']:
                if inner_nodes and inner_nodes[0] is not None:
                    items_schema = self._map_ast_type_to_json_schema(inner_nodes[0])
                    return {"type": "array", "items": items_schema}
                else:
                    return {"type": "array"} # List without specified item type
            elif container_name in ['Dict', 'dict', 'Mapping']:
                if len(inner_nodes) == 2 and inner_nodes[1] is not None:
                    # JSON Schema typically uses additionalProperties for value type
                    value_schema = self._map_ast_type_to_json_schema(inner_nodes[1])
                    # Key type (inner_nodes[0]) is usually string in JSON
                    return {"type": "object", "additionalProperties": value_schema}
                else:
                    return {"type": "object"} # Dict without specified types
            elif container_name == 'Optional':
                if inner_nodes and inner_nodes[0] is not None:
                    schema = self._map_ast_type_to_json_schema(inner_nodes[0])
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
                schemas = [self._map_ast_type_to_json_schema(node) for node in inner_nodes if node is not None]
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
                type_str = self._ast_node_to_string(annotation_node)
                return {"description": f"Unhandled generic type: {type_str}"}

        # Fallback for other node types (e.g., ast.BinOp used in type hints?)
        else:
            type_str = self._ast_node_to_string(annotation_node)
            return {"description": f"Unknown type structure: {type_str}"}


    def _ast_arguments_to_json_schema(self, args_node: ast.arguments, docstring: Optional[str] = None) -> Dict[str, Any]:
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
            param_schema = self._map_ast_type_to_json_schema(arg.annotation)
            param_schema["description"] = parsed_doc_params.get(name, param_schema.get("description", "")) # Add docstring desc

            properties[name] = param_schema

            # Check if it's required (no default value)
            has_default = i >= defaults_start_index
            if not has_default:
                required.append(name)

        # Process kwonlyargs
        for i, arg in enumerate(args_node.kwonlyargs):
            name = arg.arg
            param_schema = self._map_ast_type_to_json_schema(arg.annotation)
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



    def _code_validate_syntax(self, code_buffer):
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

                # Extract decorators and app_name
                decorator_names = []
                app_name_from_decorator = None # Initialize app_name
                if func_def_node.decorator_list:
                    for decorator_node in func_def_node.decorator_list:
                        if isinstance(decorator_node, ast.Name): # e.g. @public, @hidden
                            decorator_name = decorator_node.id
                            decorator_names.append(decorator_name)
                        elif isinstance(decorator_node, ast.Call): # e.g. @app(name="foo") or @app("foo")
                            if isinstance(decorator_node.func, ast.Name):
                                decorator_func_name = decorator_node.func.id
                                if decorator_func_name == 'app':
                                    # Extract 'name' argument from @app(name="...") or @app("...")
                                    if decorator_node.keywords: # Check keyword arguments like name="foo"
                                        for kw in decorator_node.keywords:
                                            if kw.arg == 'name' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                                if app_name_from_decorator is not None:
                                                    logger.warning(f"⚠️ Multiple @app name specifications for {func_def_node.name}. Using first one: {app_name_from_decorator}")
                                                else:
                                                    app_name_from_decorator = kw.value.value
                                    # Positional arguments like @app("foo")
                                    if not app_name_from_decorator and decorator_node.args:
                                        if len(decorator_node.args) == 1 and isinstance(decorator_node.args[0], ast.Constant) and isinstance(decorator_node.args[0].value, str):
                                            if app_name_from_decorator is not None: # Should not happen if logic is correct, but for safety
                                                logger.warning(f"⚠️ Multiple @app name specifications for {func_def_node.name}. Using first one: {app_name_from_decorator}")
                                            else:
                                                app_name_from_decorator = decorator_node.args[0].value
                                        else:
                                            logger.warning(f"⚠️ @app decorator for {func_def_node.name} has unexpected positional arguments. Expected a single string.")

                                    if app_name_from_decorator is None:
                                        logger.warning(f"⚠️ @app decorator used on {func_def_node.name} but 'name' argument was not found or not a string.")
                                    # We don't add 'app' to decorator_names, as it's handled separately by app_name_from_decorator
                                else: # It's a call decorator but not 'app'
                                    decorator_names.append(decorator_func_name)
                            else: # Decorator call but func is not a simple Name (e.g. @obj.deco())
                                # Try to reconstruct its name, could be complex e.g. ast.Attribute
                                # For now, we'll log and skip complex decorator calls for simplicity
                                logger.debug(f"Skipping complex decorator call structure: {ast.dump(decorator_node.func)}")
                        # Skipping other decorator types for now (e.g. ast.Attribute)

                # Generate schema from arguments
                try:
                     schema_parts = self._ast_arguments_to_json_schema(func_def_node.args, docstring)
                     input_schema["properties"] = schema_parts.get("properties", {})
                     input_schema["required"] = schema_parts.get("required", [])
                except Exception as schema_e:
                     logger.warning(f"⚠️ Could not generate input schema for {func_name}: {schema_e}")
                     input_schema["description"] = f"Schema generation error: {schema_e}"

                function_info = {
                    "name": func_name,
                    "description": docstring or "(No description provided)", # Provide default
                    "inputSchema": input_schema,
                    "decorators": decorator_names, # Add extracted decorators here
                    "app_name": app_name_from_decorator # Add extracted app_name
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


    def _code_generate_stub(self, name: str) -> str:
        """
        Generates a string containing a basic Python function stub with the given name.
        """
        if not name or not isinstance(name, str):
            name = "unnamed_function" # Default name if invalid

        stub = f"""\
import atlantis

async def {name}():
    \"\"\"
    This is a placeholder function for '{name}'
    \"\"\"
    print(f"Executing placeholder function: {name}...")

    await atlantis.client_log("{name} running")

    # Replace this return statement with your function's result
    return f"Placeholder function '{name}' executed successfully."

"""
        logger.debug(f"⚙️ Generated code stub for function: {name}")
        return stub

    # Cache management
    async def invalidate_all_dynamic_module_cache(self):
        """Safely removes ALL dynamic function modules AND the parent package from sys.modules cache."""

        prefix_to_clear = f"{PARENT_PACKAGE_NAME}."
        async with self._dynamic_load_lock:
            # --- Invalidate importlib finder caches ---
            logger.debug("Calling importlib.invalidate_caches()")
            importlib.invalidate_caches()
            # --- End importlib cache invalidation ---

            modules_to_remove = [
                mod for mod in sys.modules
                if mod == PARENT_PACKAGE_NAME or mod.startswith(prefix_to_clear) # Include parent package
            ]
            if modules_to_remove:
                logger.info(f"Invalidating dynamic modules (and parent) from sys.modules cache: {modules_to_remove}")
                for mod_name in modules_to_remove:
                    logger.debug(f"  Attempting to pop: {mod_name}")
                    # Use pop with default None to avoid KeyError if concurrently removed
                    popped_module = sys.modules.pop(mod_name, None)
                    if popped_module:
                        logger.debug(f"    Successfully popped {mod_name}")
                    else:
                        logger.debug(f"    Module {mod_name} not found or already popped.")
                # Log remaining keys after attempted removal
                remaining_dynamic_keys = [k for k in sys.modules if k == PARENT_PACKAGE_NAME or k.startswith(prefix_to_clear)]
                logger.debug(f"  Remaining dynamic keys (incl parent) in sys.modules after pop: {remaining_dynamic_keys}")
            else:
                logger.debug("No dynamic modules (or parent) found in sys.modules to invalidate.")

    async def function_add(self, name: str, code: Optional[str] = None) -> bool:
        '''
        Creates a new function file.
        If code is provided, it saves it. Otherwise, generates and saves a stub.
        Returns True on success, False if the function already exists or on error.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            logger.error(f"Create failed: Invalid function name '{name}'")
            return False
        file_path = os.path.join(self.functions_dir, f"{secure_name}.py")

        if os.path.exists(file_path):
            logger.warning(f"Create failed: Function '{secure_name}' already exists.")
            return False

        try:
            code_to_save = code if code is not None else self._code_generate_stub(secure_name)
            if await self._fs_save_code(secure_name, code_to_save):
                logger.info(f"Function '{secure_name}' created successfully.")
                return True
            else:
                logger.error(f"Create failed: Could not save code for '{secure_name}'.")
                return False
        except Exception as e:
            logger.error(f"Error during function creation for '{secure_name}': {e}")
            logger.debug(traceback.format_exc())
            return False


    async def function_remove(self, name: str) -> bool:
        '''
        Removes a function file by moving it to the OLD subdirectory (relative to self.functions_dir).
        Returns True on success, False if the function doesn't exist or on error.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            logger.error(f"Remove failed: Invalid function name '{name}'")
            return False

        file_path = os.path.join(self.functions_dir, f"{secure_name}.py") # Use self.functions_dir
        # self.old_dir is already correctly initialized in __init__ based on self.functions_dir
        old_file_path = os.path.join(self.old_dir, f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{secure_name}.py")


        if not os.path.exists(file_path):
            logger.warning(f"⚠️ function_remove: File not found for '{secure_name}' at {file_path}")
            return False
        try:
            # Ensure the OLD directory exists (it should be created by __init__)
            os.makedirs(self.old_dir, exist_ok=True)

            shutil.move(file_path, old_file_path)
            logger.info(f"🗑️ Function '{secure_name}' removed. Moved from {file_path} to {old_file_path}")
            await self.invalidate_all_dynamic_module_cache() # Invalidate cache

            log_file_path = os.path.join(self.functions_dir, f"{secure_name}.log") # Use self.functions_dir for log
            if os.path.exists(log_file_path):
                try:
                    os.remove(log_file_path)
                    logger.debug(f"🗑️ Removed log file {log_file_path}")
                except OSError as e:
                    logger.warning(f"⚠️ Could not remove log file {log_file_path}: {e}")
            return True
        except Exception as e:
            logger.error(f"❌ function_remove: Failed to remove function '{secure_name}': {e}")
            logger.debug(traceback.format_exc())
            await self._write_error_log(secure_name, f"Failed to remove: {e}") # Use self._write_error_log
            return False

    async def _write_error_log(self, name: str, error_message: str) -> None: # Made it async to match caller, added self
        '''
        Write an error message to a function-specific log file in the dynamic_functions folder.
        Overwrites any existing log to only keep the latest error.
        Creates a log file named {name}.log with timestamp.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            logger.error("Cannot write error log: invalid function name provided.")
            return

        log_file_path = os.path.join(self.functions_dir, f"{secure_name}.log") # Use self.functions_dir
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_content = f"[{timestamp}] ERROR: {error_message}\n"

        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
            logger.debug(f"Wrote error log to {log_file_path}")
        except Exception as e:
            # Don't let logging errors disrupt the main flow
            logger.error(f"Failed to write error log for '{secure_name}': {e}")

    async def function_call(self, name: str, client_id: str, request_id: str, user: str = None, **kwargs) -> Any:
        '''
        Loads and executes a dynamic function by its name, passing kwargs.
        Flushes ALL dynamic function caches before loading to ensure freshness, protected by a lock.
        Ensures parent package exists in sys.modules.
        Returns the function's return value.
        Raises exceptions if the function doesn't exist, fails to load, or errors during execution.
        Gets the 'user' field that tells us who is making the call and passes it to the function context.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            raise ValueError(f"Invalid function name '{name}' for calling.")

        file_path = os.path.join(self.functions_dir, f"{secure_name}.py")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dynamic function '{secure_name}' not found at {file_path}")

        context_tokens = None
        module_name = f"{PARENT_PACKAGE_NAME}.{secure_name}"
        module = None # Define module outside the lock

        # --- Clear ALL dynamic function child modules from cache FIRST ---
        # This acquires the lock internally, clears, and releases.
        await self.invalidate_all_dynamic_module_cache()
        # --- End Cache Clear ---

        # --- Acquire lock *only* for parent check and specific module loading ---
        async with self._dynamic_load_lock:
            try:
                # --- Ensure Parent Package Exists in sys.modules ---
                if PARENT_PACKAGE_NAME not in sys.modules:
                    logger.info(f"Creating namespace package entry for '{PARENT_PACKAGE_NAME}' in sys.modules")
                    parent_module = importlib.util.module_from_spec(
                        importlib.util.spec_from_loader(PARENT_PACKAGE_NAME, loader=None, is_package=True)
                    )
                    parent_module.__path__ = [self.functions_dir]
                    sys.modules[PARENT_PACKAGE_NAME] = parent_module
                # --- End Parent Package Check ---

                # Clear any previous runtime error for this function before attempting load
                self._runtime_errors.pop(name, None)

                # --- Load the requested module fresh ---
                # Check sys.modules again *inside the lock* in case the watcher re-added it
                # between the invalidate call above and acquiring this lock.
                # Although invalidate_all should have removed it, this is belt-and-suspenders.
                if module_name in sys.modules:
                    logger.warning(f"Module {module_name} unexpectedly found in cache before load, removing again.")
                    del sys.modules[module_name]

                logger.info(f"Loading module fresh: {module_name}")
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create module spec for {secure_name}")
                try:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module # Add to sys.modules before exec


                    # Inject identity decorators for known decorator names
                    # This makes @chat, @public, etc., resolvable during module load
                    module.__dict__['chat'] = _mcp_identity_decorator
                    module.__dict__['public'] = _mcp_identity_decorator
                    # Add app decorator which takes parameters
                    module.__dict__['app'] = app
                    # Add hidden decorator
                    module.__dict__['hidden'] = hidden
                    # Add other known decorator names here if they arise

                    spec.loader.exec_module(module)
                except Exception as load_err:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    error_message = f"Error loading module '{module_name}': {load_err}"
                    logger.error(error_message)
                    logger.debug(traceback.format_exc())
                    self._runtime_errors[name] = str(load_err)
                    raise ImportError(error_message) from load_err
                # --- End Load ---

            except Exception as lock_section_err:
                logger.error(f"Unexpected error during locked module handling for {name}: {lock_section_err}")
                logger.debug(traceback.format_exc())
                self._runtime_errors[name] = str(lock_section_err)
                raise

        # --- Lock is released here ---

        # Check if module was loaded successfully inside the lock
        if module is None:
            # This indicates a failure during load that should have raised earlier
            raise RuntimeError(f"Module '{module_name}' failed to load successfully.")

        try:
            # --- Context Setting ---
            bound_client_log = functools.partial(utils.client_log, request_id=request_id, client_id_for_routing=client_id)
            logger.debug(f"Prepared bound_client_log for context. Request ID: {request_id}, Client ID: {client_id}")
            logger.debug(f"Setting context variables via atlantis. User: {user}")

            context_tokens = atlantis.set_context(
                client_log_func=bound_client_log,
                request_id=request_id,
                client_id=client_id,
                user=user,  # Pass the user who made the call - only works if atlantis.py has been updated
                entry_point_name=secure_name # Pass the entry point name
            )

            # --- Function Execution ---
            logger.info(f"Attempting to get function '{secure_name}' from loaded module.")
            function_to_call = getattr(module, secure_name, None)
            if not callable(function_to_call):
                raise ValueError(f"No callable function '{secure_name}' found in module. "
                              f"Please ensure the file contains a function matching its filename.")

            # Log whether we have user context available
            if user:
                logger.debug(f"Function '{secure_name}' will be called with user context: {user}")

            logger.info(f"Calling dynamic function '{secure_name}' with args: {kwargs.get('args', {})}")

            # Extract args from the kwargs dictionary
            function_args = kwargs.get('args', {})

            if inspect.iscoroutinefunction(function_to_call):
                result = await function_to_call(**function_args)
            else:
                result = function_to_call(**function_args)

            logger.info(f"Dynamic function '{secure_name}' executed successfully.")
            return result

        except Exception as exec_err:
            error_message = f"Error executing dynamic function '{name}': {str(exec_err)}"
            logger.error(error_message)
            logger.debug(traceback.format_exc())
            self._runtime_errors[name] = str(exec_err)
            raise

        finally:
            # --- RESET CONTEXT ---
            if context_tokens:
                logger.debug("Resetting context variables via atlantis")
                atlantis.reset_context(context_tokens)
            else:
                logger.debug("No context tokens found to reset.")


    # --- Function Management Functions --- #

    async def function_validate(self, name: str) -> Dict[str, Any]:
        '''
        Validates the syntax of a function file without executing it.
        Returns a dictionary {'valid': bool, 'error': Optional[str], 'function_info': Optional[Dict]}
        with detailed error messages and extracted function details on success.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            error_msg = f"Invalid function name '{name}'"
            await self._write_error_log(name, error_msg)
            return {'valid': False, 'error': error_msg, 'function_info': None}

        try:
            code = await self._fs_load_code(secure_name)
        except FileNotFoundError as e:
            error_msg = str(e)
            await self._write_error_log(name, error_msg)
            return {'valid': False, 'error': error_msg, 'function_info': None}

        # _code_validate_syntax now returns: (is_valid, error_message, function_info)
        is_valid, error_message, function_info = self._code_validate_syntax(code)

        if is_valid:
            # Successful validation
            logger.info(f"Syntax validation successful for function '{secure_name}'")

            # If there was a previous error log, remove it since the function is now valid
            try:
                log_path = os.path.join(self.functions_dir, f"{secure_name}.log")
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
            await self._write_error_log(secure_name, error_msg_full)

            # Return the detailed error message
            return {'valid': False, 'error': error_message, 'function_info': None}

    import inspect # Add import
    import atlantis

    async def function_set(self, args: Dict[str, Any], server: Any) -> Tuple[Optional[str], List[TextContent]]:
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
        metadata = self._code_extract_basic_metadata(code_buffer)
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
            file_path = os.path.join(self.functions_dir, f"{secure_name}.py")
            if os.path.exists(file_path):
                logger.info(f"💾 Found existing file for '{secure_name}', attempting backup...")
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    # Using .py.bak for clarity
                    backup_filename = f"{secure_name}_{timestamp}.py.bak"
                    # Make sure old_dir exists
                    os.makedirs(self.old_dir, exist_ok=True)
                    backup_path = os.path.join(self.old_dir, backup_filename)
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
        saved_path = await self._fs_save_code(extracted_function_name, code_buffer)

        if not saved_path:
            error_response = f"Error saving function '{extracted_function_name}' to file."
            logger.error(f"❌ function_set: {error_response}")
            # Return extracted name (as we got this far), but with error message
            return extracted_function_name, [TextContent(type="text", text=error_response)]

        logger.info(f"💾 Function '{extracted_function_name}' code saved successfully to {saved_path}")

        # Clear any cached runtime error for this function, as it's been updated
        self._runtime_errors.pop(extracted_function_name, None)

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

    # Function to get code for a dynamic function
    async def get_function_code(self, args, mcp_server) -> list[TextContent]:
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
        code = self._fs_load_code(name)
        if code is None:
            raise ValueError(f"Function '{name}' not found or could not be read")

        logger.info(f"📋 Retrieved code for function: {name}")

        # Return the code as text content
        return [TextContent(type="text", text=code)]

