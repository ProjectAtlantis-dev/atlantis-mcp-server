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
import shutil
import asyncio
import inspect
import logging
import importlib
import traceback
import datetime
import functools

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from werkzeug.utils import secure_filename

from state import logger, CYAN, RESET


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
FUNCTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_functions")
PARENT_PACKAGE_NAME = "dynamic_functions"

class DynamicFunctionManager:
    def __init__(self, functions_dir):
        # State that was previously global
        self.functions_dir = FUNCTIONS_DIR
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


    def _code_generate_stub(self, name: str) -> str:
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
        file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")

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

    async def function_call(self, name: str, client_id: str, request_id: str, **kwargs) -> Any:
        '''
        Loads and executes a dynamic function by its name, passing kwargs.
        Flushes ALL dynamic function caches before loading to ensure freshness, protected by a lock.
        Ensures parent package exists in sys.modules.
        Returns the function's return value.
        Raises exceptions if the function doesn't exist, fails to load, or errors during execution.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            raise ValueError(f"Invalid function name '{name}' for calling.")

        file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")

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
            logger.debug("Setting context variables via atlantis")
            context_tokens = atlantis.set_context(
                client_log_func=bound_client_log,
                request_id=request_id,
                client_id=client_id,
                entry_point_name=secure_name # Pass the entry point name
            )

            # --- Function Execution ---
            logger.info(f"Attempting to get function '{secure_name}' from loaded module.")
            function_to_call = getattr(module, secure_name, None)
            if not callable(function_to_call):
                raise AttributeError(f"Function '{secure_name}' not found or not callable within its module.")

            logger.info(f"Calling dynamic function '{secure_name}' with args: {kwargs}")
            if inspect.iscoroutinefunction(function_to_call):
                result = await function_to_call(**kwargs['args'])
            else:
                result = function_to_call(**kwargs['args'])

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

    def function_validate(self, name: str) -> Dict[str, Any]:
        '''
        Validates the syntax of a function file without executing it.
        Returns a dictionary {'valid': bool, 'error': Optional[str], 'function_info': Optional[Dict]}
        with detailed error messages and extracted function details on success.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            error_msg = f"Invalid function name '{name}'"
            self._write_error_log(name, error_msg)
            return {'valid': False, 'error': error_msg, 'function_info': None}

        code = self._fs_load_code(secure_name)
        if code is None:
            error_msg = f"Function '{secure_name}' not found or could not be read."
            self._write_error_log(name, error_msg)
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
            self._write_error_log(secure_name, error_msg_full)

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


# Self-test for DynamicFunctionManager
if __name__ == "__main__":
    import tempfile
    import shutil
    import asyncio
    import sys
    import time
    from pathlib import Path

    class MockServer:
        def __init__(self):
            self._cached_tools = {}
            self._last_functions_dir_mtime = None
            self._last_servers_dir_mtime = None

    class MockLogger:
        def debug(self, msg):
            print(f"[DEBUG] {msg}")
        def info(self, msg):
            print(f"[INFO] {msg}")
        def warning(self, msg):
            print(f"[WARNING] {msg}")
        def error(self, msg):
            print(f"[ERROR] {msg}")

    # Replace the actual logger with our mock for testing
    # Backup the original values
    original_logger = logger if 'logger' in globals() else None

    # Setup utilities for testing
    async def run_tests():
        try:
            print("\n🧪 STARTING DYNAMICFUNCTIONMANAGER SELF-TEST...")
            all_tests_passed = True
            total_tests = 0
            passed_tests = 0

            # Use the actual FUNCTIONS_DIR for testing
            temp_dir = FUNCTIONS_DIR
            print(f"\n📁 Using functions directory: {temp_dir}")

            # Set up the manager
            globals()['logger'] = MockLogger()
            manager = DynamicFunctionManager(temp_dir)

            # Test 1: _fs_save_code method
            total_tests += 1
            test_name = "test_function"
            test_code = "def test_function():\n    return 'Hello, world!'\n"

            print("\n🧪 TEST 1: Testing _fs_save_code method...")
            file_path = await manager._fs_save_code(test_name, test_code)
            saved_file = Path(temp_dir) / f"{test_name}.py"

            if file_path and saved_file.exists():
                print("✅ Test 1 PASSED: File was successfully saved")
                passed_tests += 1
            else:
                print("❌ Test 1 FAILED: File was not saved")
                all_tests_passed = False

            # Test 2: _fs_load_code method
            total_tests += 1
            print("\n🧪 TEST 2: Testing _fs_load_code method...")
            loaded_code = await manager._fs_load_code(test_name)

            if loaded_code == test_code:
                print("✅ Test 2 PASSED: Loaded code matches the saved code")
                passed_tests += 1
            else:
                print("❌ Test 2 FAILED: Loaded code doesn't match the saved code")
                print(f"Expected: {test_code}")
                print(f"Got: {loaded_code}")
                all_tests_passed = False

            # Test 3: _code_extract_basic_metadata method
            total_tests += 1
            print("\n🧪 TEST 3: Testing _code_extract_basic_metadata method...")
            test_code_with_docstring = """def test_function_with_docstring():
                \"\"\"
                This is a test function with a docstring.
                \"\"\"
                return 'Hello, docstring!'
            """

            metadata = manager._code_extract_basic_metadata(test_code_with_docstring)
            if metadata.get('name') == "test_function_with_docstring" and \
               metadata.get('description') and "test function with a docstring" in metadata.get('description'):
                print("✅ Test 3 PASSED: Metadata extraction works correctly")
                passed_tests += 1
            else:
                print("❌ Test 3 FAILED: Metadata extraction failed")
                print(f"Expected name: test_function_with_docstring, got: {metadata.get('name')}")
                print(f"Expected description to contain 'test function with a docstring', got: {metadata.get('description')}")
                all_tests_passed = False

            # Test 3.5: _fs_load_code and _code_extract_basic_metadata with externally edited file
            total_tests += 1
            print("\n🧪 TEST 3.5: Testing metadata extraction from externally edited file...")
            
            # 1. Create a test function file first
            external_edit_func_name = "external_edit_test_function"
            initial_code = """def external_edit_test_function():
    \"\"\"Initial docstring.\"\"\"
    return 'Initial version'
"""
            
            # Save the initial version using our manager
            await manager._fs_save_code(external_edit_func_name, initial_code)
            initial_file_path = Path(temp_dir) / f"{external_edit_func_name}.py"
            
            if not initial_file_path.exists():
                print("❌ Test 3.5 FAILED: Could not create initial test file")
                all_tests_passed = False
            else:
                # 2. Directly modify the file to simulate external editor
                updated_code = """def external_edit_test_function():
    \"\"\"This function was modified by an external editor.\"\"\"
    # Some comments added by the editor
    return 'Modified version'
"""
                try:
                    # Write directly to the file, bypassing our manager functions
                    with open(initial_file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_code)
                    
                    # 3. Load the code using fs_load_code and extract metadata
                    loaded_code = await manager._fs_load_code(external_edit_func_name)
                    if loaded_code != updated_code:
                        print("❌ Test 3.5 FAILED: Loaded code doesn't match the externally edited version")
                        print(f"Expected: {updated_code}")
                        print(f"Got: {loaded_code}")
                        all_tests_passed = False
                    else:
                        # Extract metadata from loaded code
                        ext_metadata = manager._code_extract_basic_metadata(loaded_code)
                        
                        if ext_metadata.get('name') == "external_edit_test_function" and \
                           ext_metadata.get('description') and "modified by an external editor" in ext_metadata.get('description'):
                            print("✅ Test 3.5 PASSED: Successfully extracted metadata from externally edited file")
                            passed_tests += 1
                        else:
                            print("❌ Test 3.5 FAILED: Could not extract correct metadata from externally edited file")
                            print(f"Expected name: external_edit_test_function, got: {ext_metadata.get('name')}")
                            print(f"Expected description to contain 'modified by an external editor', got: {ext_metadata.get('description')}")
                            all_tests_passed = False
                except Exception as e:
                    print(f"❌ Test 3.5 FAILED with exception: {e}")
                    all_tests_passed = False
                finally:
                    # Clean up test file
                    await manager.function_remove(external_edit_func_name)

            # Test 4: _code_validate_syntax method (valid code)
            total_tests += 1
            print("\n🧪 TEST 4: Testing _code_validate_syntax method with valid code...")
            is_valid, error_message, function_info = manager._code_validate_syntax(test_code_with_docstring)

            if is_valid and error_message is None and function_info and function_info.get('name') == "test_function_with_docstring":
                print("✅ Test 4 PASSED: Valid code validation works correctly")
                passed_tests += 1
            else:
                print("❌ Test 4 FAILED: Valid code validation failed")
                print(f"Expected is_valid: True, got: {is_valid}")
                print(f"Expected error_message: None, got: {error_message}")
                print(f"Expected function_info.name: test_function_with_docstring, got: {function_info.get('name') if function_info else None}")
                all_tests_passed = False

            # Test 5: _code_validate_syntax method (invalid code)
            total_tests += 1
            print("\n🧪 TEST 5: Testing _code_validate_syntax method with invalid code...")
            invalid_code = "def invalid_function(:\n    return 'This code has a syntax error'\n"

            try:
                is_valid, error_message, function_info = manager._code_validate_syntax(invalid_code)

                if not is_valid and error_message is not None and function_info is None:
                    print("✅ Test 5 PASSED: Invalid code validation works correctly")
                    passed_tests += 1
                else:
                    print("❌ Test 5 FAILED: Invalid code validation didn't catch the error")
                    print(f"Expected is_valid: False, got: {is_valid}")
                    print(f"Expected error_message: not None, got: {error_message}")
                    print(f"Expected function_info: None, got: {function_info}")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 5 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 6: function_add method
            total_tests += 1
            print("\n🧪 TEST 6: Testing function_add method...")
            new_function_name = "new_test_function"

            try:
                result = await manager.function_add(new_function_name)
                new_file = Path(temp_dir) / f"{new_function_name}.py"

                if result and new_file.exists():
                    print("✅ Test 6 PASSED: Function was successfully added")
                    passed_tests += 1
                else:
                    print("❌ Test 6 FAILED: Function was not added")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 6 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 7: function_remove method
            total_tests += 1
            print("\n🧪 TEST 7: Testing function_remove method...")
            try:
                result = await manager.function_remove(new_function_name)
                new_file = Path(temp_dir) / f"{new_function_name}.py"

                if result and not new_file.exists():
                    print("✅ Test 7 PASSED: Function was successfully removed")
                    passed_tests += 1
                else:
                    print("❌ Test 7 FAILED: Function was not removed")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 7 FAILED with exception: {e}")
                all_tests_passed = False

            # Test 8: function_set method
            total_tests += 1
            print("\n🧪 TEST 8: Testing function_set method...")
            try:
                mock_server = MockServer()
                test_code = "def set_test_function():\n    return 'Function set via function_set method'\n"
                args = {"code": test_code}

                name, response = await manager.function_set(args, mock_server)
                set_file = Path(temp_dir) / "set_test_function.py"

                if name == "set_test_function" and set_file.exists():
                    print("✅ Test 8 PASSED: Function was successfully set")
                    passed_tests += 1
                else:
                    print("❌ Test 8 FAILED: Function was not set")
                    print(f"Expected name: set_test_function, got: {name}")
                    print(f"File exists: {set_file.exists()}")
                    all_tests_passed = False
            except Exception as e:
                print(f"❌ Test 8 FAILED with exception: {e}")
                all_tests_passed = False

            # Print test summary
            print(f"\n🧪 DYNAMICFUNCTIONMANAGER SELF-TEST COMPLETE")
            print(f"Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")

            if all_tests_passed:
                print("\n✅ ALL TESTS PASSED! DynamicFunctionManager is working correctly.")
                return True
            else:
                print("\n❌ SOME TESTS FAILED. See above for details.")
                return False

        finally:
            # Restore the original logger
            if original_logger:
                globals()['logger'] = original_logger
            else:
                if 'logger' in globals(): # if mock was set but no original, remove it
                    del globals()['logger']
            
            # Clean up leftover test files
            print("\n🧹 Cleaning up test files...")
            leftover_files = [
                "test_function",            # From Test 1 & 2
                "set_test_function"         # From Test 8
            ]
            
            for test_file in leftover_files:
                try:
                    if os.path.exists(os.path.join(FUNCTIONS_DIR, f"{test_file}.py")):
                        await manager.function_remove(test_file)
                        print(f"  Removed {test_file}.py")
                except Exception as e:
                    print(f"  ⚠️ Could not remove {test_file}.py: {e}")
            
            print("\n🏁 DYNAMICFUNCTIONMANAGER SELF-TEST COMPLETE 🏁")
            print(f"Passed {passed_tests}/{total_tests} tests.")
            if all_tests_passed:
                print("🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
                # Clean up OLD directory if it exists and we're using FUNCTIONS_DIR
                old_dir_final_cleanup = Path(FUNCTIONS_DIR) / "OLD"
                if old_dir_final_cleanup.exists():
                    print(f"🧹 Cleaning up OLD directory: {old_dir_final_cleanup}")
                    shutil.rmtree(old_dir_final_cleanup) # remove OLD and its contents
            else:
                print("💔 SOME TESTS FAILED. Please review logs. 💔")
                # Advise on manual cleanup if using FUNCTIONS_DIR
                print(f"👉 NOTE: Tests were run against {temp_dir}. Review and manually clean up test files if necessary.")
            sys.exit(0 if all_tests_passed else 1)

    # Run the tests
    try:
        # Check if running in async environment
        try:
            asyncio.get_running_loop()
            print("Already in an event loop, creating a task")
            asyncio.create_task(run_tests())
        except RuntimeError:
            # No running event loop
            print("Creating new event loop for tests")
            asyncio.run(run_tests())
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
