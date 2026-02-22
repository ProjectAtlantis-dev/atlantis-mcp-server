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

# Visibility decorators that allow remote function calls
VISIBILITY_DECORATORS = ['visible', 'public', 'protected', 'tick', 'chat', 'text', 'session', 'index', 'price', 'location', 'app', 'copy']

# --- Identity Decorator Definition ---
def _mcp_identity_decorator(f):
    """A simple identity decorator that returns the function unchanged. Used as a placeholder for @chat, @public, etc."""
    return f

# --- Text Decorator Definition ---
def text(content_type: Optional[str] = None):
    """Decorator that marks a function as returning text content.
    Can be used bare (@text) or with an optional content type string (@text("markdown")).

    Usage: @text
           @text("markdown")
           @text("text/html")
           def my_text_function():
               ...
    """
    # Handle bare @text (no parentheses) - func is passed directly
    if callable(content_type):
        func = content_type
        setattr(func, '_text_content_type', None)
        return func
    # Handle @text("markdown") or @text() - returns a decorator
    def decorator(func):
        setattr(func, '_text_content_type', content_type)
        return func
    return decorator

# --- App Decorator Definition ---
def app(func):
    """
    Decorator that marks a function as an 'app' function.
    Grants visibility and owner-only access (for now - will use session-based logic later).

    Usage: @app
           def my_app_function():
               # This function will be visible and callable by owner
               ...
    """
    # Mark the function as visible by setting an attribute
    setattr(func, '_is_visible', True)
    setattr(func, '_is_app', True)
    return func

# --- Location Decorator Definition ---
def location(name: str):
    """Decorator to associate a dynamic function with a location name.
    Usage: @location(name="your_location_name")
    """
    def decorator(func_to_decorate):
        setattr(func_to_decorate, '_location_name', name)
        # functools.update_wrapper(decorator, func_to_decorate) # Not strictly needed for AST parsing but good practice
        return func_to_decorate
    return decorator

# Make the app and location decorators available for dynamic functions to import/use.
# This is a simplified way; a more robust way might involve adding it to a shared module
# that dynamic functions can import from, or injecting it into their global scope upon loading.
# For now, this definition here allows _code_validate_syntax to recognize them by name 'app' and 'location'.

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

# --- Visible Decorator Definition ---
def visible(func):
    """
    Decorator that marks a function as 'visible'.
    When applied, the function will be included in the tools list.
    This is required for all functions to be callable (opt-in visibility).

    Usage: @visible
           def my_visible_function():
               # This function will be visible in tools/list
               ...
    """
    # Mark the function as visible by setting an attribute
    setattr(func, '_is_visible', True)
    return func

# --- Tick Decorator Definition ---
def tick(func):
    """
    Decorator that marks a function as a 'tick' function.
    When applied, the function will be included in the tools list.
    Acts similar to @visible decorator.

    Usage: @tick
           def my_tick_function():
               # This function will be visible in tools/list
               ...
    """
    # Mark the function as visible by setting an attribute (same as @visible)
    setattr(func, '_is_visible', True)
    # Also mark it as a tick function for potential future use
    setattr(func, '_is_tick', True)
    return func

# --- Protected Decorator Definition ---
def protected(name: str):
    """
    Decorator that marks a function as 'protected' with a required protection name.
    Makes function visible in tools list with access control.

    Usage: @protected("protection_name")
           def my_protected_function():
               # This function will be visible in tools/list with protection
               ...

    Args:
        name: Required protection name (must be a legal Python function name).
              Will be validated later for proper naming conventions.
    """
    def decorator(func):
        # Mark the function as protected by setting an attribute
        setattr(func, '_is_protected', True)
        setattr(func, '_protection_name', name)
        return func

    return decorator

# --- Index Decorator Definition ---
def index(func):
    """
    Decorator that marks a function as eligible to be an app index.
    When applied, the function will be flagged as a potential app entry point.

    Usage: @index
           def my_index_function():
               # This function is eligible to be the app's index
               ...
    """
    # Mark the function as an index by setting an attribute
    setattr(func, '_is_index', True)
    return func

# --- Price Decorator Definition ---
def price(pricePerCall: float, pricePerSec: float):
    """
    Decorator that associates pricing information with a function.
    When applied, the function will have pricing metadata for billing purposes.

    Usage: @price(pricePerCall=0.01, pricePerSec=0.001)
           def my_paid_function():
               # This function has associated pricing
               ...

    Args:
        pricePerCall: Price charged per function call (float)
        pricePerSec: Price charged per second of execution (float)
    """
    def decorator(func):
        # Mark the function with pricing information
        setattr(func, '_has_price', True)
        setattr(func, '_price_per_call', pricePerCall)
        setattr(func, '_price_per_sec', pricePerSec)
        return func

    return decorator

# --- Copy Decorator Definition ---
def copy(func):
    """
    Decorator that marks a function as 'copyable' via _function_get.
    When applied, non-owners can retrieve the function's source code
    according to the function's visibility rules (@public, @protected, @visible).

    Usage: @copy
           @public
           async def my_copyable_function():
               # Non-owners can view source via _function_get
               ...

    Note: @copy only affects _function_get access control.
    - @copy + @public = Anyone can read the source code
    - @copy + @protected("func") = Custom authorization via protection function
    - @copy + @visible = Owner-only source code access (same as without @copy)
    """
    # Mark the function as copyable by setting an attribute
    setattr(func, '_is_copyable', True)
    return func

class DynamicFunctionManager:
    def __init__(self, functions_dir):
        # State that was previously global
        self.functions_dir = functions_dir
        self._runtime_errors = {}
        self._dynamic_functions_cache = {}
        self._dynamic_load_lock = asyncio.Lock()

        # NEW: Function-to-file mapping cache
        self._function_file_mapping = {}  # function_name -> filename mapping
        self._function_file_mapping_by_app = {}  # app_name -> {function_name -> filename mapping}
        self._function_metadata_by_app = {}  # app_name -> {function_name -> func_info dict (includes protection_name)}
        self._function_file_mapping_mtime = 0.0  # track when mapping was last built
        self._skipped_functions = []  # Track functions skipped (syntax errors, missing decorators, etc.)
        self._duplicate_functions = []  # Track duplicates: [(app_path, func_name, [file_paths])]

        # Per-function execution queues for automatic serialization
        # Key is fully qualified function name (e.g., "Home/kitty" or "App.Module/function")
        self._function_queues = {}  # function_name -> asyncio.Queue of (future, args_dict)
        self._function_queue_tasks = {}  # function_name -> asyncio.Task (queue processor)
        self._function_queue_lock = asyncio.Lock()  # Protects queue dict operations

        # Create directories if they don't exist
        os.makedirs(self.functions_dir, exist_ok=True)
        self.old_dir = os.path.join(self.functions_dir, "OLD")
        os.makedirs(self.old_dir, exist_ok=True)

    # Helper functions for app name <-> path conversion
    @staticmethod
    def _app_name_to_path(app_name: Optional[str]) -> Optional[str]:
        """
        Convert app name with dot notation to filesystem path.
        Example: "App.SubModule.Feature" -> "App/SubModule/Feature"
        Returns None if app_name is None.
        """
        if not app_name:
            return None
        return app_name.replace('.', os.sep)

    @staticmethod
    def _path_to_app_name(app_path: str) -> Optional[str]:
        """
        Convert app path (slash notation) to app name (dot notation) for output to MCP/user.
        Example: "App/SubModule/Feature" -> "App.SubModule.Feature"
        Returns None if app_path is None or empty.

        Note: Pass the app_path directly (directory path), not the full file path.
        If you have a file path like "App/Sub/func.py", use os.path.dirname() first.
        """
        if not app_path:
            return None
        # Convert path separators to dots
        return app_path.replace(os.sep, '.')

    # File operations
    async def _fs_add_code(self, name: str, code: str, app: Optional[str] = None) -> Optional[str]:
        """
        Adds a NEW function to a file (used by function_add).
        The function should NOT already exist (caller must check first).
        All functions go to main.py (either in app subdirectory or at root).
        If app is provided, uses app subdirectory's main.py.
        If app is None, uses root-level main.py.

        Args:
            name: Function name
            code: The function code to add
            app: App name in dot notation (e.g., "Examples.Markdown"), or None for root-level

        Returns:
            Full path if successful, None otherwise.
        """
        # Determine target directory based on app
        if app:
            app_path = self._app_name_to_path(app)
            assert app_path is not None  # app is truthy so path won't be None
            target_dir = os.path.join(self.functions_dir, app_path)
            location_display = f"{app_path}/main.py"
        else:
            target_dir = self.functions_dir
            location_display = "main.py"

        # Ensure directory exists
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, "main.py")

        # Simple rule: if main.py exists, append; otherwise create
        if os.path.exists(file_path):
            logger.debug(f"📝 Appending to existing {location_display}")
            mode = 'a'
        else:
            logger.debug(f"📝 Creating new {location_display}")
            mode = 'w'

        try:
            if mode == 'a':
                # When appending, ensure there's a newline separator
                # First check if file ends with newline
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()

                with open(file_path, 'a', encoding='utf-8') as f:
                    # Add newline separator if file doesn't end with one
                    if existing_content and not existing_content.endswith('\n'):
                        f.write('\n\n')
                    elif existing_content:
                        f.write('\n')  # Just one newline if already has one
                    f.write(code)
            else:
                # mode='w', just write
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)

            logger.debug(f"💾 Added function to {file_path}")
            return file_path
        except IOError as e:
            logger.error(f"❌ _fs_add_code: Failed to write file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ _fs_add_code: Unexpected error saving {file_path}: {e}")
            return None

    async def _fs_update_code(self, name: str, code: str, app: Optional[str] = None) -> Optional[str]:
        """
        Updates an existing function's file with complete code (used by function_set).
        Overwrites the entire file with the provided code.
        The code should contain ALL functions for that file, not just one.
        The function MUST already exist in the mapping.

        Args:
            name: Function name (used to find the file in the mapping)
            code: Complete file code
            app: App name in dot notation for app-specific lookup, None for root-level

        Returns:
            Full path if successful, None otherwise.
        """
        # Find the file containing this function - MUST exist
        existing_file = await self._find_file_containing_function(name, app)

        if not existing_file:
            error_msg = f"Cannot update function '{name}' - not found in mapping"
            if app:
                error_msg += f" for app '{app}'"
            logger.error(f"❌ _fs_update_code: {error_msg}")
            return None

        # Use the file from the mapping (could be main.py or user-created file)
        file_path = os.path.join(self.functions_dir, existing_file)
        logger.debug(f"📝 Updating entire file: {existing_file}")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.debug(f"💾 Updated file {file_path}")
            return file_path
        except IOError as e:
            logger.error(f"❌ _fs_update_code: Failed to write file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ _fs_update_code: Unexpected error saving {file_path}: {e}")
            return None

    async def _fs_load_code(self, name, app_name=None):
        """
        Loads code for a function by name using the function-to-file mapping.
        Supports subdirectories and app-specific targeting.
        Returns code string or raises FileNotFoundError if not found/error.
        """
        if not name or not isinstance(name, str):
            logger.error("❌ _fs_load_code: Invalid name provided.")
            raise ValueError(f"Invalid function name '{name}'")

        # Find which file contains this function
        target_file = await self._find_file_containing_function(name, app_name)
        if not target_file:
            if app_name:
                error_message = f"Function '{name}' does not exist in app '{app_name}'."
            else:
                error_message = f"Function '{name}' does not exist."
            logger.warning(f"⚠️ _fs_load_code: {error_message}")
            raise FileNotFoundError(error_message)

        # Load the code directly from the full file path
        file_path = os.path.join(self.functions_dir, target_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            logger.info(f"{CYAN}📋 === LOADING {name} from {target_file} ==={RESET}")
            logger.debug(f"💾 Loaded code for '{name}' from {file_path}")
            return code
        except (OSError, IOError) as e:
            error_message = f"Function '{name}' found at '{target_file}' but could not be read: {e}"
            logger.error(f"❌ _fs_load_code: {error_message}")
            raise FileNotFoundError(error_message) from e


    # Metadata extraction and validation
    def _code_extract_basic_metadata(self, code_buffer):
        """
        Extracts function name and description using basic regex from a code string buffer.
        Designed to be resilient to minor syntax errors. Returns {'name': ..., 'description': ...}.
        Values can be None if not found.
        """
        metadata: Dict[str, Optional[str]] = {'name': None, 'description': None}
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
        Validates syntax using ast.parse and extracts info about ALL function definitions found.

        Returns:
            tuple[bool, Optional[str], Optional[List[Dict[str, Any]]]]:
            - is_valid (bool): True if syntax is correct.
            - error_message (Optional[str]): Error details if invalid, None otherwise.
            - functions_info (Optional[List[Dict[str, Any]]]):
                List of dicts with 'name', 'description', 'inputSchema' for each function found,
                None if no functions found or invalid syntax.
        """
        if not code_buffer or not isinstance(code_buffer, str):
            return False, "Empty or invalid code buffer", None

        try:
            tree = ast.parse(code_buffer)
            #logger.debug("⚙️ Code validation successful (AST parse).")

            functions_info = []
            # Find ALL top-level function definitions
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_def_node = node
                    #logger.debug(f"⚙️ Found function definition: {func_def_node.name}")

                    func_name = func_def_node.name
                    docstring = ast.get_docstring(func_def_node)
                    input_schema = {"type": "object"} # Default empty schema

                    # Extract decorators, app_name, location_name, protection_name, is_index, is_copyable, text_content_type, and price
                    decorator_names = []
                    app_name_from_decorator = None # Initialize app_name
                    location_name_from_decorator = None # Initialize location_name
                    text_content_type_from_decorator = None # Initialize text_content_type
                    protection_name_from_decorator = None # Initialize protection_name
                    is_index_from_decorator = False # Initialize is_index
                    is_copyable_from_decorator = False # Initialize is_copyable
                    price_per_call_from_decorator = None # Initialize price_per_call
                    price_per_sec_from_decorator = None # Initialize price_per_sec
                    if func_def_node.decorator_list:
                        for decorator_node in func_def_node.decorator_list:
                            if isinstance(decorator_node, ast.Name): # e.g. @public, @hidden, @index, @copy
                                decorator_name = decorator_node.id
                                decorator_names.append(decorator_name)
                                # Check if it's the @index decorator
                                if decorator_name == 'index':
                                    is_index_from_decorator = True
                                # Check if it's the @copy decorator
                                if decorator_name == 'copy':
                                    is_copyable_from_decorator = True
                            elif isinstance(decorator_node, ast.Call): # e.g. @app(name="foo") or @app("foo"), @location(name="bar") or @location("bar")
                                if isinstance(decorator_node.func, ast.Name):
                                    decorator_func_name = decorator_node.func.id
                                    if decorator_func_name == 'app':
                                        # Extract 'name' argument from @app(name="...") or @app("...")
                                        if decorator_node.keywords: # Check keyword arguments like name="foo"
                                            for kw in decorator_node.keywords:
                                                if kw.arg == 'name' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                                    if app_name_from_decorator is not None:
                                                        logger.error(f"❌ Multiple @app name specifications for {func_def_node.name}. Using first one: {app_name_from_decorator}")
                                                    else:
                                                        app_name_from_decorator = kw.value.value
                                        # Positional arguments like @app("foo")
                                        if not app_name_from_decorator and decorator_node.args:
                                            if len(decorator_node.args) == 1 and isinstance(decorator_node.args[0], ast.Constant) and isinstance(decorator_node.args[0].value, str):
                                                if app_name_from_decorator is not None: # Should not happen if logic is correct, but for safety
                                                    logger.error(f"❌ Multiple @app name specifications for {func_def_node.name}. Using first one: {app_name_from_decorator}")
                                                else:
                                                    app_name_from_decorator = decorator_node.args[0].value
                                            else:
                                                logger.error(f"❌ @app decorator for {func_def_node.name} has unexpected positional arguments. Expected a single string.")

                                        if app_name_from_decorator is None:
                                            logger.error(f"❌ @app decorator used on {func_def_node.name} but 'name' argument was not found or not a string.")
                                        # We don't add 'app' to decorator_names, as it's handled separately by app_name_from_decorator
                                    elif decorator_func_name == 'location':
                                        # Extract 'name' argument from @location(name="...") or @location("...")
                                        if decorator_node.keywords: # Check keyword arguments like name="foo"
                                            for kw in decorator_node.keywords:
                                                if kw.arg == 'name' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                                    if location_name_from_decorator is not None:
                                                        logger.error(f"❌ Multiple @location name specifications for {func_def_node.name}. Using first one: {location_name_from_decorator}")
                                                    else:
                                                        location_name_from_decorator = kw.value.value
                                        # Positional arguments like @location("foo")
                                        if not location_name_from_decorator and decorator_node.args:
                                            if len(decorator_node.args) == 1 and isinstance(decorator_node.args[0], ast.Constant) and isinstance(decorator_node.args[0].value, str):
                                                if location_name_from_decorator is not None: # Should not happen if logic is correct, but for safety
                                                    logger.error(f"❌ Multiple @location name specifications for {func_def_node.name}. Using first one: {location_name_from_decorator}")
                                                else:
                                                    location_name_from_decorator = decorator_node.args[0].value
                                            else:
                                                logger.error(f"❌ @location decorator for {func_def_node.name} has unexpected positional arguments. Expected a single string.")

                                        if location_name_from_decorator is None:
                                            logger.error(f"❌ @location decorator used on {func_def_node.name} but 'name' argument was not found or not a string.")
                                        # We don't add 'location' to decorator_names, as it's handled separately by location_name_from_decorator
                                    elif decorator_func_name == 'protected':
                                        # Extract required 'name' argument from @protected(name="...") or @protected("...")
                                        if decorator_node.keywords: # Check keyword arguments like name="foo"
                                            for kw in decorator_node.keywords:
                                                if kw.arg == 'name' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                                    if protection_name_from_decorator is not None:
                                                        logger.error(f"❌ Multiple @protected name specifications for {func_def_node.name}. Using first one: {protection_name_from_decorator}")
                                                    else:
                                                        protection_name_from_decorator = kw.value.value
                                        # Positional arguments like @protected("foo")
                                        if not protection_name_from_decorator and decorator_node.args:
                                            if len(decorator_node.args) == 1 and isinstance(decorator_node.args[0], ast.Constant) and isinstance(decorator_node.args[0].value, str):
                                                if protection_name_from_decorator is not None: # Should not happen if logic is correct, but for safety
                                                    logger.error(f"❌ Multiple @protected name specifications for {func_def_node.name}. Using first one: {protection_name_from_decorator}")
                                                else:
                                                    protection_name_from_decorator = decorator_node.args[0].value
                                            else:
                                                logger.error(f"❌ @protected decorator for {func_def_node.name} has unexpected positional arguments. Expected a single string.")

                                        if protection_name_from_decorator is None:
                                            logger.error(f"❌ @protected decorator used on {func_def_node.name} but 'name' argument was not found or not a string.")
                                        # Add 'protected' to decorator_names so visibility check works
                                        decorator_names.append('protected')
                                    elif decorator_func_name == 'price':
                                        # Extract pricePerCall and pricePerSec arguments from @price(pricePerCall=..., pricePerSec=...)
                                        # or positional @price(0.01, 0.001)
                                        if decorator_node.keywords: # Check keyword arguments
                                            for kw in decorator_node.keywords:
                                                if kw.arg == 'pricePerCall' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                                                    price_per_call_from_decorator = float(kw.value.value)
                                                elif kw.arg == 'pricePerSec' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                                                    price_per_sec_from_decorator = float(kw.value.value)
                                        # Positional arguments like @price(0.01, 0.001)
                                        if decorator_node.args:
                                            if len(decorator_node.args) >= 1 and isinstance(decorator_node.args[0], ast.Constant) and isinstance(decorator_node.args[0].value, (int, float)):
                                                price_per_call_from_decorator = float(decorator_node.args[0].value)
                                            if len(decorator_node.args) >= 2 and isinstance(decorator_node.args[1], ast.Constant) and isinstance(decorator_node.args[1].value, (int, float)):
                                                price_per_sec_from_decorator = float(decorator_node.args[1].value)

                                        # Validate that both parameters were provided
                                        if price_per_call_from_decorator is None or price_per_sec_from_decorator is None:
                                            logger.error(f"❌ @price decorator used on {func_def_node.name} but both 'pricePerCall' and 'pricePerSec' arguments are required.")
                                        # Add 'price' to decorator_names
                                        decorator_names.append('price')
                                    elif decorator_func_name == 'text':
                                        # Extract optional content type from @text("markdown") or @text(content_type="markdown")
                                        if decorator_node.keywords:
                                            for kw in decorator_node.keywords:
                                                if kw.arg == 'content_type' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                                    text_content_type_from_decorator = kw.value.value
                                        # Positional argument like @text("markdown")
                                        if not text_content_type_from_decorator and decorator_node.args:
                                            if len(decorator_node.args) == 1 and isinstance(decorator_node.args[0], ast.Constant) and isinstance(decorator_node.args[0].value, str):
                                                text_content_type_from_decorator = decorator_node.args[0].value
                                            elif decorator_node.args:
                                                logger.error(f"❌ @text decorator for {func_def_node.name} has unexpected positional arguments. Expected an optional single string.")
                                        # @text() with no args is valid - just means plain text
                                        # Add 'text' to decorator_names
                                        decorator_names.append('text')
                                    else: # It's a call decorator but not 'app', 'location', 'protected', 'price', or 'text'
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
                         logger.error(f"❌ Could not generate input schema for {func_name}: {schema_e}")
                         input_schema["description"] = f"Schema generation error: {schema_e}"

                    function_info = {
                        "name": func_name,
                        "description": docstring or "(No description provided)", # Provide default
                        "inputSchema": input_schema,
                        "decorators": decorator_names, # Add extracted decorators here
                        "app_name": app_name_from_decorator, # Add extracted app_name
                        "location_name": location_name_from_decorator, # Add extracted location_name
                        "protection_name": protection_name_from_decorator, # Add extracted protection_name
                        "is_index": is_index_from_decorator, # Add extracted is_index flag
                        "is_copyable": is_copyable_from_decorator, # Add extracted is_copyable flag
                        "price_per_call": price_per_call_from_decorator, # Add extracted price_per_call
                        "price_per_sec": price_per_sec_from_decorator, # Add extracted price_per_sec
                        "text_content_type": text_content_type_from_decorator # Add extracted text content type (e.g. "markdown")
                    }
                    functions_info.append(function_info)

            if functions_info:
                #logger.debug(f"⚙️ Found {len(functions_info)} function(s) in file")
                return True, None, functions_info
            else:
                logger.error("❌ Syntax valid, but no top-level function definition found.")
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
            logger.error(f"❌ Code validation failed (AST parse): {error_msg}")
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Unexpected error during validation or AST processing: {str(e)}"
            logger.error(f"❌ {error_msg}\n{traceback.format_exc()}") # Log full traceback
            return False, error_msg, None


    def _code_generate_stub(self, name: str, location: Optional[str] = None) -> str:
        """
        Generates a string containing a basic Python function stub with the given name.
        """
        if not name or not isinstance(name, str):
            name = "unnamed_function" # Default name if invalid

        # Add location decorator if provided
        location_decorator = f"@location('{location}')\n" if location else ""

        stub = f"""\
import atlantis
import logging

logger = logging.getLogger("mcp_server")


{location_decorator}@visible
async def {name}():
    \"\"\"
    This is a placeholder function for '{name}'
    \"\"\"
    logger.info(f"Executing placeholder function: {name}...")

    await atlantis.client_log("{name} running")

    # Replace this return statement with your function's result
    return f"Placeholder function '{name}' executed successfully."

"""
        logger.debug(f"⚙️ Generated code stub for function: {name}")
        return stub

    # Cache management
    async def invalidate_function_mapping_cache(self):
        """Invalidate the function-to-file mapping cache."""
        self._function_file_mapping.clear()
        self._function_file_mapping_by_app.clear()
        self._function_metadata_by_app.clear()
        self._function_file_mapping_mtime = 0.0
        logger.debug("🧹 Function-to-file mapping cache invalidated")

    async def _build_function_file_mapping(self):
        """Build the function-to-file mapping by scanning all files recursively."""
        try:
            # Check if we need to rebuild the mapping
            current_mtime = os.path.getmtime(self.functions_dir)
            if (self._function_file_mapping and
                current_mtime == self._function_file_mapping_mtime):
                logger.debug("⚡ Using cached function-to-file mapping")
                return

            logger.info("🔍 Building function-to-file mapping...")
            self._function_file_mapping.clear()
            self._function_file_mapping_by_app.clear()
            self._function_metadata_by_app.clear()
            self._skipped_functions.clear()  # Clear skipped functions list to allow previously invalid functions to be retried
            self._duplicate_functions.clear()  # Clear duplicate tracking

            # Track ALL occurrences to detect duplicates at the end
            # Key: (app_path, func_name), Value: list of (rel_path, func_info)
            all_occurrences = {}

            # Scan all Python files in the functions directory and subdirectories
            ignore_dirs = ['OLD', '__pycache__']
            for root, dirs, files in os.walk(self.functions_dir, followlinks=True):
                # Skip ignored directories and any directories starting with dot
                dirs_to_remove = []
                for dir_name in dirs:
                    if dir_name in ignore_dirs or dir_name.startswith('.'):
                        dirs_to_remove.append(dir_name)
                        #logger.debug(f"🚫 Skipping directory: {os.path.join(root, dir_name)}")

                for dir_name in dirs_to_remove:
                    dirs.remove(dir_name)

                # Check if we're in a subdirectory and log it prominently
                if root != self.functions_dir:
                    subdir_name = os.path.basename(root)
                    #logger.info(f"🎯 EXPLORING SUBFOLDER: {CYAN}{subdir_name}{RESET}")

                for filename in files:
                    if not filename.endswith('.py'):
                        continue

                    file_path = os.path.join(root, filename)
                    # Calculate relative path from functions_dir
                    rel_path = os.path.relpath(file_path, self.functions_dir)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()

                        # Validate and extract function info
                        is_valid, error_message, functions_info = self._code_validate_syntax(code)

                        if not is_valid:
                            # Track files with syntax errors
                            logger.error(f"❌ SYNTAX ERROR in {rel_path}: {error_message}")
                            # Determine app_path from file path (slash notation)
                            track_app_path = os.path.dirname(rel_path) if '/' in rel_path else None

                            # Clean up error message - extract only the first line (before "Line content:")
                            clean_error = error_message.split('\n')[0] if error_message else 'unknown syntax error'

                            # Track this file with syntax error
                            self._skipped_functions.append({
                                'name': os.path.splitext(os.path.basename(rel_path))[0],  # Use filename as name
                                'app': track_app_path,  # Store slash path
                                'file': rel_path,
                                'reason': f'syntax error: {clean_error}',
                                'is_error': True,  # Explicit flag for error conditions
                                'decorators': []
                            })
                            continue

                        if is_valid and functions_info:
                            for func_info in functions_info:
                                func_name = func_info['name']

                                # NEW OPT-IN VISIBILITY: Check if function has a visibility decorator or is internal
                                decorators_from_info = func_info.get("decorators", [])
                                protection_name = func_info.get("protection_name")
                                is_internal = func_name.startswith('_function') or func_name.startswith('_server') or func_name.startswith('_admin')
                                is_visible = any(dec in decorators_from_info for dec in VISIBILITY_DECORATORS) if decorators_from_info else False

                                # Check if @protected has a valid protection name (required parameter)
                                has_invalid_protected = "protected" in decorators_from_info and not protection_name

                                # Skip if not visible and not internal OR if protected without valid name
                                if (not is_visible and not is_internal) or has_invalid_protected:
                                    if has_invalid_protected:
                                        skip_reason = "invalid @protected (missing required protection name)"
                                    else:
                                        skip_reason = "missing @visible decorator"

                                    # Use error level for invalid @protected, info level for others
                                    log_level = logger.error if has_invalid_protected else logger.info
                                    log_level(f"🙈 SKIPPING NON-VISIBLE FUNCTION: {CYAN}{func_name}{RESET} -> {rel_path} ({skip_reason})")
                                    # Determine app_path for tracking (slash notation)
                                    app_name_from_decorator = func_info.get('app_name')
                                    if app_name_from_decorator:
                                        # Convert decorator's dot notation to slash path immediately
                                        track_app_path = self._app_name_to_path(app_name_from_decorator)
                                    else:
                                        # Extract directory from file path
                                        track_app_path = os.path.dirname(rel_path) if '/' in rel_path else None
                                    # Track this skipped function
                                    self._skipped_functions.append({
                                        'name': func_name,
                                        'app': track_app_path,  # Store slash path
                                        'file': rel_path,
                                        'reason': skip_reason,
                                        'is_error': has_invalid_protected,  # True if error condition, False if intentional hiding
                                        'decorators': decorators_from_info
                                    })
                                    continue

                                # Determine app_path: prioritize @app() decorator, then file path (always slash notation)
                                app_name_from_decorator = func_info.get('app_name')
                                if app_name_from_decorator:
                                    # Convert decorator's dot notation to slash path immediately
                                    app_path = self._app_name_to_path(app_name_from_decorator)
                                else:
                                    # Extract directory from file path
                                    app_path = os.path.dirname(rel_path) if '/' in rel_path else None

                                # PHASE 1: Collect ALL occurrences for later duplicate detection
                                key = (app_path, func_name)
                                if key not in all_occurrences:
                                    all_occurrences[key] = []
                                all_occurrences[key].append((rel_path, func_info))
                        else:
                            if root != self.functions_dir:
                                logger.warning(f"⚠️ NO FUNCTIONS FOUND in {rel_path} (subfolder: {os.path.basename(root)})")
                            else:
                                logger.debug(f"  📍 No functions found in {rel_path}")

                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {rel_path} for function mapping: {e}")
                        continue

            # PHASE 2-5: Process all collected occurrences to detect duplicates and store non-duplicates
            duplicate_count = 0
            dropped_function_count = 0

            for (app_path, func_name), occurrences in all_occurrences.items():
                # PHASE 2: Detect duplicates (count > 1)
                if len(occurrences) > 1:
                    # PHASE 3: Report ALL occurrences of the duplicate
                    occurrence_list = "\n".join([f"   📂 Occurrence {i+1}: {path}" for i, (path, _) in enumerate(occurrences)])
                    logger.error(
                        f"❌ DUPLICATE FUNCTION DETECTED: '{func_name}' for app '{app_path}'\n"
                        f"{occurrence_list}\n"
                        f"   ⚠️  This is an error - same function cannot exist in multiple files for the same app!\n"
                        f"   ⚠️  REMOVING ALL {len(occurrences)} occurrences from tool list!"
                    )
                    duplicate_count += 1
                    dropped_function_count += len(occurrences)

                    # Store duplicate info for console report
                    file_paths = [path for path, _ in occurrences]
                    self._duplicate_functions.append((app_path, func_name, file_paths))

                    # Skip storing duplicates - they won't be in any mapping
                else:
                    # PHASE 4: Store non-duplicates in app-specific mapping and metadata
                    rel_path, func_info = occurrences[0]

                    # Store in app-specific mapping
                    if app_path not in self._function_file_mapping_by_app:
                        self._function_file_mapping_by_app[app_path] = {}
                    self._function_file_mapping_by_app[app_path][func_name] = rel_path

                    # Store metadata
                    if app_path not in self._function_metadata_by_app:
                        self._function_metadata_by_app[app_path] = {}
                    self._function_metadata_by_app[app_path][func_name] = func_info

                    # PHASE 5: Update main mapping for backward compatibility (prioritize top-level)
                    if func_name not in self._function_file_mapping or app_path is None:
                        self._function_file_mapping[func_name] = rel_path

            self._function_file_mapping_mtime = current_mtime

            # Final summary
            if duplicate_count > 0:
                logger.info(f"✅ Built function-to-file mapping with {len(self._function_file_mapping)} functions ({duplicate_count} duplicates found, {dropped_function_count} occurrences dropped)")
            else:
                logger.info(f"✅ Built function-to-file mapping with {len(self._function_file_mapping)} functions")

        except Exception as e:
            logger.error(f"❌ Error building function-to-file mapping: {e}")

    async def _find_file_containing_function(self, function_name: str, app_name: Optional[str] = None) -> Optional[str]:
        """
        Find which file contains the specified function.
        app_name can be in dot notation (e.g., "App.SubModule") and will be converted to slash path internally.
        When app_name is None, checks top-level functions (app_path=None) first.
        """
        await self._build_function_file_mapping()

        # Convert app_name to app_path (None stays None for top-level)
        app_path = self._app_name_to_path(app_name) if app_name else None

        # Always check app-specific mapping first
        app_mapping = self._function_file_mapping_by_app.get(app_path, {})
        if function_name in app_mapping:
            return app_mapping[function_name]

        # Only fall back to main mapping if no app was specified
        # When app_name is specified, we only check that specific app to allow
        # creating functions with the same name in different apps (e.g., index() in multiple apps)
        if app_name is None:
            return self._function_file_mapping.get(function_name)

        return None

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

        # NEW: Also invalidate function mapping cache
        await self.invalidate_function_mapping_cache()

    def _find_skipped_function(self, name: str, app: Optional[str] = None) -> Optional[dict]:
        '''
        Find a function in the skipped functions list (functions without visibility decorators).
        Returns the skipped function info dict if found, None otherwise.
        '''
        app_path = self._app_name_to_path(app) if app else None

        for skipped in self._skipped_functions:
            if skipped['name'] == name:
                # If app specified, must match
                if app_path is not None:
                    if skipped.get('app') == app_path:
                        return skipped
                else:
                    # No app specified - return first match
                    return skipped
        return None

    def _get_function_ast_info(self, name: str, file_path: str) -> tuple:
        '''
        Read a file, parse AST, and find a function by name.
        Returns (source, lines, func_node) tuple.
        Raises IOError for file issues, ValueError for parse/lookup issues.
        '''
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")

        # Find the function definition
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == name:
                    func_node = node
                    break

        if not func_node:
            raise ValueError(f"Function '{name}' not found in {file_path}")

        lines = source.splitlines(keepends=True)
        return source, lines, func_node

    async def _add_visible_decorator(self, name: str, file_path: str) -> bool:
        '''
        Add @visible decorator to an existing function that has no decorators.
        Returns True on success.
        '''
        source, lines, func_node = self._get_function_ast_info(name, file_path)

        # Insert @visible before the function definition
        insert_index = func_node.lineno - 1  # Convert to 0-indexed

        # Determine indentation from the function definition line
        func_line = lines[insert_index]
        indent = len(func_line) - len(func_line.lstrip())
        indent_str = func_line[:indent]

        # Insert @visible decorator with same indentation
        decorator_line = f"{indent_str}@visible\n"
        lines.insert(insert_index, decorator_line)

        new_source = ''.join(lines)

        # Write back
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_source)
            logger.info(f"Added @visible decorator to function '{name}' in {file_path}")
            return True
        except Exception as e:
            raise IOError(f"Failed to write file {file_path}: {e}")

    async def function_add(self, name: str, code: Optional[str] = None, app: Optional[str] = None, location: Optional[str] = None) -> bool:
        '''
        Creates a new function or re-enables a hidden one.
        If the function exists but has no decorators (was "removed"), adds @visible to re-enable it.
        If code is provided, it saves it. Otherwise, generates and saves a stub.
        If app is provided, creates the function in the app-specific subdirectory (supports dot notation).
        If location is provided, adds @location() decorator to the generated stub.
        Returns True on success, raises ValueError/IOError/RuntimeError on failure.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            error_msg = f"Invalid function name '{name}'"
            logger.error(f"Create failed: {error_msg}")
            raise ValueError(error_msg)

        # Check for reserved function name prefixes
        if name.startswith('_admin') or name.startswith('_function') or name.startswith('_server'):
            logger.error(f"Create failed: Function name '{name}' uses a reserved prefix")
            raise ValueError(f"Function names starting with '_admin', '_function', or '_server' are reserved for internal tools")

        # Check for reserved app names
        if app and app == "Internal":
            logger.error(f"Create failed: 'Internal' is a reserved app name")
            raise ValueError("'Internal' is a reserved app name and cannot be used for creating functions")

        # Ensure mapping is up to date
        await self._build_function_file_mapping()

        # Check if function already exists and is visible
        existing_file = await self._find_file_containing_function(secure_name, app)
        if existing_file:
            error_msg = f"Function '{secure_name}' already exists in {existing_file}"
            logger.warning(f"Create failed: {error_msg}")
            raise ValueError(error_msg)

        # Check if function exists but is hidden (no decorators)
        skipped = self._find_skipped_function(secure_name, app)
        if skipped:
            # Function exists but was hidden - re-enable it by adding @visible
            file_path = os.path.join(self.functions_dir, skipped['file'])
            logger.info(f"Function '{secure_name}' exists but is hidden, re-enabling with @visible")
            return await self._add_visible_decorator(secure_name, file_path)

        # Function doesn't exist anywhere - create new
        try:
            code_to_save = code if code is not None else self._code_generate_stub(secure_name, location)
            if await self._fs_add_code(secure_name, code_to_save, app):
                logger.info(f"Function '{secure_name}' created successfully.")
                return True
            else:
                error_msg = f"Could not save code for '{secure_name}'"
                logger.error(f"Create failed: {error_msg}")
                raise IOError(error_msg)
        except (ValueError, IOError):
            raise
        except Exception as e:
            error_msg = f"Error during function creation for '{secure_name}': {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise RuntimeError(error_msg) from e


    async def function_remove(self, name: str, app: Optional[str] = None) -> bool:
        '''
        "Removes" a function by stripping all its decorators, making it invisible to the tool system.
        The function code remains but won't be picked up as a tool.

        Args:
            name: Function name to remove
            app: Optional app name to disambiguate if multiple functions have the same name

        Returns:
            True on success, raises ValueError on failure.
        '''
        secure_name = utils.clean_filename(name)
        if not secure_name:
            raise ValueError(f"Invalid function name '{name}'")

        # Find the file containing this function
        function_file = await self._find_file_containing_function(secure_name, app)
        if not function_file:
            if app:
                raise ValueError(f"Function '{secure_name}' not found in app '{app}'")
            else:
                raise ValueError(f"Function '{secure_name}' not found")

        source, lines, func_node = self._get_function_ast_info(secure_name, function_file)

        # If no decorators, nothing to strip
        if not func_node.decorator_list:
            logger.info(f"Function '{secure_name}' has no decorators to strip")
            return True

        # Get line numbers of decorators to remove (AST line numbers are 1-indexed)
        decorator_lines = set(dec.lineno - 1 for dec in func_node.decorator_list)

        # Remove decorator lines (rebuild file without them)
        new_lines = [line for i, line in enumerate(lines) if i not in decorator_lines]
        new_source = ''.join(new_lines)

        # Write back
        try:
            with open(function_file, 'w', encoding='utf-8') as f:
                f.write(new_source)
            logger.info(f"Stripped {len(decorator_lines)} decorator(s) from function '{secure_name}' in {function_file}")
            return True
        except Exception as e:
            raise IOError(f"Failed to write file {function_file}: {e}")

    async def function_move(
        self,
        source_name: str,
        source_app: Optional[str],
        dest_app: str,
        dest_name: Optional[str] = None,
        dest_location: Optional[str] = None
    ) -> str:
        '''
        Moves a function from one location to another.
        Extracts the function code (with decorators) from source, adds to destination, removes from source.

        Args:
            source_name: Function name to move
            source_app: Optional source app name (if not specified, searches all apps)
            dest_app: Destination app name
            dest_name: Optional new name for the function (defaults to source_name)
            dest_location: Optional location decorator value to add/update

        Returns:
            Success message string

        Raises:
            ValueError: If function not found or destination already has it
            IOError: If file operations fail
        '''
        secure_source_name = utils.clean_filename(source_name)
        if not secure_source_name:
            raise ValueError(f"Invalid function name '{source_name}'")

        # Use source name if dest_name not provided
        final_name = dest_name if dest_name else secure_source_name
        secure_dest_name = utils.clean_filename(final_name)
        if not secure_dest_name:
            raise ValueError(f"Invalid destination function name '{final_name}'")

        # Find source file
        source_file = await self._find_file_containing_function(secure_source_name, source_app)
        if not source_file:
            if source_app:
                raise ValueError(f"Function '{secure_source_name}' not found in app '{source_app}'")
            else:
                raise ValueError(f"Function '{secure_source_name}' not found")

        source_file_path = os.path.join(self.functions_dir, source_file)

        # Check destination doesn't already have this function
        dest_file = await self._find_file_containing_function(secure_dest_name, dest_app)
        if dest_file:
            raise ValueError(f"Function '{secure_dest_name}' already exists in destination app '{dest_app}'")

        # Extract function code from source using AST
        source_code, lines, func_node = self._get_function_ast_info(secure_source_name, source_file_path)

        # Get the function's line range (including decorators)
        if func_node.decorator_list:
            start_line = func_node.decorator_list[0].lineno - 1  # 0-indexed
        else:
            start_line = func_node.lineno - 1

        end_line = func_node.end_lineno  # end_lineno is 1-indexed but we want exclusive

        # Extract function lines
        func_lines = lines[start_line:end_line]
        func_code = ''.join(func_lines)

        # If renaming, update the function name in the code
        if secure_dest_name != secure_source_name:
            # Replace function definition name
            func_code = re.sub(
                rf'(async\s+)?def\s+{re.escape(secure_source_name)}\s*\(',
                rf'\1def {secure_dest_name}(',
                func_code,
                count=1
            )

        # Handle @location decorator
        if dest_location:
            # Check if there's already a @location decorator
            location_pattern = r'@location\s*\([^)]*\)\s*\n'
            if re.search(location_pattern, func_code):
                # Update existing @location
                func_code = re.sub(location_pattern, f'@location("{dest_location}")\n', func_code)
            else:
                # Add @location decorator before the function def
                func_code = re.sub(
                    r'((?:async\s+)?def\s+)',
                    f'@location("{dest_location}")\n\\1',
                    func_code,
                    count=1
                )

        # Add to destination
        dest_file_path = await self._fs_add_code(secure_dest_name, func_code, dest_app)
        if not dest_file_path:
            raise IOError(f"Failed to add function to destination app '{dest_app}'")

        # Remove from source by deleting those lines
        new_lines = lines[:start_line] + lines[end_line:]

        # Clean up extra blank lines that might result
        new_source = ''.join(new_lines)
        # Remove excessive blank lines (more than 2 consecutive)
        new_source = re.sub(r'\n{3,}', '\n\n', new_source)

        # Write back source file (or delete if empty)
        new_source_stripped = new_source.strip()
        if not new_source_stripped:
            # File is empty, delete it
            os.remove(source_file_path)
            logger.info(f"Deleted empty source file {source_file_path}")
        else:
            with open(source_file_path, 'w', encoding='utf-8') as f:
                f.write(new_source)
            logger.info(f"Removed function '{secure_source_name}' from {source_file}")

        # Rebuild mapping
        await self._build_function_file_mapping()

        # Determine source display name
        if source_app:
            source_display = source_app
        else:
            # Extract from file path
            source_display = os.path.dirname(source_file) or "root"

        # Build result message
        if secure_dest_name != secure_source_name:
            return f"Function '{secure_source_name}' moved from '{source_display}' to '{dest_app}' as '{secure_dest_name}'"
        else:
            return f"Function '{secure_source_name}' moved from '{source_display}' to '{dest_app}'"

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

    async def _queue_processor(self, function_key: str):
        """
        Background task that processes queued function calls sequentially.
        Each function gets its own queue processor that runs one call at a time.

        Args:
            function_key: Unique identifier for the function (e.g., "Home/kitty")
        """
        logger.info(f"Queue processor started for function: {function_key}")

        try:
            while True:
                # Get next queued call (blocks until available)
                queue = self._function_queues.get(function_key)
                if not queue:
                    logger.warning(f"Queue for {function_key} disappeared, stopping processor")
                    break

                future, call_args = await queue.get()
                req_id = call_args.get('request_id', 'unknown')

                logger.info(f"Processing queued call for {function_key} (queue size: {queue.qsize()}) - request_id: {req_id}")

                try:
                    # Execute the actual function call
                    result = await self._execute_function(**call_args)
                    future.set_result(result)
                    logger.debug(f"Queue processor set result for {function_key} - request_id: {req_id}, type: {type(result)}")
                except Exception as e:
                    logger.error(f"Queue processor caught exception for {function_key}: {e}", exc_info=True)
                    future.set_exception(e)
                finally:
                    queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Queue processor for {function_key} cancelled")
            raise
        except Exception as e:
            logger.error(f"Queue processor for {function_key} failed: {e}", exc_info=True)
            # IMPORTANT: If we have a future that hasn't been resolved, resolve it with the exception
            # This prevents the caller from hanging forever
            if 'future' in locals() and not future.done():
                logger.error(f"Queue processor resolving pending future with exception for {function_key}")
                future.set_exception(e)

    async def _get_or_create_queue(self, function_key: str) -> asyncio.Queue:
        """
        Get or create a queue for a function, and ensure its processor task is running.

        Args:
            function_key: Unique identifier for the function

        Returns:
            asyncio.Queue for the function
        """
        async with self._function_queue_lock:
            if function_key not in self._function_queues:
                logger.info(f"Creating new queue for function: {function_key}")
                self._function_queues[function_key] = asyncio.Queue()
                # Start queue processor task
                task = asyncio.create_task(self._queue_processor(function_key))
                self._function_queue_tasks[function_key] = task

            return self._function_queues[function_key]

    async def function_call(self, name: str, client_id: str, request_id: str, user: Optional[str] = None, **kwargs) -> Any:
        """
        Public API for calling a dynamic function.
        Directly executes the function (bypassing queue for now).

        Returns the function's return value.
        """
        return await self._execute_function(name, client_id, request_id, user, **kwargs)

    async def function_call_queued(self, name: str, client_id: str, request_id: str, user: Optional[str] = None, **kwargs) -> Any:
        """
        Queued version of function_call. Automatically queues the call
        so that each function's invocations run sequentially (one at a time).

        Multiple concurrent calls to the same function will be queued and processed
        in order. Different functions have separate queues and run concurrently.

        Returns the function's return value.
        """
        # Build function key for queueing - use app+name to ensure proper isolation
        app_name = kwargs.get("app")
        if app_name:
            function_key = f"{app_name}/{name}"
        else:
            function_key = name

        logger.info(f"Queueing call to function: {function_key}")

        # Get or create the queue for this function
        queue = await self._get_or_create_queue(function_key)

        # Create a future to receive the result
        result_future = asyncio.Future()

        # Package all the call arguments
        call_args = {
            'name': name,
            'client_id': client_id,
            'request_id': request_id,
            'user': user,
            **kwargs
        }

        # Add to queue
        await queue.put((result_future, call_args))
        logger.info(f"Call queued for {function_key} (queue size: {queue.qsize()}) - request_id: {request_id}")

        # Wait for the result from the queue processor
        logger.debug(f"Waiting for result from queue processor for {function_key} - request_id: {request_id}")
        result = await result_future
        logger.debug(f"Received result from queue processor for {function_key} - request_id: {request_id}, type: {type(result)}")
        return result

    async def _execute_function(self, name: str, client_id: str, request_id: str, user: Optional[str] = None, **kwargs) -> Any:
        """
        Internal method that actually executes a dynamic function.
        This is called by the queue processor to run functions sequentially.

        Loads and executes a dynamic function by its name, passing kwargs.
        Flushes ALL dynamic function caches before loading to ensure freshness, protected by a lock.
        Ensures parent package exists in sys.modules.
        Returns the function's return value.
        Raises exceptions if the function doesn't exist, fails to load, or errors during execution.
        Gets the 'user' field that tells us who is making the call and passes it to the function context.
        """
        # Function name is now pre-parsed by server.py, so 'name' is the actual function name
        actual_function_name = name

        secure_name = utils.clean_filename(actual_function_name)
        if not secure_name:
            raise ValueError(f"Invalid function name '{actual_function_name}' for calling.")

        # Extract app name from kwargs (already set by server.py parsing logic)
        app_name = kwargs.get("app")

        # Special handling for click and upload callback functions
        if actual_function_name.startswith("_click_callback_") or actual_function_name.startswith("_upload_callback_"):
            # This is a temporary click callback - handle it directly from atlantis
            if hasattr(atlantis, actual_function_name):
                callback_func = getattr(atlantis, actual_function_name)

                # Set up atlantis context for the callback
                context_tokens = atlantis.set_context(
                    client_log_func=lambda message, level="INFO", message_type="text": utils.client_log(
                        client_id_for_routing=client_id,
                        request_id=request_id,
                        entry_point_name=actual_function_name,
                        message_type=message_type,
                        message=message,
                        level=level
                    ),
                    request_id=request_id,
                    client_id=client_id,
                    entry_point_name=actual_function_name,
                    user=user,
                    session_id=kwargs.get("session_id"),
                    command_seq=kwargs.get("command_seq"),
                    shell_path=kwargs.get("shell_path")
                )

                try:
                    # Execute the callback with proper atlantis context
                    if actual_function_name.startswith("_upload_callback_"):
                        # Upload callbacks get arguments
                        upload_args = kwargs.get("args", {})
                        filename = upload_args.get("filename")
                        filetype = upload_args.get("filetype")
                        base64Content = upload_args.get("base64Content")

                        if inspect.iscoroutinefunction(callback_func):
                            result = await callback_func(filename, filetype, base64Content)
                        else:
                            result = callback_func(filename, filetype, base64Content)
                    else:
                        # Click callbacks get no arguments
                        if inspect.iscoroutinefunction(callback_func):
                            result = await callback_func()
                        else:
                            result = callback_func()

                    # Click callbacks should not return anything to the MCP client
                    return None

                finally:
                    # Reset atlantis context
                    if context_tokens:
                        atlantis.reset_context(context_tokens)
            else:
                raise FileNotFoundError(f"Click callback function '{actual_function_name}' not found in atlantis")

        # Normal file-based function handling
        target_file = await self._find_file_containing_function(actual_function_name, app_name)
        if not target_file:
            raise FileNotFoundError(f"Dynamic function '{name}' not found in any file")

        file_path = os.path.join(self.functions_dir, target_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dynamic function '{name}' not found at {file_path}")

        context_tokens = None
        # Use the relative path (without .py) for module name, replacing slashes with dots
        module_path = os.path.splitext(target_file)[0].replace(os.sep, '.')
        module_name = f"{PARENT_PACKAGE_NAME}.{module_path}"
        module = None # Define module outside the lock

        # Cache invalidation is now handled by the file watcher to avoid unnecessary rebuilds
        # on every function call. Modules will be reloaded when files actually change.

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

                # --- Load the requested module (using cache if available) ---
                # If the file watcher invalidated the module, it won't be in sys.modules
                # If it IS in sys.modules, the file hasn't changed so we can use the cached version
                if module_name in sys.modules:
                    logger.debug(f"Module {module_name} found in cache, using cached version.")
                    module = sys.modules[module_name]
                else:
                    logger.info(f"Loading module fresh: {module_name}")
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Could not create module spec for {target_file}")
                    try:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module # Add to sys.modules before exec


                        # Inject identity decorators for known decorator names
                        # This makes @chat, @public, @session, etc., resolvable during module load
                        module.__dict__['chat'] = _mcp_identity_decorator
                        module.__dict__['text'] = text
                        module.__dict__['public'] = _mcp_identity_decorator
                        module.__dict__['session'] = _mcp_identity_decorator
                        module.__dict__['game'] = _mcp_identity_decorator
                        # Add app decorator which takes parameters
                        module.__dict__['app'] = app
                        # Add location decorator which takes parameters
                        module.__dict__['location'] = location
                        # Add visible decorator
                        module.__dict__['visible'] = visible
                        # Add tick decorator
                        module.__dict__['tick'] = tick
                        # Add protected decorator
                        module.__dict__['protected'] = protected
                        # Add index decorator
                        module.__dict__['index'] = index
                        # Add price decorator
                        module.__dict__['price'] = price
                        # Add copy decorator
                        module.__dict__['copy'] = copy
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

        # --- Protection Check ---
        # Retrieve function metadata from cache to check if function is protected
        await self._build_function_file_mapping()  # Ensure cache is built
        # Convert app_name (dot notation) to app_path (slash notation) for internal lookup
        app_path = self._app_name_to_path(app_name) if app_name else None
        func_metadata = self._function_metadata_by_app.get(app_path, {}).get(actual_function_name)

        if func_metadata:
            protection_name = func_metadata.get('protection_name')
            if protection_name:
                # Prevent infinite recursion: protection functions cannot be protected
                if protection_name == actual_function_name:
                    error_msg = f"Protection function '{protection_name}' cannot protect itself"
                    logger.error(f"❌ {error_msg}")
                    raise PermissionError(error_msg)

                logger.info(f"🔒 Function '{actual_function_name}' is protected by '{protection_name}'")

                # Call the protection function (which must be a top-level function, not in any app)
                # The protection function takes the user as a parameter and returns True/False
                try:
                    logger.info(f"🔐 Calling protection function '{protection_name}' for user '{user}'")

                    # Call protection function using the same function_call mechanism
                    # Protection functions must be top-level (app_name=None)
                    is_allowed = await self.function_call(
                        name=protection_name,
                        client_id=client_id,
                        request_id=request_id,
                        user=user,
                        app=None,  # Protection functions must be top-level
                        args={'user': user}  # Pass user as argument to protection function
                    )

                    # Check the result
                    if not is_allowed:
                        error_msg = f"Access denied: User '{user}' call to '{actual_function_name}' is not authorized by '{protection_name}'"
                        logger.warning(f"🚫 {error_msg}")
                        raise PermissionError(error_msg)

                    logger.info(f"✅ Access granted for user '{user}' to function '{actual_function_name}'")

                except PermissionError:
                    # Re-raise permission errors
                    raise
                except Exception as prot_err:
                    error_msg = f"Error executing protection function '{protection_name}': {prot_err}"
                    logger.error(f"❌ {error_msg}")
                    logger.debug(traceback.format_exc())
                    raise PermissionError(error_msg) from prot_err
        # --- End Protection Check ---

        try:
            # --- Context Setting ---
            bound_client_log = functools.partial(utils.client_log, request_id=request_id, client_id_for_routing=client_id)
            logger.debug(f"Prepared bound_client_log for context. Request ID: {request_id}, Client ID: {client_id}")
            logger.debug(f"Setting context variables via atlantis. User: {user}")

            # Extract session_id from kwargs if present
            session_id = kwargs.get('session_id', None)
            # Extract command_seq from kwargs if present
            command_seq = kwargs.get('command_seq', None)

            # Extract shell_path from kwargs if present
            shell_path = kwargs.get('shell_path', None)

            context_tokens = atlantis.set_context(
                client_log_func=bound_client_log,
                request_id=request_id,
                client_id=client_id,
                user=user,  # Pass the user who made the call - only works if atlantis.py has been updated
                session_id=session_id,  # Pass the session_id
                command_seq=command_seq,  # Pass the command_seq
                shell_path=shell_path,  # Pass the shell_path
                entry_point_name=actual_function_name # Pass the actual function name (not filename)
            )

            # --- Function Execution ---
            logger.info(f"Attempting to get function '{actual_function_name}' from loaded module.")
            function_to_call = getattr(module, actual_function_name, None)
            if not callable(function_to_call):
                raise ValueError(f"No callable function '{actual_function_name}' found in module '{target_file}'. "
                              f"Please ensure the file contains a function with this name.")

            # Log whether we have user context available
            if user:
                logger.debug(f"Function '{actual_function_name}' will be called with user context: {user}")

            # Extract args from the kwargs dictionary
            function_args = kwargs.get('args', {})
            logger.info(f"Calling dynamic function '{actual_function_name}' with args: {function_args}")
            logger.info(f"📊 Args as JSON: {utils.format_json_log(function_args)}")

            if inspect.iscoroutinefunction(function_to_call):
                result = await function_to_call(**function_args)
            else:
                result = function_to_call(**function_args)

            logger.info(f"Dynamic function '{actual_function_name}' executed successfully.")
            return result

        except Exception as exec_err:
            # Error already enhanced and logged at source, just store and re-raise
            self._runtime_errors[actual_function_name] = str(exec_err)
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
        Returns a dictionary {'valid': bool, 'error': Optional[str], 'function_info': Optional[List[Dict]]}
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

        # _code_validate_syntax now returns: (is_valid, error_message, functions_info)
        is_valid, error_message, functions_info = self._code_validate_syntax(code)

        if is_valid:
            # Successful validation
            logger.info(f"Syntax validation successful for function file '{secure_name}'")

            # If there was a previous error log, remove it since the function is now valid
            try:
                log_path = os.path.join(self.functions_dir, f"{secure_name}.log")
                if os.path.exists(log_path):
                    os.remove(log_path)
                    logger.debug(f"Removed error log for '{secure_name}' as validation now passes")
            except Exception as e:
                logger.debug(f"Failed to remove old error log for '{secure_name}': {e}")

            # Return success and the extracted function info (now a list)
            return {'valid': True, 'error': None, 'function_info': functions_info}
        else:
            # Failed validation - write to the error log
            error_msg_full = f"Syntax validation failed: {error_message}"
            logger.warning(f"{error_msg_full} Function file: '{secure_name}'")
            await self._write_error_log(secure_name, error_msg_full)

            # Return the detailed error message
            return {'valid': False, 'error': error_message, 'function_info': None}

    async def function_set(self, args: Dict[str, Any], server: Any) -> Tuple[Optional[str], Any]:
        """
        Handles the _function_set tool call.
        Updates an EXISTING function's file with complete code.
        The function must already exist in the mapping.
        Extracts all function names using AST parsing, uses the first one to find the file.
        Supports optional app parameter for app-specific function targeting.
        Returns the function name (if successful) and a status message (string or dict).
        Does *not* perform full syntax validation before saving.
        """
        logger.info("⚙️ Handling _function_set call (using AST parsing for all functions)")
        code_buffer = args.get("code")
        app_name = args.get("app")  # Optional app name for disambiguation

        if not code_buffer or not isinstance(code_buffer, str):
            logger.warning("⚠️ function_set: Missing or invalid 'code' parameter.")
            # Return None for name, and the error message (plain string, MCP formatting in _format_mcp_response)
            return None, "Error: Missing or invalid 'code' parameter."

        # 1. Extract ALL function names using AST parsing
        is_valid, error_message, functions_info = self._code_validate_syntax(code_buffer)

        if not is_valid:
            error_response = f"Error: Could not parse function code: {error_message}"
            logger.warning(f"⚠️ function_set: Failed to parse code via AST.")
            return None, error_response

        if not functions_info:
            error_response = "Error: Could not extract any function names from the provided code. Ensure it contains at least one function definition."
            logger.warning(f"⚠️ function_set: No functions found in code.")
            return None, error_response

        # Extract function names
        function_names = [func_info['name'] for func_info in functions_info]
        logger.info(f"⚙️ Extracted {len(function_names)} function(s) via AST: {', '.join(function_names)}")

        # 2. Check if at least one function exists in the mapping (required for update)
        # Try each function name to find one that exists in the mapping
        # (The buffer may contain new functions not yet in the mapping)
        matched_func_name = None
        existing_file = None
        for func_name in function_names:
            existing_file = await self._find_file_containing_function(func_name, app_name)
            if existing_file:
                matched_func_name = func_name
                logger.info(f"⚙️ Matched function '{func_name}' to file: {existing_file}")
                break

        if not existing_file or not matched_func_name:
            error_response = f"Cannot update functions - none of the functions ({', '.join(function_names)}) found in mapping. Use function_add to create new functions."
            logger.error(f"❌ function_set: {error_response}")
            return None, error_response

        logger.info(f"⚙️ Updating existing file: {existing_file}")

        # 3. Save the code using _fs_update_code (overwrites entire file with complete code)
        saved_path = await self._fs_update_code(matched_func_name, code_buffer, app_name)

        if not saved_path:
            error_response = f"Error saving functions to file '{existing_file}'."
            logger.error(f"❌ function_set: {error_response}")
            # Return matched function name, but with error message
            return matched_func_name, error_response

        logger.info(f"💾 Functions saved successfully to {saved_path}")

        # Clear any cached runtime errors for all functions, as they've been updated
        for func_name in function_names:
            self._runtime_errors.pop(func_name, None)

        # 4. Attempt AST parsing for immediate feedback (but save regardless)
        syntax_error = None
        try:
            ast.parse(code_buffer)
            logger.info(f"✅ Basic syntax validation (AST parse) successful for '{existing_file}'.")
        except SyntaxError as e:
            syntax_error = str(e)
            logger.warning(f"⚠️ Basic syntax validation (AST parse) failed for '{existing_file}': {syntax_error}")

        # 5. Clear cache (server needs to reload tools)
        logger.info(f"🧹 Clearing tool cache timestamp on server due to function_set for '{existing_file}'.")
        server._last_functions_dir_mtime = None # Reset mtime to force reload
        server._last_servers_dir_mtime = None # Reset mtime to force reload

        # 6. Prepare success message, including validation status
        #save_status = f"Functions saved to '{filename_to_use}.py': {', '.join(function_names)}"
        #save_status = f"Function saved"
        # Calculate relative path for display
        rel_path = os.path.relpath(saved_path, self.functions_dir)

        if syntax_error:
            # If validation failed, return dict with message and validation info
            response_message = f"Function saved to {rel_path}"
            logger.warning(f"⚠️ {response_message}")
            return matched_func_name, {
                "message": response_message,
                "validationStatus": "ERROR",
                "validationMessage": syntax_error
            }
        else:
            # If validation succeeded, return plain string
            response_message = f"Function saved to {rel_path}"
            logger.info(f"✅ {response_message}")
            return matched_func_name, response_message

    # Function to get code for a dynamic function
    async def get_function_code(self, args, mcp_server) -> str:
        """
        Get the source code for a dynamic function by name using function-to-file mapping.
        Supports optional app parameter for app-specific function targeting.
        Returns the code as a plain string (MCP formatting happens in _format_mcp_response).
        """
        # Get function name and optional app name
        name = args.get("name")
        app_name = args.get("app")

        # Validate parameters
        if not name:
            raise ValueError("Missing required parameter: name")

        # Load the code using the centralized _fs_load_code method
        try:
            code = await self._fs_load_code(name, app_name)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e

        if not code:
            raise ValueError(f"Function '{name}' file is empty")

        # Return the code as plain string
        return code

