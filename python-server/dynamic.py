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

def _code_validate_syntax(code_buffer: str) -> tuple[bool, Optional[str]]:
    """
    Validates if the provided code buffer is syntactically correct Python using ast.parse.
    Returns a tuple of (is_valid: bool, error_message: Optional[str]).
    """
    if not code_buffer or not isinstance(code_buffer, str):
        return False, "Empty or invalid code buffer"

    try:
        ast.parse(code_buffer)
        logger.debug("⚙️ Code validation successful (AST parse).")
        # Note: This only checks syntax. It doesn't check for runtime errors,
        # correct imports, or if the expected function actually exists or runs.
        # Full runtime validation typically requires execution context or temporary files.
        return True, None
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
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during validation: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return False, error_msg

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
