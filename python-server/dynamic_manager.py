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
from typing import Optional, Any, Dict, Union
from werkzeug.utils import secure_filename
from state import logger, FUNCTIONS_DIR # Import FUNCTIONS_DIR
from dynamic import _fs_save_code, _fs_load_code, _code_validate_syntax, _code_generate_stub # Import the utility functions

# Ensure FUNCTIONS_DIR exists at startup (or wherever appropriate)
os.makedirs(FUNCTIONS_DIR, exist_ok=True)
OLD_DIR = os.path.join(FUNCTIONS_DIR, "OLD")
os.makedirs(OLD_DIR, exist_ok=True)

def function_create(name: str, code: Optional[str] = None) -> bool:
    '''
    Creates a new function file.
    If code is provided, it saves it. Otherwise, generates and saves a stub.
    Returns True on success, False if the function already exists or on error.
    '''
    secure_name = secure_filename(name)
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


def function_update(name: str, code: str) -> bool:
    '''
    Updates an existing function file with new code.
    Moves the old version to the OLD subdirectory before saving.
    Returns True on success, False if the function doesn't exist or on error.
    '''
    secure_name = secure_filename(name)
    if not secure_name:
        logger.error(f"Update failed: Invalid function name '{name}'")
        return False
    file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")

    if not os.path.exists(file_path):
        logger.warning(f"Update failed: Function '{secure_name}' does not exist.")
        return False

    try:
        # Backup existing file
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        backup_path = os.path.join(OLD_DIR, f"{secure_name}_{timestamp}.py.old")
        shutil.move(file_path, backup_path) # Use shutil.move for robustness
        logger.info(f"Backed up existing function '{secure_name}' to '{backup_path}'")

        # Save new code
        if _fs_save_code(secure_name, code):
            logger.info(f"Function '{secure_name}' updated successfully.")
            return True
        else:
            # Attempt to restore backup if save fails?
            logger.error(f"Update failed: Could not save new code for '{secure_name}'. Backup retained.")
            # Consider trying to move backup_path back to file_path here, though it might also fail.
            return False
    except Exception as e:
        logger.error(f"Error during function update for '{secure_name}': {e}")
        logger.debug(traceback.format_exc())
        # Attempt to restore backup if an unexpected error occurred during the process
        try:
            if 'backup_path' in locals() and os.path.exists(backup_path) and not os.path.exists(file_path):
                 shutil.move(backup_path, file_path)
                 logger.info(f"Restored backup for '{secure_name}' due to update error.")
        except Exception as restore_e:
            logger.error(f"Failed to restore backup for '{secure_name}' after update error: {restore_e}")
        return False

def function_remove(name: str) -> bool:
    '''
    Removes a function file by moving it to the OLD subdirectory.
    Returns True on success, False if the function doesn't exist or on error.
    '''
    secure_name = secure_filename(name)
    if not secure_name:
        logger.error(f"Remove failed: Invalid function name '{name}'")
        return False
    file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")

    if not os.path.exists(file_path):
        logger.warning(f"Remove failed: Function '{secure_name}' does not exist.")
        return False

    try:
        # Move to OLD directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        backup_path = os.path.join(OLD_DIR, f"{secure_name}_{timestamp}.py.old")
        shutil.move(file_path, backup_path)
        logger.info(f"Function '{secure_name}' removed (moved to '{backup_path}')")
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
        secure_name = secure_filename(name)
        log_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Open in write mode to overwrite previous content
        with open(log_path, 'w') as log_file:
            log_file.write(f"{timestamp} [ERROR] {error_message}\n")
            
        logger.debug(f"Wrote error log for '{secure_name}' at {log_path}")
    except Exception as e:
        # Don't let logging errors disrupt the main flow
        logger.error(f"Failed to write error log for '{name}': {e}")

def function_validate(name: str) -> Dict[str, Union[bool, Optional[str]]]:
    '''
    Validates the syntax of a function file without executing it.
    Returns a dictionary {'valid': bool, 'error': Optional[str]} with detailed error messages.
    '''
    secure_name = secure_filename(name)
    if not secure_name:
        error_msg = f"Invalid function name '{name}'"
        _write_error_log(name, error_msg)
        return {'valid': False, 'error': error_msg}

    code = _fs_load_code(secure_name)
    if code is None:
        error_msg = f"Function '{secure_name}' not found or could not be read."
        _write_error_log(name, error_msg)
        return {'valid': False, 'error': error_msg}

    # Now _code_validate_syntax returns a tuple of (is_valid, error_message)
    is_valid, error_message = _code_validate_syntax(code)
    
    if is_valid:
        # Successful validation - log to server log but don't create function log
        logger.info(f"Syntax validation successful for function '{secure_name}'")
        
        # If there was a previous error log, remove it since the function is now valid
        try:
            log_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.log")
            if os.path.exists(log_path):
                os.remove(log_path)
                logger.debug(f"Removed error log for '{secure_name}' as validation now passes")
        except Exception as e:
            logger.debug(f"Failed to remove old error log for '{secure_name}': {e}")
            
        return {'valid': True, 'error': None}
    else:
        # Failed validation - write to the error log
        error_msg = f"Syntax validation failed: {error_message}"
        logger.warning(f"{error_msg} Function: '{secure_name}'")
        _write_error_log(secure_name, error_msg)
        
        # Return the detailed error message
        return {'valid': False, 'error': error_message}

def function_call(name: str, **kwargs) -> Any:
    '''
    Loads and executes a dynamic function by its name, passing kwargs.
    Returns the function's return value.
    Raises exceptions if the function doesn't exist, fails to load, or errors during execution.
    '''
    secure_name = secure_filename(name)
    if not secure_name:
        raise ValueError(f"Invalid function name '{name}' for calling.")

    file_path = os.path.join(FUNCTIONS_DIR, f"{secure_name}.py")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dynamic function '{secure_name}' not found at {file_path}")

    try:
        # Dynamically import the module
        # The module name should be unique, using the file name is common
        module_name = f"dynamic_functions.{secure_name}" 
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {secure_name}")

        module = importlib.util.module_from_spec(spec)
        # Add to sys.modules *before* execution to handle relative imports within the func
        # sys.modules[module_name] = module # Be careful with namespace pollution if many functions

        spec.loader.exec_module(module)
        logger.info(f"Dynamically loaded module for function '{secure_name}'.")

        # Assume the function name inside the file matches the filename
        func_to_call = getattr(module, secure_name, None)

        if not callable(func_to_call):
            # Maybe try to find *any* function if name doesn't match?
            # For now, strictly enforce matching name.
            raise AttributeError(f"Function '{secure_name}' not found or not callable within its module.")

        logger.info(f"Calling dynamic function '{secure_name}' with args: {kwargs}")
        result = func_to_call(**kwargs)
        logger.info(f"Dynamic function '{secure_name}' executed successfully.")
        return result

    except Exception as e:
        logger.error(f"Error calling dynamic function '{secure_name}': {e}")
        logger.debug(traceback.format_exc())
        # Re-raise the exception so the caller knows something went wrong
        raise # Or return a specific error object/dict