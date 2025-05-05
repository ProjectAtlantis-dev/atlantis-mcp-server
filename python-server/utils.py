#!/usr/bin/env python3
import os
import psutil
import logging
import json
import asyncio
from state import logger

# ANSI escape codes for colors
PINK = "\x1b[95m"  # Added Pink
RESET = "\x1b[0m"

# Path for the PID file
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.pid")

def check_server_running():
    """Check if a server is already running by examining the PID file.

    Returns:
        int or None: The PID of the running server, or None if no server is running.
    """
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

def create_pid_file():
    """Create a PID file with the current process ID.

    Returns:
        bool: True if the file was created successfully, False otherwise.
    """
    pid = os.getpid()
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(pid))
        logger.info(f"📝 Created PID file with server process ID: {pid}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create PID file: {str(e)}")
        return False

def remove_pid_file():
    """Remove the PID file if it exists."""
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            logger.info(f"🧹 Removed PID file")
    except Exception as e:
        logger.error(f"❌ Failed to remove PID file: {str(e)}")


"""
Utility functions available for dynamic functions to use.
Provides easy access to client-side logging and other shared functionality.
"""

import logging
from typing import Any

# Empty function for filename cleaning - placeholder for future implementation
def clean_filename(name: str) -> str:
    """
    Cleans/sanitizes a filename for filesystem usage.
    This is a placeholder function that will be implemented later.

    Args:
        name: The filename to clean

    Returns:
        Cleaned filename safe for filesystem usage
    """
    return name

# Global server reference to be set at startup
_server_instance = None

def set_server_instance(server):
    """Set the server instance for dynamic functions to use."""
    global _server_instance
    _server_instance = server
    logger.debug("Server instance set in utils module")

def get_server_instance():
    global _server_instance
    return _server_instance

def client_log(
    message: Any, 
    level: str = "info", 
    logger_name: str = None, 
    request_id: str = None, 
    client_id_for_routing: str = None,
    seq_num: int = None,
    entry_point_name: str = None
    ):
    """
    Send a log message to the client.

    This function can be imported and called from dynamic functions to send
    logs directly to the client using MCP notifications.

    Args:
        message: The message to log (can be a string or structured data)
        level: Log level ("debug", "info", "warning", "error")
        logger_name: Optional name to identify the logger source
        request_id: Optional ID of the original request that triggered this log
        client_id_for_routing: Optional client identifier to route the log message
        seq_num: Optional sequence number for client-side ordering
        entry_point_name: Name of the top-level function called by the request.
    """
    # Log locally first (always using INFO level for local display)
    seq_prefix = f"(Seq: {seq_num}) " if seq_num is not None else ""
    log_prefix = f"{PINK}CLIENT LOG [{level.upper()}] {seq_prefix}"
    log_suffix = f"(Client: {client_id_for_routing}, Req: {request_id}, Entry: {entry_point_name}, Logger: {logger_name}): {message}{RESET}"
    logger.info(f"{log_prefix}{log_suffix}") # Add seq_num to local log too

    # Send to client if server is available
    if _server_instance is not None:
        try:
            # Create a task to send the log asynchronously
            loop = asyncio.get_event_loop()
            if logger_name is None:
                logger_name = "dynamic_function"

            # Pass request_id, client_id_for_routing, AND seq_num to the server's method
            # NOTE: _server_instance.send_client_log MUST be updated to accept seq_num
            asyncio.create_task(_server_instance.send_client_log(level, message, logger_name, request_id, client_id_for_routing, seq_num, entry_point_name))
        except Exception as e:
            logger.error(f"Error scheduling client log task: {e}") # Updated error message clarity
    else:
        logger.warning("Cannot send client log: server instance not set")


# --- JSON Formatting Utility --- #

def format_json_log(data: dict) -> str:
    """Formats a Python dictionary into a pretty-printed JSON string for logging."""
    try:
        return json.dumps(data, indent=2, default=str) # Added default=str to handle non-serializable types gracefully
    except Exception as e:
        logger.error(f"❌ Error formatting JSON for logging: {e}")
        return str(data) # Fallback to string representation
