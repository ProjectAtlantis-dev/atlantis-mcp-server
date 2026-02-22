#!/usr/bin/env python3
import os
import psutil
import logging
import json
import asyncio
import re
import traceback
from state import logger
from typing import Any, Optional, TypedDict

# ANSI escape codes for colors
PINK = "\x1b[95m"  # Added Pink
RESET = "\x1b[0m"


"""
Utility functions available for dynamic functions to use.
Provides easy access to client-side logging and other shared functionality.
"""


# Empty function for filename cleaning - placeholder for future implementation
def clean_filename(name: str) -> str:
    """
    Validates and cleans a filename for filesystem usage.
    For function names, enforces Python identifier rules.
    Checks for double .py extensions and throws an exception if found.

    Args:
        name: The filename to clean

    Returns:
        Cleaned filename safe for filesystem usage

    Raises:
        ValueError: If the filename is invalid or contains illegal characters
    """
    if not name or not isinstance(name, str):
        raise ValueError("Invalid filename: must be a non-empty string")

    # Check for double .py extension
    if name.endswith('.py.py'):
        raise ValueError(f"Invalid filename '{name}': ends with double .py extension. Please provide a name without the .py suffix.")

    # Validate as a Python identifier (function names must be valid Python identifiers)
    # Python identifier rules: must start with letter or underscore, followed by letters, digits, or underscores
    # This rejects dots, asterisks, hyphens, and other special characters
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        # Provide helpful error message about what's wrong
        invalid_chars = re.findall(r'[^a-zA-Z0-9_]', name)
        if invalid_chars:
            invalid_chars_str = ', '.join(f"'{c}'" for c in sorted(set(invalid_chars)))
            raise ValueError(f"Invalid function name '{name}': contains illegal characters: {invalid_chars_str}. Function names must contain only letters, digits, and underscores, and must start with a letter or underscore.")
        elif name[0].isdigit():
            raise ValueError(f"Invalid function name '{name}': cannot start with a digit. Function names must start with a letter or underscore.")
        else:
            raise ValueError(f"Invalid function name '{name}': must be a valid Python identifier.")

    return name


class ParsedSearchTerm(TypedDict):
    """Parsed components of a compound search term."""
    owner: Optional[str]        # Remote owner (e.g., "admin")
    remote: Optional[str]       # Remote name (e.g., "admin")
    app: str                    # App/folder name (e.g., "Home")
    location: Optional[str]     # Location identifier
    function: str               # Function name (e.g., "chat")
    filename: str               # Derived filename (e.g., "Home/chat.py")


def parse_search_term(search_term: str) -> ParsedSearchTerm:
    """
    Parse a compound search term into its components.

    Format: "remote_owner*remote_name*app*location*function"

    Examples:
        "admin*admin*Home**chat" -> {owner: "admin", remote: "admin", app: "Home", location: None, function: "chat", filename: "Home/chat.py"}
        "**Home**kitty" -> {owner: None, remote: None, app: "Home", location: None, function: "kitty", filename: "Home/kitty.py"}

    Args:
        search_term: The compound search term to parse

    Returns:
        ParsedSearchTerm with all components extracted

    Raises:
        ValueError: If search_term is empty, malformed, or missing required fields (app, function)
    """
    if not search_term:
        raise ValueError("Search term cannot be empty")

    if "*" not in search_term:
        raise ValueError(f"Invalid search term '{search_term}': must be in compound format (owner*remote*app*location*function)")

    parts = search_term.split("*")
    if len(parts) < 5:
        raise ValueError(f"Invalid search term '{search_term}': expected 5 parts (owner*remote*app*location*function), got {len(parts)}")

    # Full format: remote_owner*remote_name*app*location*function
    owner = parts[0] if parts[0] else None
    remote = parts[1] if parts[1] else None
    app = parts[2] if parts[2] else None
    location = parts[3] if parts[3] else None
    function = parts[4] if parts[4] else None

    # Validate required fields
    if not function:
        raise ValueError(f"Invalid search term '{search_term}': function name is required (5th field)")

    # App is optional - keep None for root-level functions (function lookup expects None)

    # Derive filename from app + function
    if app:
        filename = f"{app}/{function}.py"
    else:
        filename = f"{function}.py"

    return {
        'owner': owner,
        'remote': remote,
        'app': app,
        'location': location,
        'function': function,
        'filename': filename
    }


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

async def client_log(
    message: Any,
    level: str = "info",
    logger_name: Optional[str] = None,
    request_id: Optional[str] = None,
    client_id_for_routing: Optional[str] = None,
    seq_num: Optional[int] = None,
    entry_point_name: Optional[str] = None,
    message_type: str = "text",
    stream_id: Optional[str] = None,
    is_private: bool = True
    ):
    """
    Send a log message to the client.

    IMPORTANT: This is the LOW-LEVEL implementation of client logging.
    Dynamic functions should NOT call this directly! Instead, use atlantis.client_log,
    which is a context-aware wrapper that automatically provides all the necessary
    context variables (request_id, client_id, etc.) to this function.

    atlantis.client_log -> utils.client_log -> server.send_client_log

    This function can be imported and called from dynamic functions to send
    logs directly to the client using MCP notifications, but atlantis.client_log
    is preferred in most cases.

    Args:
        message: The message to log (can be a string or structured data)
        level: Log level ("debug", "info", "warning", "error")
        logger_name: Optional name to identify the logger source
        request_id: Optional ID of the original request that triggered this log
        client_id_for_routing: Optional client identifier to route the log message
        seq_num: Optional sequence number for client-side ordering
        entry_point_name: Name of the top-level function called by the request.
        message_type: Type of message content ("text", "json", "image/png", etc.). Default is "text"
        is_private: If True (default), send only to requesting client.
                   If False, broadcast to all connected clients (used by scripts).
    """
    # Log locally first (always using INFO level for local display)
    seq_prefix = f"(Seq: {seq_num}) " if seq_num is not None else ""
    log_prefix = f"{PINK}CLIENT LOG [{level.upper()}] {seq_prefix}"

    # Trim long messages for debug output
    # If message is a dict or list, format it as pretty JSON
    if isinstance(message, (dict, list)):
        message_str = format_json_log(message, colored=True)
        # Don't truncate formatted JSON - we want to see the full structure
        trimmed_message = message_str
    else:
        message_str = str(message)
        # Try to detect if this is a JSON string and parse it
        if message_str.strip().startswith(('{', '[')):
            try:
                parsed_data = json.loads(message_str)
                message_str = format_json_log(parsed_data, colored=True)
                trimmed_message = message_str
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON, treat as regular string
                if len(message_str) > 200:
                    trimmed_message = message_str[:200] + f"... (truncated, full length: {len(message_str)})"
                else:
                    trimmed_message = message_str
        else:
            # Only truncate non-JSON strings
            if len(message_str) > 200:
                trimmed_message = message_str[:200] + f"... (truncated, full length: {len(message_str)})"
            else:
                trimmed_message = message_str

    log_suffix = f"(Client: {client_id_for_routing}, Req: {request_id}, Entry: {entry_point_name}, Logger: {logger_name}): {trimmed_message}{RESET}"
    logger.info(f"{log_prefix}{log_suffix}") # Add seq_num to local log too

    # Send to client if server is available
    if _server_instance is not None:
        try:

            # Await the server call to send the log/command and get a response
            if logger_name is None:
                logger_name = "dynamic_function"

            # Pass request_id, client_id_for_routing, AND seq_num to the server's method
            # Pass all parameters including message_type to the server's method
            # Create a task to send the client log without awaiting its completion here.
            # This makes utils.client_log effectively fire-and-forget from the caller's perspective.
            async def send_log_task():
                try:
                    task_result = await _server_instance.send_client_log(
                        level,
                        message,
                        logger_name,
                        request_id,
                        client_id_for_routing,
                        seq_num,
                        entry_point_name,
                        message_type,
                        stream_id,
                        is_private
                    )

                except Exception as task_e:
                    logger.error(f"❌ Error in send_log_task: {task_e}")
                    logger.error(f"❌ Exception details: {type(task_e).__name__}: {task_e}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")

            asyncio.create_task(send_log_task())
            #logger.debug(f"📋 CLIENT LOG/COMMAND TASK CREATED for client_id={client_id_for_routing}, request_id={request_id}, seq_num={seq_num}")
            return None # Return immediately, indicating task creation
        except Exception as e:
            logger.error(f"❌ Error in awaitable client_log: {e}")
            logger.error(f"❌ Exception details: {type(e).__name__}: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise # Re-raise the exception so the caller (atlantis.py) can handle it
    else:
        logger.warning("Cannot send client log/command: server instance not set")
        return None # Or raise an error, depending on desired behavior

async def execute_client_command_awaitable(
    client_id_for_routing: str,
    request_id: str, # Original MCP request ID for client context
    command: str, # The command string for the client
    command_data: Optional[Any] = None, # Optional data for the command
    seq_num: Optional[int] = None, # Sequence number for client-side ordering
    entry_point_name: Optional[str] = None, # Entry point name for logging
    user: Optional[str] = None, # User who initiated the request
    session_id: Optional[str] = None, # Session ID for request isolation
    shell_path: Optional[str] = None, # Shell path for request isolation
    message_type: str = "command", # Message type for the protocol (default "command" for backwards compat)
    is_private: bool = True # If False, cloud should broadcast to all clients
    ) -> Any:
    """
    Sends a command to a specific client via the server and waits for a dedicated response.
    This is a low-level utility intended to be called by atlantis.client_command for awaitable operations.
    It calls the server's 'send_awaitable_client_command' method.

    Args:
        client_id_for_routing: The ID of the client to send the command to.
        request_id: The original MCP request ID, for client-side context.
        command: The command identifier string.
        command_data: Optional data payload for the command.
        seq_num: Optional sequence number for client-side ordering.
        entry_point_name: Optional name of the entry point function for logging.
        user: Optional user who initiated the request (for request isolation).
        session_id: Optional session ID (for request isolation).
        shell_path: Optional shell path in the command tree (for request isolation).

    Returns:
        The result from the client's command execution, as returned by the server.

    Raises:
        McpError: If the server call fails, client response times out, or client returns an error.
        RuntimeError: If the server instance is not available or doesn't support awaitable commands.
    """
    global _server_instance
    if _server_instance is None:
        logger.error("❌ Cannot execute awaitable client command: server instance not set.")
        raise RuntimeError("Server instance not available for awaitable command.")

    if not hasattr(_server_instance, 'send_awaitable_client_command'):
        logger.error("❌ Server instance does not have 'send_awaitable_client_command' method. This is required for true awaitable commands.")
        raise RuntimeError("Server instance is outdated or incorrect for awaitable commands.")

    try:
        logger.info(f"🚀 Utils: Relaying dedicated awaitable command '{command}' to server for client {client_id_for_routing}, seq_num={seq_num}")
        if isinstance(command_data, dict):
            logger.info(f"   📦 Command data type: {type(command_data)}, data:\n{format_json_log(command_data)}")
        else:
            logger.info(f"   📦 Command data type: {type(command_data)}, data: {command_data}")
        # This specifically calls the method designed for awaitable command-response cycles
        result = await _server_instance.send_awaitable_client_command(
            client_id_for_routing=client_id_for_routing,
            request_id=request_id,
            command=command,
            command_data=command_data,
            seq_num=seq_num,
            entry_point_name=entry_point_name,
            user=user,
            session_id=session_id,
            shell_path=shell_path,
            message_type=message_type,
            is_private=is_private
        )
        logger.info(f"✅ Utils: Received result for dedicated awaitable command '{command}' from server.")
        logger.info(f"   📥 Result type: {type(result)}, length: {len(str(result)) if result else 0}")
        return result
    except Exception as e:
        # Errors (including McpError for timeouts/client errors from server's send_awaitable_client_command) will propagate
        # Server layer already logged with command context, just re-raise
        raise

async def execute_stream_awaitable(
    client_id_for_routing: str,
    request_id: str,
    message_type: str,  # 'stream_start', 'stream', or 'stream_end'
    message: Any,
    stream_id: str,
    seq_num: Optional[int] = None,
    entry_point_name: Optional[str] = None,
    level: str = "INFO",
    logger_name: Optional[str] = None
    ) -> Any:
    """
    Sends a stream message to a specific client via the server and waits for acknowledgment.
    This is the awaitable version of stream operations for when AWAIT_STREAM_ACK is enabled.

    Args:
        client_id_for_routing: The ID of the client to send the stream message to.
        request_id: The original MCP request ID, for client-side context.
        message_type: The type of stream message ('stream_start', 'stream', or 'stream_end').
        message: The message content to send.
        stream_id: The unique identifier for this stream.
        seq_num: Optional sequence number for client-side ordering.
        entry_point_name: Optional name of the entry point function for logging.
        level: Log level (default "INFO").
        logger_name: Optional name to identify the logger source.

    Returns:
        The acknowledgment result from the client.

    Raises:
        McpError: If the server call fails, client response times out, or client returns an error.
        RuntimeError: If the server instance is not available or doesn't support awaitable streams.
    """
    global _server_instance
    if _server_instance is None:
        logger.error("❌ Cannot execute awaitable stream: server instance not set.")
        raise RuntimeError("Server instance not available for awaitable stream.")

    if not hasattr(_server_instance, 'send_awaitable_stream'):
        logger.error("❌ Server instance does not have 'send_awaitable_stream' method. This is required for awaitable streams.")
        raise RuntimeError("Server instance is outdated or incorrect for awaitable streams.")

    try:
        #logger.info(f"🌊 Utils: Relaying awaitable stream '{message_type}' to server for client {client_id_for_routing}, stream_id={stream_id}, seq_num={seq_num}")
        result = await _server_instance.send_awaitable_stream(
            client_id_for_routing=client_id_for_routing,
            request_id=request_id,
            message_type=message_type,
            message=message,
            stream_id=stream_id,
            seq_num=seq_num,
            entry_point_name=entry_point_name,
            level=level,
            logger_name=logger_name
        )
        #logger.info(f"✅ Utils: Received ack for awaitable stream '{message_type}' (stream_id={stream_id}): {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Utils: Exception in execute_stream_awaitable: {type(e).__name__}: {e}")
        # Server layer already logged, just re-raise
        raise

# --- JSON Formatting Utility --- #

def format_json_log(data: dict | list, colored: bool = True) -> str:
    """
    Formats a Python dictionary or list into a pretty-printed JSON string for logging.

    Args:
        data: Dictionary or list to format
        colored: If True, adds ANSI color codes for terminal output (default: True)

    Returns:
        Formatted JSON string with optional colors
    """
    try:
        json_str = json.dumps(data, indent=2, default=str)

        if not colored:
            return json_str

        # ANSI color codes
        KEY_COLOR = "\x1b[36m"      # Cyan for keys
        STRING_COLOR = "\x1b[33m"   # Yellow for string values
        NUMBER_COLOR = "\x1b[35m"   # Magenta for numbers
        BOOL_COLOR = "\x1b[32m"     # Green for booleans
        NULL_COLOR = "\x1b[90m"     # Grey for null
        RESET = "\x1b[0m"

        # Color keys (strings followed by colon)
        json_str = re.sub(r'"([^"]+)"\s*:', rf'{KEY_COLOR}"\1"{RESET}:', json_str)

        # Color string values (strings not followed by colon)
        json_str = re.sub(r':\s*"([^"]*)"', rf': {STRING_COLOR}"\1"{RESET}', json_str)

        # Color numbers
        json_str = re.sub(r':\s*(-?\d+\.?\d*)', rf': {NUMBER_COLOR}\1{RESET}', json_str)

        # Color booleans
        json_str = re.sub(r':\s*(true|false)', rf': {BOOL_COLOR}\1{RESET}', json_str)

        # Color null
        json_str = re.sub(r':\s*(null)', rf': {NULL_COLOR}\1{RESET}', json_str)

        return json_str
    except Exception as e:
        logger.error(f"❌ Error formatting JSON for logging: {e}")
        return str(data) # Fallback to string representation
