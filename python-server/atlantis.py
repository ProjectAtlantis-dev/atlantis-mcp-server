import contextvars
import inspect # Ensure inspect is imported
import asyncio # Added for Lock
from typing import Callable, Optional, Any, List # Added List
from utils import client_log as util_client_log # For client_log, client_image, client_html
from utils import execute_client_command_awaitable, execute_stream_awaitable, format_json_log # For client_command and streaming
import uuid
import os
import os.path
import json
import base64
import logging
import mimetypes
from datetime import datetime, timezone

logger = logging.getLogger("mcp_server")

# --- Context Variables ---
# client_log_func: Holds the partially bound client_log function for the current request
_client_log_var: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar("_client_log_var", default=None)
# request_id: The unique ID of the current MCP request
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_request_id_var", default=None)
# client_id: The ID of the client that initiated the current request
_client_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_client_id_var", default=None)

# entry_point_name: The name of the top-level dynamic function called by the request
_entry_point_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_entry_point_name_var", default=None)

_user_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_user_var", default=None)
_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_session_id_var", default=None)
_command_seq_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("_command_seq_var", default=None)
_shell_path_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_shell_path_var", default=None)

# Owner of the remote server instance
_owner: Optional[str] = ""
_owner_usernames: List[str] = []  # List of owner usernames for permission checks

# Simple task collection for logging tasks
_log_tasks_var: contextvars.ContextVar[Optional[List[asyncio.Task]]] = contextvars.ContextVar("_log_tasks_var", default=None)

# Per-stream sequence numbers - each stream_id gets its own counter
_stream_seq_counters: dict = {}

# Per-request sequence numbers - each (request_id, shell_path) gets its own counter
_request_seq_counters: dict = {}

# Click callback lookup table - maps click keys to callback functions
_click_callbacks: dict = {}

# upload callback lookup table - maps upload keys to callback functions
_upload_callbacks: dict = {}

# Flags to control whether streaming calls wait for acknowledgment
# Set to True to enable awaitable streaming (helps fix streaming issues)
# Set to False for fire-and-forget streaming (original behavior)
AWAIT_STREAM_START_ACK: bool = True   # Wait for ack on stream_start
AWAIT_STREAM_MSG_ACK: bool = True     # Wait for ack on each stream message
AWAIT_STREAM_END_ACK: bool = True     # Wait for ack on stream_end



# --- Shared Object Container ---
# This container persists across dynamic function reloads and can store
# shared resources like database connections, cache objects, etc.
class SharedContainer:
    """Container for objects that need to persist across dynamic function reloads"""
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        """Get a value from the shared container"""
        return self._data.get(key, default)

    def set(self, key, value):
        """Store a value in the shared container"""
        self._data[key] = value
        return value

    def remove(self, key):
        """Remove a value from the shared container"""
        if key in self._data:
            del self._data[key]
            return True
        return False

    def keys(self):
        """Get all keys in the shared container"""
        return list(self._data.keys())

# Initialize the shared container
shared = SharedContainer()

# --- Helper Functions ---

def _trim_message_for_debug(message: Any, max_len: int = 200) -> str:
    """Trim a message for debug output to avoid console spam from long content like base64."""
    msg_str = str(message)
    if len(msg_str) <= max_len:
        return msg_str
    return msg_str[:max_len] + f"... (truncated, full length: {len(msg_str)})"

async def get_and_increment_stream_seq_num(stream_id: str) -> int:
    """Get and increment the sequence number for a specific stream.

    Args:
        stream_id: The unique stream identifier

    Returns:
        The current sequence number for this stream before incrementing
    """
    if stream_id not in _stream_seq_counters:
        _stream_seq_counters[stream_id] = 1

    current_seq = _stream_seq_counters[stream_id]
    _stream_seq_counters[stream_id] += 1
    return current_seq

async def get_and_increment_seq_num(context_name: str = "operation") -> int:
    """Get and increment the sequence number in a thread-safe way.

    Sequence numbers are tracked per (request_id, shell_path) to ensure uniqueness
    across function hops within the same request and shell path.

    Args:
        context_name: Name of the calling context for error reporting

    Returns:
        The current sequence number before incrementing, or -1 if error
    """
    # NOTE: No lock needed because the server handles message ordering.
    # The server ensures messages are processed sequentially, eliminating the need for
    # client-side locking of sequence number generation. While concurrent tasks within
    # the same request context share the same counter, server-side ordering
    # guarantees prevent race conditions that could cause duplicate sequence numbers.
    request_id = _request_id_var.get()
    if request_id is None:
        logger.error(f"{context_name} - request_id is None. Cannot get sequence number.")
        return -1

    shell_path = _shell_path_var.get()
    # Use composite key of (request_id, shell_path) for per-shell sequencing
    counter_key = (request_id, shell_path)

    if counter_key not in _request_seq_counters:
        _request_seq_counters[counter_key] = 1

    current_seq = _request_seq_counters[counter_key]
    _request_seq_counters[counter_key] += 1
    return current_seq

# --- Accessor Functions ---

async def client_log(message: Any, level: str = "INFO", message_type: str = "text", is_private: bool = True):
    """Sends a log message back to the requesting client for the current context.
    Includes a sequence number and automatically determines the calling function name.
    Also includes the original entry point function name.

    NOTE: This is a WRAPPER around the lower-level utils.client_log function.
    This function automatically captures context data (like request_id, caller name, etc.)
    and forwards it to utils.client_log. Dynamic functions should use THIS version,
    not utils.client_log directly.

    The message_type parameter specifies what kind of content is being sent:
    - "text" (default): A plain text message
    - "json": A JSON object or structured data
    - "image/*": Various image formats (e.g., "image/png", "image/jpeg")

    Args:
        is_private: If True (default), send only to requesting client.
                   If False, broadcast to all connected clients (used by client_script).

    Calls the underlying log function directly; async dispatch is handled internally.
    """
    log_func = _client_log_var.get()
    if log_func:
        caller_name = "unknown_caller" # Default for immediate caller
        entry_point_name = _entry_point_name_var.get() or "unknown_entry_point" # Get entry point from context

        try:
            # --- Get immediate caller function name ---
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_name = frame.f_back.f_code.co_name
            del frame
        except Exception as inspect_err:
            logger.warning(f"Could not inspect caller frame for client_log: {inspect_err}")

        try:
            # Get current sequence number and increment it for the next call
            current_seq_to_send = await get_and_increment_seq_num(context_name="client_log")
            # If current_seq_to_send is -1, the helper function already logged the error

            # Get context values with null checks
            client_id = _client_id_var.get()
            request_id = _request_id_var.get()

            if client_id is None or request_id is None:
                logger.warning(f"Missing context data - client_id: {client_id}, request_id: {request_id}")
                return None

            task = await util_client_log(
                client_id_for_routing=client_id,
                request_id=request_id,
                entry_point_name=entry_point_name,
                message_type=message_type,
                message=message,
                level=level,
                logger_name=caller_name,  # Pass the caller function name
                seq_num=current_seq_to_send, # Pass the obtained sequence number
                is_private=is_private  # For broadcast control (scripts only)
            )
            # task is the asyncio.Task returned by utils.client_log

            # Track the task if we have one
            if task is not None:
                tasks = _log_tasks_var.get()
                if tasks is None:
                    # Initialize task list if not already done
                    tasks = []
                    _log_tasks_var.set(tasks)
                tasks.append(task)

            return None # Return None in either case
        except Exception as e:
            logger.error(f"Failed during async client_log call (after inspect): {e}")
            # Decide if to re-raise or return an error indicator
            raise
    else:
        logger.warning(f"client_log called but no logger in context. Message: {message}")
        return None # Or raise an error

async def tool_result(name: str, result: Any):
    """Sends a tool call result back to the requesting client to be added to the transcript.
    This allows the LLM to see tool results in the next conversation turn.
    """
    return await client_command("tool", f"Tool {name} result: {result}", message_type="tool")

# --- Other Accessors ---
# this is established by the tool caller
def get_request_id() -> Optional[str]:
    """Returns the request_id"""
    return _request_id_var.get()

# this is established in server.py when a connection is made
def get_client_id() -> Optional[str]:
    """Returns the client_id"""
    return _client_id_var.get()

# this is usually the cloud session num, which is per user
def get_session_id() -> Optional[str]:
    """Returns the session_id for this function call"""
    return _session_id_var.get()

def get_caller() -> Optional[str]:
    """Returns the username who called this function"""
    return _user_var.get()

def get_command_seq() -> Optional[int]:
    """Returns the command sequence number for this function call"""
    return _command_seq_var.get()

def get_invoking_tool_name() -> Optional[str]:
    """Returns the name of the tool that initiated the current execution chain."""
    return _entry_point_name_var.get()

def get_owner() -> str:
    """Returns the user who owns this remote"""
    return _owner or ""

def get_owner_usernames() -> List[str]:
    """Returns the list of owner usernames for permission checks"""
    return _owner_usernames

def is_owner(username: str) -> bool:
    """Check if the given username is an owner"""
    return username in _owner_usernames

# --- Setter Functions (primarily for internal use by dynamic_function_manager) ---

def _set_owner(new_owner: str):
    """Sets the owner of the remote server instance. For internal use."""
    global _owner
    _owner = new_owner

def _set_owner_usernames(usernames: List[str]):
    """Sets the list of owner usernames. For internal use."""
    global _owner_usernames
    _owner_usernames = usernames


def set_context(
        client_log_func: Callable,
        request_id: str,
        client_id: str,
        entry_point_name: str,
        user: Optional[str] = None,
        session_id: Optional[str] = None,
        command_seq: Optional[int] = None,
        shell_path: Optional[str] = None):
    """Sets all context variables and returns a tuple of their tokens for resetting."""
    client_log_token = _client_log_var.set(client_log_func)
    request_id_token = _request_id_var.set(request_id)
    client_id_token = _client_id_var.set(client_id)
    entry_point_token = _entry_point_name_var.set(entry_point_name)

    # Handle optional user context
    # Ensure _user_var is always set, even if to None, to get a valid token for reset_context
    actual_user = user if user is not None else None # Explicitly use None if user is not provided
    user_token = _user_var.set(actual_user)

    # Handle optional session_id context
    # Ensure _session_id_var is always set, even if to None, to get a valid token for reset_context
    actual_session_id = session_id if session_id is not None else None # Explicitly use None if session_id is not provided
    session_id_token = _session_id_var.set(actual_session_id)

    # Handle optional command_seq context
    # Ensure _command_seq_var is always set, even if to None, to get a valid token for reset_context
    actual_command_seq = command_seq if command_seq is not None else None # Explicitly use None if command_seq is not provided
    command_seq_token = _command_seq_var.set(actual_command_seq)

    # Handle optional shell_path context
    actual_shell_path = shell_path if shell_path is not None else None
    shell_path_token = _shell_path_var.set(actual_shell_path)

    return (client_log_token, request_id_token, client_id_token, entry_point_token, user_token, session_id_token, command_seq_token, shell_path_token)

# use sendChatter to send commands directly from browser

def reset_context(tokens: tuple):
    """Resets the context variables using the provided tuple of tokens."""
    # Expected order: client_log, request_id, client_id, entry_point, user, session_id, command_seq, shell_path
    if not isinstance(tokens, tuple) or len(tokens) != 8:
        logger.error(f"reset_context expected a tuple of 8 tokens, got {tokens}")
        # Add more robust error handling or logging as needed
        return

    # Unpack tokens
    client_log_token, request_id_token, client_id_token, entry_point_token, user_token, session_id_token, command_seq_token, shell_path_token = tokens

    # Reset each context variable if its token is present (not strictly necessary with .set(None) giving a token)
    _client_log_var.reset(client_log_token)
    _request_id_var.reset(request_id_token)
    _client_id_var.reset(client_id_token)
    _entry_point_name_var.reset(entry_point_token)
    _user_var.reset(user_token) # user_token will be valid even if user was None
    _session_id_var.reset(session_id_token) # session_id_token will be valid even if session_id was None
    _command_seq_var.reset(command_seq_token) # command_seq_token will be valid even if command_seq was None
    _shell_path_var.reset(shell_path_token) # shell_path_token will be valid even if shell_path was None


# --- Utility Functions ---

async def client_image(image_path: str, image_format: Optional[str] = None):
    """Sends an image back to the requesting client for the current context.
    This is a wrapper around client_log that automatically loads the image,
    converts it to base64, and sets the appropriate message type.

    Args:
        image_path: Path to the image file to send
        image_format: Optional MIME type of the image (e.g., "image/png", "image/jpeg").
                     If not provided, will be auto-detected from file extension.

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the file
    """
    # Auto-detect MIME type from file extension if not provided
    if image_format is None:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            # Default to PNG if we can't determine the type
            image_format = "image/png"
        else:
            image_format = mime_type

    # Convert image to base64
    base64_data = image_to_base64(image_path)

    # Format as proper data URL
    prefixed_data = f"data:{image_format};base64,{base64_data}"

    # Send via client_command with appropriate message_type for awaitable behavior
    result = await client_command("image", prefixed_data, message_type=image_format)
    return result

def image_to_base64(image_path: str) -> str:
    """Loads an image from the given file path and converts it to a base64 string.

    Args:
        image_path: The path to the image file to load

    Returns:
        A base64-encoded string representation of the image

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the file
    """
    # Verify file exists to provide a helpful error
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        # Read the binary data from the file
        with open(image_path, "rb") as image_file:
            # Convert binary data to base64 encoded string
            encoded_string = base64.b64encode(image_file.read())
            # Return as UTF-8 string
            return encoded_string.decode('utf-8')
    except IOError as e:
        # Log the error for debugging
        logger.error(f"Error converting image to base64: {e}")
        # Re-raise to allow caller to handle
        raise

async def client_video(video_path: str, video_format: Optional[str] = None):
    """Sends a video back to the requesting client for the current context.
    This is a wrapper around client_log that automatically loads the video,
    converts it to base64, and sets the appropriate message type.

    Args:
        video_path: Path to the video file to send
        video_format: Optional MIME type of the video (e.g., "video/mp4", "video/webm").
                     If not provided, will be auto-detected from file extension.

    Raises:
        FileNotFoundError: If the video file doesn't exist
        IOError: If there's an error reading the file
    """
    # Auto-detect MIME type from file extension if not provided
    if video_format is None:
        mime_type, _ = mimetypes.guess_type(video_path)
        if not mime_type or not mime_type.startswith('video/'):
            # Default to MP4 if we can't determine the type
            video_format = "video/mp4"
        else:
            video_format = mime_type

    # Convert video to base64
    base64_data = video_to_base64(video_path)

    # Format as proper data URL
    prefixed_data = f"data:{video_format};base64,{base64_data}"

    # Send via client_command with appropriate message_type for awaitable behavior
    result = await client_command("video", prefixed_data, message_type=video_format)
    return result

def video_to_base64(video_path: str) -> str:
    """Loads a video from the given file path and converts it to a base64 string.

    Args:
        video_path: The path to the video file to load

    Returns:
        A base64-encoded string representation of the video

    Raises:
        FileNotFoundError: If the video file doesn't exist
        IOError: If there's an error reading the file
    """
    # Verify file exists to provide a helpful error
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        # Read the binary data from the file
        with open(video_path, "rb") as video_file:
            # Convert binary data to base64 encoded string
            encoded_string = base64.b64encode(video_file.read())
            # Return as UTF-8 string
            return encoded_string.decode('utf-8')
    except IOError as e:
        # Log the error for debugging
        logger.error(f"Error converting video to base64: {e}")
        # Re-raise to allow caller to handle
        raise

async def stream_start(sid: str, who: str) -> str:
    """Starts a new stream and returns a unique stream_id.
    Sends a 'stream_start' message to the client.

    Args:
        sid: Optional session ID to associate with this stream
        who: String identifier for who/what is starting the stream

    """
    stream_id_to_send = str(uuid.uuid4())
    actual_client_id = _client_id_var.get() # Get the actual client ID from context
    request_id = _request_id_var.get()
    entry_point_name = _entry_point_name_var.get() or "unknown_entry_point"

    # Check for required context data
    if actual_client_id is None or request_id is None:
        logger.warning(f"Missing context data in stream_start - client_id: {actual_client_id}, request_id: {request_id}")
        raise ValueError("Missing required context data for stream_start")

    caller_name = "unknown_caller"
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_name = frame.f_back.f_code.co_name
        del frame
    except Exception as inspect_err:
        logger.warning(f"Could not inspect caller frame for stream_start: {inspect_err}")

    # Get and increment sequence number using the per-stream helper function
    current_seq_to_send = await get_and_increment_stream_seq_num(stream_id_to_send)

    try:
        message_data = {"status": "started", "sid":sid, "who": who}

        # Use awaitable pattern if AWAIT_STREAM_START_ACK is enabled
        if AWAIT_STREAM_START_ACK:
            #logger.info(f"🌊 stream_start: AWAIT_STREAM_START_ACK is True, calling execute_stream_awaitable...")
            ack_result = await execute_stream_awaitable(
                client_id_for_routing=actual_client_id,
                request_id=request_id,
                message_type='stream_start',
                message=message_data,
                stream_id=stream_id_to_send,
                seq_num=current_seq_to_send,
                entry_point_name=entry_point_name,
                level="INFO",
                logger_name=caller_name
            )
            #logger.info(f"🌊 stream_start ack received: {ack_result}")
        else:
            # Original fire-and-forget behavior
            await util_client_log(
                seq_num=current_seq_to_send, # Pass the obtained sequence number
                message=message_data,
                level="INFO",
                logger_name=caller_name,
                request_id=request_id,
                client_id_for_routing=actual_client_id, # Route using actual client_id
                entry_point_name=entry_point_name,
                message_type='stream_start',
                stream_id=stream_id_to_send # Pass the generated stream_id separately
            )
        return stream_id_to_send # Return the generated stream_id to the caller
    except Exception as e:
        logger.error(f"Failed during async stream_start call: {e}")
        raise

async def stream(message: str, stream_id_param: str):
    """Sends a stream message snippet back to the client using a provided stream_id.
    """
    actual_client_id = _client_id_var.get() # Get the actual client ID from context
    request_id = _request_id_var.get()
    entry_point_name = _entry_point_name_var.get() or "unknown_entry_point"

    # Check for required context data
    if actual_client_id is None or request_id is None:
        logger.warning(f"Missing context data in stream - client_id: {actual_client_id}, request_id: {request_id}")
        raise ValueError("Missing required context data for stream")

    caller_name = "unknown_caller"
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_name = frame.f_back.f_code.co_name
        del frame
    except Exception as inspect_err:
        logger.warning(f"Could not inspect caller frame for stream: {inspect_err}")

    # Get and increment sequence number using the per-stream helper function
    current_seq_to_send = await get_and_increment_stream_seq_num(stream_id_param)

    try:
        # Use awaitable pattern if AWAIT_STREAM_MSG_ACK is enabled
        if AWAIT_STREAM_MSG_ACK:
            #logger.debug(f"🌊 stream: AWAIT_STREAM_MSG_ACK is True, calling execute_stream_awaitable for seq {current_seq_to_send}...")
            result = await execute_stream_awaitable(
                client_id_for_routing=actual_client_id,
                request_id=request_id,
                message_type='stream',
                message=message,
                stream_id=stream_id_param,
                seq_num=current_seq_to_send,
                entry_point_name=entry_point_name,
                level="INFO",
                logger_name=caller_name
            )
            #logger.debug(f"🌊 stream ack received for seq {current_seq_to_send}: {result}")
        else:
            # Original fire-and-forget behavior
            result = await util_client_log(
                seq_num=current_seq_to_send, # Pass the obtained sequence number
                message=message,
                level="INFO",
                logger_name=caller_name,
                request_id=request_id,
                client_id_for_routing=actual_client_id, # Route using actual client_id
                entry_point_name=entry_point_name,
                message_type='stream',
                stream_id=stream_id_param # Pass the provided stream_id separately
            )
        return result
    except Exception as e:
        logger.error(f"Failed during async stream call: {e}")
        raise

async def stream_end(stream_id_param: str):
    """Sends a stream_end message to the client, indicating the end of a stream, using a provided stream_id.
    """
    actual_client_id = _client_id_var.get() # Get the actual client ID from context
    request_id = _request_id_var.get()
    entry_point_name = _entry_point_name_var.get() or "unknown_entry_point"

    # Check for required context data
    if actual_client_id is None or request_id is None:
        logger.warning(f"Missing context data in stream_end - client_id: {actual_client_id}, request_id: {request_id}")
        raise ValueError("Missing required context data for stream_end")

    caller_name = "unknown_caller"
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_name = frame.f_back.f_code.co_name
        del frame
    except Exception as inspect_err:
        logger.warning(f"Could not inspect caller frame for stream_end: {inspect_err}")

    # Get and increment sequence number using the per-stream helper function
    current_seq_to_send = await get_and_increment_stream_seq_num(stream_id_param)

    try:
        # Use awaitable pattern if AWAIT_STREAM_END_ACK is enabled
        if AWAIT_STREAM_END_ACK:
            #logger.info(f"🌊 stream_end: AWAIT_STREAM_END_ACK is True, calling execute_stream_awaitable...")
            result = await execute_stream_awaitable(
                client_id_for_routing=actual_client_id,
                request_id=request_id,
                message_type='stream_end',
                message="",
                stream_id=stream_id_param,
                seq_num=current_seq_to_send,
                entry_point_name=entry_point_name,
                level="INFO",
                logger_name=caller_name
            )
            #logger.info(f"🌊 stream_end ack received: {result}")
        else:
            # Original fire-and-forget behavior
            result = await util_client_log(
                seq_num=current_seq_to_send, # Pass the obtained sequence number
                message="",
                level="INFO",
                logger_name=caller_name,
                request_id=request_id,
                client_id_for_routing=actual_client_id, # Route using actual client_id
                entry_point_name=entry_point_name,
                message_type='stream_end',
                stream_id=stream_id_param # Pass the provided stream_id separately
            )
        return result
    except Exception as e:
        logger.error(f"Failed during async stream_end call: {e}")
        raise


async def client_command(command: str, data: Any = None, message_type: str = "command", is_private: bool = True) -> Any:
    """Sends a command message to the client and waits for a specific acknowledgment and result.

    This function is for commands that require the server to wait for completion
    or a specific response from the client before proceeding.

    Args:
        command: The command string identifier.
        data: Optional JSON-serializable data associated with the command.
        message_type: The message type for the protocol (default "command").
        is_private: If True (default), send only to requesting client.
                   If False, cloud should broadcast to all connected clients.

    Returns:
        The result returned by the client for the command.

    Raises:
        RuntimeError: If context variables (client_id, request_id) are not set.
        McpError: Propagated from underlying calls if timeouts or client-side errors occur.
    """
    # === ENTRY POINT LOGGING - ALWAYS LOG ===
    logger.warning(f"🚨🚨🚨 CLIENT_COMMAND ENTERED 🚨🚨🚨")
    logger.warning(f"🚨 Command: '{command}'")
    logger.warning(f"🚨 First char: '{command[0] if command else 'EMPTY'}' (ord: {ord(command[0]) if command else 'N/A'})")
    logger.warning(f"🚨 Starts with %: {command.startswith('%') if command else False}")
    if isinstance(data, (dict, list)):
        logger.warning(f"🚨 Data:\n{format_json_log(data, colored=True)}")
    else:
        logger.warning(f"🚨 Data: {data}")
    logger.warning(f"🚨🚨🚨 END ENTRY LOG 🚨🚨🚨")

    # Get necessary context for routing and correlation
    client_id = _client_id_var.get()
    request_id = _request_id_var.get()
    entry_point_name = _entry_point_name_var.get() # Now needed for logging with seq_num
    user = _user_var.get()  # Who's calling
    session_id = _session_id_var.get()  # Which session
    shell_path = _shell_path_var.get()  # Where in the command tree

    logger.warning(f"🚨 Context: client_id={client_id}, request_id={request_id}, entry_point={entry_point_name}, user={user}, session={session_id}, shell={shell_path}")

    if not client_id or not request_id:
        # This should ideally not happen if called within a proper request context
        logger.error(f"client_command called without client_id or request_id in context. Command: {command}")
        raise RuntimeError("Client ID or Request ID not found in context for client_command.")

    try:
        # Get current sequence number and increment it for the next call
        # Using the helper function for consistent sequence number management
        current_seq_to_send = await get_and_increment_seq_num(context_name="client_command")
        logger.warning(f"🚨 Got seq_num: {current_seq_to_send}")

        # Extra distinctive logging for tool calls (commands starting with %)
        if command.startswith('%'):
            logger.warning(f"🔧🔧🔧 TOOL CALL DETECTED (% prefix) 🔧🔧🔧")
            logger.warning(f"🔧 Command: {command}")
            logger.warning(f"🔧 Client: {client_id}")
            logger.warning(f"🔧 Request: {request_id}")
            logger.warning(f"🔧 Seq: {current_seq_to_send}")
            logger.warning(f"🔧 Data: {format_json_log(data) if isinstance(data, dict) else data}")
            logger.warning(f"🔧🔧🔧 END TOOL CALL INFO 🔧🔧🔧")

        logger.warning(f"🚨 About to call execute_client_command_awaitable...")
        logger.info(f"Atlantis: Sending awaitable command '{command}' for client {client_id}, request {request_id}, seq {current_seq_to_send}")
        # Call the dedicated utility function for awaitable commands
        logger.warning(f"🚨 Calling execute_client_command_awaitable with command='{command}'")
        result = await execute_client_command_awaitable(
            client_id_for_routing=client_id,
            request_id=request_id,
            command=command,
            command_data=data,
            seq_num=current_seq_to_send,  # Pass the sequence number
            entry_point_name=entry_point_name,  # Pass the entry point name for logging
            user=user,  # Pass user for unique request tracking
            session_id=session_id,  # Pass session_id for unique request tracking
            shell_path=shell_path,  # Pass shell_path for unique request tracking
            message_type=message_type,  # Pass message_type for the protocol
            is_private=is_private  # Pass is_private for broadcast control
        )
        logger.debug(f"Atlantis: Received result for awaitable command '{command}', type: {type(result)}")

        return result
    except Exception as e:
        logger.warning(f"🚨 EXCEPTION in client_command for '{command}': {type(e).__name__}: {e}")
        # Server layer already logged with enhanced error message including command context
        # Just re-raise to let the dynamic function manager handle final logging
        raise

async def client_html(content: str):
    """Sends HTML content back to the requesting client for rendering

    Args:
        content: The HTML content to send
    """
    # Use client_command for awaitable response with proper message_type
    result = await client_command("html", content, message_type="html")
    return result

async def client_script(content: str, is_private: bool = True):
    """Sends Javascript content back to the requesting client for rendering

    Args:
        content: The Javascript content to send
        is_private: If True (default), script only runs on the requesting client.
                   If False, script broadcasts to all connected clients.
    """
    # Use client_command for awaitable response with proper message_type
    result = await client_command("script", content, message_type="script", is_private=is_private)
    return result

async def set_background(image_path: str, image_format: Optional[str] = None):
    """Sets the background image for the client UI.

    Args:
        image_path: Path to the image file
        image_format: Optional MIME type of the image (e.g., "image/png", "image/jpeg").
                     If not provided, will be auto-detected from file extension.
    """
    # Auto-detect MIME type from file extension if not provided
    if image_format is None:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            # Default to PNG if we can't determine the type
            image_format = "image/png"
        else:
            image_format = mime_type

    # Convert image to base64
    base64_data = image_to_base64(image_path)

    # Format as proper data URL
    prefixed_data = f"data:{image_format};base64,{base64_data}"

    result = await client_command("background", prefixed_data, message_type="background")
    return result

async def client_markdown(content: str):
    """Sends Markdown content back to the requesting client for rendering

    Args:
        content: The Markdown content to send
    """
    # Send via client_command with message_type for awaitable behavior
    result = await client_command("md", content, message_type="md")
    return result

async def client_data(description: str, data: Any, column_formatter: Optional[dict] = None):
    """Sends a Python object as serialized JSON back to the requesting client for styled rendering.
    If an array of objects, will automatically be displayed as a table.

    Args:
        description: A title/description of what the data represents
        data: The Python object to serialize and send (must be JSON-serializable)
        column_formatter: Optional dict mapping column names to formatting options (e.g., {"name": {"title": "Name"}})

    Returns:
        The result returned by the underlying client_log function

    Raises:
        TypeError: If the data cannot be serialized to JSON
    """
    try:
        # Create a wrapper object with description and data
        wrapped_data = {
            "description": description,
            "data": data
        }

        # Add column_formatter if provided
        if column_formatter is not None:
            wrapped_data["format"] = column_formatter

        # Try to serialize the data to JSON to verify it's valid
        json_str = json.dumps(wrapped_data)

        # Send via client_command with message_type for awaitable behavior
        result = await client_command("data", json_str, message_type="data")
        return result
    except TypeError as e:
        logger.error(f"Failed to serialize data to JSON: {e}")
        raise TypeError(f"Data is not JSON-serializable: {e}")
    except Exception as e:
        logger.error(f"Failed during client_data call: {e}")
        raise

async def gather_logs():
    """Wait for all pending client log tasks to complete.

    This is useful if you want to ensure all logs are sent before returning from
    a dynamic function or before starting a new operation that depends on logs
    being delivered.

    Returns:
        bool: True if tasks were gathered, False if no tasks to gather

    Example:
        ```python
        # Send some logs
        await atlantis.client_log("Log 1")
        await atlantis.client_log("Log 2")

        # Wait for them all to be sent
        await atlantis.gather_logs()
        ```
    """
    tasks = _log_tasks_var.get()
    if not tasks:
        return False

    # Create a copy of the tasks list
    tasks_to_gather = tasks.copy()
    # Clear the original list
    tasks.clear()

    # Wait for all tasks to complete
    if tasks_to_gather:
        await asyncio.gather(*tasks_to_gather)
        return True

    return False

async def owner_log(message: str):
    """
    Appends a message to a JSON log file (log/owner_log.json), automatically including
    the invoking tool name and username from the Atlantis context.

    Args:
        message: The string message to log.
    """
    invoking_tool_name = get_invoking_tool_name() or "unknown_tool"
    username = get_caller() or "unknown_user"

    # The log directory is relative to the server's execution path.
    log_dir = "log"
    log_file_path = os.path.join(log_dir, "owner_log.json")

    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    except Exception as e:
        # Using print as a fallback for logging since this is a logging function
        logger.warning(f"Could not create directory {log_dir}. Error: {e}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": str(invoking_tool_name),
        "caller": str(username),
        "message": str(message)
    }

    entries = []
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    loaded_json = json.loads(content)
                    if isinstance(loaded_json, list):
                        entries = loaded_json
                    else:
                        logger.warning(f"Log file {log_file_path} did not contain a JSON list. Re-initializing.")
                        entries = []
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {log_file_path}. Re-initializing log.")
            entries = []
        except Exception as e:
            logger.error(f"Error reading log file {log_file_path}: {e}. Re-initializing log.")
            entries = []

    entries.append(log_entry)

    # Echo to console so owner_log is visible in server output
    logger.info(f"📋 OWNER_LOG [{invoking_tool_name}] ({username}): {message}")

    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=4)
        return f"Message logged to {log_file_path}"
    except Exception as e:
        logger.error(f"Error writing to log file {log_file_path}: {e}")
        return f"Error writing to log file {log_file_path}: {e}"

    return False

async def client_onclick(key: str, callback: Callable):
    """Registers an onclick handler for a click key.

    Args:
        key: The unique key to identify this click handler
        callback: The async function to call when the click event occurs
    """
    # Store the callback in the global lookup table
    _click_callbacks[key] = callback
    #await client_log(f"DEBUG: Stored callback for key '{key}', total callbacks: {len(_click_callbacks)}")

    # Send registration message to client via awaitable client_command
    await client_command("onclick_register", key, message_type="onclick_register")

async def client_upload(key: str, callback: Callable):
    """Registers an upload handler

    Args:
        key: The unique key to identify this upload handler
        callback: The async function to call when upload occurs
    """
    # Store the callback in the global lookup table
    _upload_callbacks[key] = callback

    # Send registration message to client via awaitable client_command
    await client_command("upload_register", key, message_type="upload_register")


# possibly obsolete
async def invoke_click_callback(key: str) -> Any:
    """Invokes a stored click callback by its key.

    Args:
        key: The key of the callback to invoke

    Returns:
        The result of the callback function, or None if key not found
    """
    logger = logging.getLogger("mcp_server")
    logger.info(f"DEBUG: Looking up callback for key '{key}', available keys: {list(_click_callbacks.keys())}")

    callback = _click_callbacks.get(key)
    if callback:
        logger.info(f"DEBUG: Found callback for key '{key}', executing...")
        # Callback runs in the current context, which should have client_log available
        if inspect.iscoroutinefunction(callback):
            result = await callback()
            if isinstance(result, (dict, list)):
                logger.info(f"DEBUG: Callback executed, result:\n{format_json_log(result, colored=True)}")
            else:
                logger.info(f"DEBUG: Callback executed, result: {result}")
            return result
        else:
            result = callback()
            if isinstance(result, (dict, list)):
                logger.info(f"DEBUG: Callback executed, result:\n{format_json_log(result, colored=True)}")
            else:
                logger.info(f"DEBUG: Callback executed, result: {result}")
            return result
    else:
        logger.info(f"DEBUG: No callback found for key '{key}'")
    return None

# possibly obsolete
async def invoke_click_callback_with_context(key: str, bound_client_log) -> Any:
    """Invokes a stored click callback with a specific client_log context.

    Args:
        key: The key of the callback to invoke
        bound_client_log: A bound client_log function with proper context

    Returns:
        The result of the callback function, or None if key not found
    """
    logger = logging.getLogger("mcp_server")
    logger.info(f"DEBUG: Looking up callback for key '{key}' with context")

    callback = _click_callbacks.get(key)
    if callback:
        logger.info(f"DEBUG: Found callback for key '{key}', executing with context...")

        # Temporarily replace the client_log implementation for this callback
        original_client_log = globals().get('client_log')

        async def context_client_log(message, level="INFO", message_type="text"):
            # Use the bound client_log directly instead of the context-dependent one
            await bound_client_log(message, level=level, message_type=message_type,
                                 caller_name="callback", entry_point_name="click_callback")

        # Replace client_log temporarily
        globals()['client_log'] = context_client_log

        try:
            if inspect.iscoroutinefunction(callback):
                result = await callback()
            else:
                result = callback()
            if isinstance(result, (dict, list)):
                logger.info(f"DEBUG: Callback executed with context, result:\n{format_json_log(result, colored=True)}")
            else:
                logger.info(f"DEBUG: Callback executed with context, result: {result}")
            return result
        finally:
            # Restore original client_log
            if original_client_log:
                globals()['client_log'] = original_client_log
    else:
        logger.info(f"DEBUG: No callback found for key '{key}'")
    return None
