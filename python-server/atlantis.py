import contextvars
import inspect # Ensure inspect is imported
import asyncio # Added for Lock
import ast
from typing import Callable, Optional, Any, List # Added List
from call_context import CallContext, ToolCallPayload
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
_current_context: contextvars.ContextVar[Optional[CallContext]] = contextvars.ContextVar("current_call_context", default=None)

# Default owner of the remote server instance
_default_owner: Optional[str] = ""
_owner_usernames: List[str] = []  # List of owner usernames for permission checks
_remote_name: Optional[str] = None

# Simple task collection for logging tasks
_current_log_tasks: contextvars.ContextVar[Optional[List[asyncio.Task]]] = contextvars.ContextVar("current_log_tasks", default=None)

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



# --- Shared Object Containers ---
# Two containers that persist across dynamic function reloads:
#   server_shared — global, server-wide (DB connections, busy tracking, etc.)
#   session_shared — auto-scoped by session ID, isolated per user session

class _SharedContainer:
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


class _SessionSharedContainer:
    """Session-scoped shared container. Keys are automatically namespaced by session ID
    so dynamic functions cannot access another user's session data."""
    def __init__(self, backing: _SharedContainer):
        self._backing = backing

    def _scoped_key(self, key):
        session_key = get_session_key()
        if not session_key:
            raise RuntimeError("session_shared requires a session context (no session_key set)")
        return f"__session:{session_key}:{key}"
    def get(self, key, default=None):
        """Get a value scoped to the current session"""
        return self._backing.get(self._scoped_key(key), default)

    def set(self, key, value):
        """Store a value scoped to the current session"""
        return self._backing.set(self._scoped_key(key), value)

    def remove(self, key):
        """Remove a value scoped to the current session"""
        return self._backing.remove(self._scoped_key(key))

    def keys(self):
        """Get keys belonging to the current session only"""
        prefix = self._scoped_key("")
        return [k[len(prefix):] for k in self._backing.keys() if k.startswith(prefix)]


# Initialize shared containers
server_shared = _SharedContainer()
session_shared = _SessionSharedContainer(server_shared)

# --- Helper Functions ---

def get_context() -> Optional[CallContext]:
    """Return the current CallContext, if one is active."""
    return _current_context.get()

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
    request_id = get_request_id()
    if request_id is None:
        logger.error(f"{context_name} - request_id is None. Cannot get sequence number.")
        return -1

    exec_shell_path = get_exec_shell_path()
    # Use composite key of (request_id, exec_shell_path) for per-shell sequencing
    counter_key = (request_id, exec_shell_path)

    if counter_key not in _request_seq_counters:
        _request_seq_counters[counter_key] = 1

    current_seq = _request_seq_counters[counter_key]
    _request_seq_counters[counter_key] += 1
    return current_seq

# --- Accessor Functions ---

async def client_log(message: Any, level: str = "INFO", message_type: str = "text", is_private: bool = True, location: Optional[str] = None):
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
                   If False, pass a cloud-side routing hint.

    Calls the underlying log function directly; async dispatch is handled internally.
    """
    ctx = get_context()
    log_func = ctx.client_log_func if ctx else None
    if log_func:
        caller_name = "unknown_caller" # Default for immediate caller
        entry_point_name = get_entry_point_name() or "unknown_entry_point" # Get entry point from context

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
            client_id = _get_client_id()
            request_id = get_request_id()

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
                is_private=is_private,
                visibility_scope="sid" if is_private else "game",
                location=location,
                caller_sid=get_caller()
            )
            # task is the asyncio.Task returned by utils.client_log

            # Track the task if we have one
            if task is not None:
                tasks = _current_log_tasks.get()
                if tasks is None:
                    # Initialize task list if not already done
                    tasks = []
                    _current_log_tasks.set(tasks)
                tasks.append(task)

            return None # Return None in either case
        except Exception as e:
            logger.error(f"Failed during async client_log call (after inspect): {e}")
            # Decide if to re-raise or return an error indicator
            raise
    else:
        logger.warning(f"client_log called but no logger in context. Message: {message}")
        return None # Or raise an error

async def client_description(message: Any, level: str = "INFO", is_private: bool = True, location: Optional[str] = None):
    """Sends a description message back to the requesting client for the current context."""
    return await client_log(
        message,
        level=level,
        message_type="description",
        is_private=is_private,
        location=location
    )

async def client_warning(message: Any, level: str = "WARNING", is_private: bool = True, location: Optional[str] = None):
    """Sends a warning message back to the requesting client for the current context."""
    return await client_log(
        message,
        level=level,
        message_type="warning",
        is_private=is_private,
        location=location
    )

async def tool_result(name: str, result: Any):
    """Sends a tool call result back to the requesting client to be added to the transcript.
    This allows the LLM to see tool results in the next conversation turn.
    """
    return await client_command("tool", result, message_type="text")

# --- Other Accessors ---
# this is established by the tool caller
def get_request_id() -> Optional[str]:
    """Returns the request_id"""
    ctx = get_context()
    return ctx.request_id if ctx else None

def _get_client_id() -> Optional[str]:
    """Returns the client_id (internal plumbing for routing messages)."""
    ctx = get_context()
    return ctx.client_id if ctx else None

def get_entry_point_name() -> Optional[str]:
    """Returns the entry point function for the current call."""
    ctx = get_context()
    return ctx.entry_point_name if ctx else None

def get_script_folder() -> Optional[str]:
    """Return the executing dynamic function's stable /username/remote/folder path.

    Nested folders use slash notation. Root-level scripts return
    "/username/remote". Returns None when there is no active dynamic-function
    context.
    """
    ctx = get_context()
    return ctx.script_folder if ctx else None

def get_script_name() -> Optional[str]:
    """Return the executing dynamic function's filename, including its extension."""
    ctx = get_context()
    return ctx.script_name if ctx else None

# locally-derived stable session identifier — see CallContext.session_key
def get_session_key() -> Optional[str]:
    """Returns the session_key for this function call (None if any component missing)."""
    ctx = get_context()
    return ctx.get_session_key() if ctx else None

# session key narrowed to the originating terminal — see CallContext.terminal_key
def get_terminal_key() -> Optional[str]:
    """Returns the terminal_key for this function call (None if any component missing).

    A terminal is one of a human's terminals within a session: session_key plus
    caller_shell_path (the user's root shell)."""
    ctx = get_context()
    return ctx.get_terminal_key() if ctx else None

def get_caller() -> Optional[str]:
    """Returns the caller sid who called this function."""
    ctx = get_context()
    return ctx.caller_sid if ctx else None

def get_exec_shell_path() -> Optional[str]:
    """Returns the exec shell path - the shell where this call's work runs.
    Outbound client_command callbacks are tagged with this. NOT the user's
    root shell (that's ctx.caller_shell_path, used only for attribution)."""
    ctx = get_context()
    if not ctx:
        return None
    return ctx.exec_shell_path or ctx.caller_shell_path

def get_caller_shell_path() -> Optional[str]:
    ctx = get_context()
    return ctx.caller_shell_path if ctx else None

def get_user_game_id() -> Optional[int]:
    """Returns the user_game_id for this function call."""
    ctx = get_context()
    return ctx.user_game_id if ctx else None

def set_exec_shell_path(path: Optional[str]) -> None:
    """Set the exec shell path contextvar directly (e.g. for lobster socket tasks,
    where the lobster's single shell IS the exec shell)."""
    ctx = get_context()
    if ctx:
        _current_context.set(ctx.with_payload_updates(exec_shell_path=path))
    else:
        _current_context.set(CallContext(payload=ToolCallPayload(exec_shell_path=path)))

def get_owner_usernames() -> List[str]:
    """Returns the list of owner usernames for permission checks"""
    return _owner_usernames

def get_default_owner() -> Optional[str]:
    """Returns the default owner username for this remote server instance."""
    return _default_owner

def _get_remote_name() -> Optional[str]:
    """Returns this remote server instance's configured service name."""
    return _remote_name

def is_owner(username: str) -> bool:
    """Check if the given username is an owner"""
    return username in _owner_usernames

# --- Setter Functions (primarily for internal use by dynamic_function_manager) ---

def _set_default_owner(new_owner: Optional[str]):
    """Sets the default owner username. For internal use."""
    global _default_owner
    _default_owner = new_owner

def _set_remote_name(new_remote_name: Optional[str]):
    """Sets this remote server instance's configured service name. For internal use."""
    global _remote_name
    _remote_name = new_remote_name

def _set_owner_usernames(usernames: List[str]):
    """Sets the list of owner usernames. For internal use."""
    global _owner_usernames
    _owner_usernames = usernames


def set_context(ctx: CallContext) -> None:
    """Sets the current CallContext."""
    _current_context.set(ctx)


def reset_context() -> None:
    """Clears the current CallContext."""
    _current_context.set(None)


# --- Utility Functions ---

def get_uncalled_dynamic_functions(functions_dir: Optional[str] = None) -> List[dict]:
    """Return dynamic functions with no static call sites.

    The result is a list of {"filename": "relative/path.py", "function": "qualname"} objects.
    This is a static AST name match over the dynamic_functions tree; Atlantis
    tool routing, decorators, getattr/registry calls, and browser callbacks may
    still invoke functions that appear here.
    """
    root = os.path.realpath(
        functions_dir
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_functions")
    )
    skip_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules"}
    definitions = []
    calls_by_name = {}

    class _CallVisitor(ast.NodeVisitor):
        def __init__(self, file_path: str):
            self.file_path = file_path
            self.function_stack = []
            self.scope_stack = []

        def visit_ClassDef(self, node):
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_FunctionDef(self, node):
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node):
            self._visit_function(node)

        def _visit_function(self, node):
            qualname = ".".join(self.scope_stack + [node.name])
            definitions.append((self.file_path, node.lineno, node.name, qualname))
            self.scope_stack.append(node.name)
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()
            self.scope_stack.pop()

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = None

            if name:
                enclosing = ".".join(self.scope_stack) if self.scope_stack else None
                calls_by_name.setdefault(name, []).append((self.file_path, enclosing))
            self.generic_visit(node)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=file_path)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            _CallVisitor(file_path).visit(tree)

    uncalled = []
    for file_path, _line, name, qualname in definitions:
        called_elsewhere = any(
            not (call_file == file_path and enclosing == qualname)
            for call_file, enclosing in calls_by_name.get(name, [])
        )
        if not called_elsewhere:
            rel_path = os.path.relpath(file_path, root)
            uncalled.append({"filename": rel_path, "function": qualname})

    return sorted(uncalled, key=lambda item: (item["filename"], item["function"]))

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
    actual_client_id = _get_client_id()
    request_id = get_request_id()
    entry_point_name = get_entry_point_name() or "unknown_entry_point"

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

    caller_sid = get_caller()

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
                logger_name=caller_name,
                caller_sid=caller_sid
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
                stream_id=stream_id_to_send, # Pass the generated stream_id separately
                caller_sid=caller_sid
            )
        return stream_id_to_send # Return the generated stream_id to the caller
    except Exception as e:
        logger.error(f"Failed during async stream_start call: {e}")
        raise

async def stream(message: str, stream_id_param: str):
    """Sends a stream message snippet back to the client using a provided stream_id.
    """
    actual_client_id = _get_client_id()
    request_id = get_request_id()
    entry_point_name = get_entry_point_name() or "unknown_entry_point"

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

    caller_sid = get_caller()

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
                logger_name=caller_name,
                caller_sid=caller_sid
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
                stream_id=stream_id_param, # Pass the provided stream_id separately
                caller_sid=caller_sid
            )
        return result
    except Exception as e:
        logger.error(f"Failed during async stream call: {e}")
        raise

async def stream_end(stream_id_param: str):
    """Sends a stream_end message to the client, indicating the end of a stream, using a provided stream_id.
    """
    actual_client_id = _get_client_id()
    request_id = get_request_id()
    entry_point_name = get_entry_point_name() or "unknown_entry_point"

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

    caller_sid = get_caller()

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
                logger_name=caller_name,
                caller_sid=caller_sid
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
                stream_id=stream_id_param, # Pass the provided stream_id separately
                caller_sid=caller_sid
            )
        return result
    except Exception as e:
        logger.error(f"Failed during async stream_end call: {e}")
        raise


async def _client_command(
    command: str,
    data: Any = None,
    message_type: str = "command",
    is_private: bool = True,
    notification_params: Optional[dict[str, Any]] = None
) -> Any:
    """Sends a command message to the client and waits for a specific acknowledgment and result.

    This function is for commands that require the server to wait for completion
    or a specific response from the client before proceeding.

    Args:
        command: The command string identifier.
        data: Optional JSON-serializable data associated with the command.
        message_type: The message type for the protocol (default "command").
        is_private: If True, send only to requesting client.
        notification_params: Internal-only params flattened beside messageType/data.

    Returns:
        The result returned by the client for the command.

    Raises:
        RuntimeError: If context variables (client_id, request_id) are not set.
        McpError: Propagated from underlying calls if timeouts or client-side errors occur.
    """
    # Get necessary context for routing and correlation
    client_id = _get_client_id()
    request_id = get_request_id()
    entry_point_name = get_entry_point_name()
    caller_sid = get_caller()
    session_key = get_session_key()
    exec_shell_path = get_exec_shell_path()

    logger.info(f"📡 client_command '{command}' (entry={entry_point_name}, caller_sid={caller_sid})")
    if isinstance(data, (dict, list)):
        logger.debug(f"   📦 data: {format_json_log(data, colored=True)}")
    elif data is not None:
        log_data = "[base64 image]" if isinstance(data, str) and data.startswith("data:image/") else data
        logger.debug(f"   📦 data: {log_data}")

    if not client_id or not request_id:
        # This should ideally not happen if called within a proper request context
        logger.error(f"client_command called without client_id or request_id in context. Command: {command}")
        raise RuntimeError("Client ID or Request ID not found in context for client_command.")

    try:
        # Get current sequence number and increment it for the next call
        # Using the helper function for consistent sequence number management
        current_seq_to_send = await get_and_increment_seq_num(context_name="client_command")

        # Extra logging for tool calls (commands starting with %)
        if command.startswith('%'):
            logger.info(f"🔧 TOOL CALL seq={current_seq_to_send}: {command}")
            if isinstance(data, dict):
                logger.info(f"🔧 TOOL DATA: {format_json_log(data)}")

        logger.info(f"📡 Sending awaitable '{command}' seq={current_seq_to_send}")
        result = await execute_client_command_awaitable(
            client_id_for_routing=client_id,
            request_id=request_id,
            command=command,
            command_data=data,
            seq_num=current_seq_to_send,  # Pass the sequence number
            entry_point_name=entry_point_name,  # Pass the entry point name for logging
            caller_sid=caller_sid,
            session_key=session_key,
            shell_path=exec_shell_path,  # Wire kwarg name is legacy; semantically this is exec_shell_path
            message_type=message_type,  # Pass message_type for the protocol
            is_private=is_private,  # Pass is_private for broadcast control
            message_params=notification_params
        )
        logger.debug(f"📡 Result for '{command}': type={type(result).__name__}")

        return result
    except Exception as e:
        logger.warning(f"❌ client_command '{command}' FAILED: {type(e).__name__}: {e}")
        # Server layer already logged with enhanced error message including command context
        # Just re-raise to let the dynamic function manager handle final logging
        raise


async def client_command(command: str, data: Any = None, message_type: str = "command", is_private: bool = True) -> Any:
    """Sends a command message to the client and waits for acknowledgment."""
    return await _client_command(command, data=data, message_type=message_type, is_private=is_private)


async def client_html(content: str, modal: bool = False, title: Optional[str] = None):
    """Sends HTML content back to the requesting client for rendering

    Args:
        content: The HTML content to send
        modal: If True, render the HTML in a client modal.
        title: Optional modal title.
    """
    # Internal carrier only: these keys are flattened into notifications/message.params
    # beside messageType and data; no wrapper is exposed or sent to clients.
    notification_params: dict[str, Any] = {}
    if modal:
        notification_params["modal"] = True
    if title is not None:
        notification_params["title"] = title

    # Use client_command for awaitable response with proper message_type
    result = await _client_command("html", content, message_type="html", notification_params=notification_params or None)
    if modal:
        if not isinstance(result, dict) or not result.get("modalId"):
            raise RuntimeError(f"Expected modal html ack with modalId, got: {result!r}")
    return result


async def client_modal(content: str, title: Optional[str] = None) -> str:
    """Sends HTML content back to the requesting client in a modal.

    Returns:
        The modal UUID returned by the client ack.
    """
    notification_params: dict[str, Any] = {
        "modal": True,
    }
    if title is not None:
        notification_params["title"] = title

    result = await _client_command("html", content, message_type="html", notification_params=notification_params)
    if not isinstance(result, dict) or not result.get("modalId"):
        raise RuntimeError(f"Expected modal html ack with modalId, got: {result!r}")
    return result["modalId"]


async def client_modal_close(modal_id: str) -> Any:
    """Closes a client modal by ID."""
    if not modal_id:
        raise ValueError("modal_id is required")

    logger.info(f"close modal panel modalId={modal_id}")
    notification_params: dict[str, Any] = {
        "modal": True,
        "action": "close",
        "modalId": modal_id,
    }
    return await _client_command("html", None, message_type="html", notification_params=notification_params)


async def client_script(content: str, is_private: bool = True):
    """Sends Javascript content back to the requesting client for rendering

    Args:
        content: The Javascript content to send
        is_private: If True (default), script only runs on the requesting client.
                   If False, pass a cloud-side routing hint.
    """
    notification_params = {
        "visibilityScope": "sid" if is_private else "game",
    }

    # Use client_command for awaitable response with proper message_type
    result = await _client_command(
        "script",
        content,
        message_type="script",
        is_private=is_private,
        notification_params=notification_params,
    )
    return result

async def client_terminal_script(content: str, is_private: bool = True):
    """Sends Javascript that re-runs on every render (a 'terminal' event).

    Unlike client_script, which is one-shot (the client skips it once its event
    has a completed_at), terminal events are not deduped, so the script re-applies
    after page reloads. Use this for cosmetic DOM effects that must survive a
    reload — frosted styling, terminal-mode chrome, etc.

    Args:
        content: The Javascript content to send.
        is_private: If True (default), script only runs on the requesting client.
    """
    # Same "script" command, but message_type "terminal" routes it to the
    # client's re-runnable terminal-event path instead of the one-shot script path.
    result = await _client_command(
        "script",
        content,
        message_type="terminal",
        is_private=is_private,
        notification_params={"visibilityScope": "shell"},
    )
    return result

async def set_background(image_path: str, image_format: Optional[str] = None, vertical_align: str = "bottom"):
    """Sets the background image for the client UI.

    Args:
        image_path: Path to the image file
        image_format: Optional MIME type of the image (e.g., "image/png", "image/jpeg").
                     If not provided, will be auto-detected from file extension.
        vertical_align: Vertical alignment for background-position ("top", "center", "bottom",
                       or a CSS length/percentage). Defaults to "bottom".
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

    result = await _client_command(
        "background",
        prefixed_data,
        message_type="background",
        notification_params={
            "verticalAlign": vertical_align,
            "visibilityScope": "game",
        },
    )
    return result

async def set_background_video(
    video_source: str,
    video_format: Optional[str] = None,
    vertical_align: str = "center",
    playback_rate: Optional[float] = None,
    brightness: Optional[float] = None,
    loop: bool = False,
    muted: bool = True,
    autoplay: bool = True,
    plays_inline: bool = True,
    remove_on_ended: bool = True,
    toggle_audio: bool = True,
    replay: bool = False,
):
    """Sets a background video for the client UI.

    Args:
        video_source: URL, data URL, or local path to a video file.
        video_format: Optional MIME type for local files (e.g., "video/mp4", "video/webm").
                     If not provided, will be auto-detected from file extension.
        vertical_align: Vertical alignment for object-position ("top", "center", "bottom",
                       or a CSS length/percentage). Defaults to "center".
        playback_rate: Optional playback speed multiplier. Use values below 1.0 to slow
                       the video down, e.g. 0.5 for half speed.
        brightness: Optional brightness multiplier. Use values below 1.0 to darken
                    the video, e.g. 0.6 for 60% brightness.
        loop: Whether the video should loop. Defaults to False.
        muted: Whether the video starts muted. Defaults to True for browser autoplay.
        autoplay: Whether the video should autoplay. Defaults to True.
        plays_inline: Whether mobile browsers should play inline. Defaults to True.
        remove_on_ended: Whether to remove the video when playback ends. Defaults to True.
        toggle_audio: Whether clicking the background toggles mute. Defaults to True.
        replay: Whether the completed background video should replay on refresh. Defaults to False.
    """
    if video_source.startswith(("http://", "https://", "data:")):
        video_data = video_source
    else:
        if video_format is None:
            mime_type, _ = mimetypes.guess_type(video_source)
            if not mime_type or not mime_type.startswith("video/"):
                video_format = "video/mp4"
            else:
                video_format = mime_type
        base64_data = video_to_base64(video_source)
        video_data = f"data:{video_format};base64,{base64_data}"

    result = await _client_command(
        "background_video",
        video_data,
        message_type="background_video",
        notification_params={
            "verticalAlign": vertical_align,
            "playbackRate": playback_rate,
            "brightness": brightness,
            "loop": loop,
            "muted": muted,
            "autoplay": autoplay,
            "playsInline": plays_inline,
            "removeOnEnded": remove_on_ended,
            "toggleAudio": toggle_audio,
            "replay": replay,
            "visibilityScope": "game",
        },
    )
    return result

async def set_background_player(
    video_source: str,
    video_format: Optional[str] = None,
    vertical_align: str = "center",
    playback_rate: Optional[float] = None,
    brightness: Optional[float] = None,
    loop: bool = False,
    muted: bool = True,
    autoplay: bool = True,
    plays_inline: bool = True,
    remove_on_ended: bool = True,
    controls: bool = False,
    replay: bool = False,
):
    """Sets a controllable background video player for the client UI.

    Args:
        video_source: URL, data URL, or local path to a video file.
        video_format: Optional MIME type for local files (e.g., "video/mp4", "video/webm").
                     If not provided, will be auto-detected from file extension.
        vertical_align: Vertical alignment for object-position ("top", "center", "bottom",
                       or a CSS length/percentage). Defaults to "center".
        playback_rate: Optional playback speed multiplier. Use values below 1.0 to slow
                       the video down, e.g. 0.5 for half speed.
        brightness: Optional brightness multiplier. Use values below 1.0 to darken
                    the video, e.g. 0.6 for 60% brightness.
        loop: Whether the video should loop. Defaults to False.
        muted: Whether the player starts muted. Defaults to True.
        autoplay: Whether the player should autoplay. Defaults to True.
        plays_inline: Whether mobile browsers should play inline. Defaults to True.
        remove_on_ended: Whether to remove the player when playback ends. Defaults to True.
        controls: Whether browser video controls are shown. Defaults to False.
        replay: Whether the completed background player should replay on refresh. Defaults to False.
    """
    if video_source.startswith(("http://", "https://", "data:")):
        video_data = video_source
    else:
        if video_format is None:
            mime_type, _ = mimetypes.guess_type(video_source)
            if not mime_type or not mime_type.startswith("video/"):
                video_format = "video/mp4"
            else:
                video_format = mime_type
        base64_data = video_to_base64(video_source)
        video_data = f"data:{video_format};base64,{base64_data}"

    result = await _client_command(
        "background_player",
        video_data,
        message_type="background_player",
        notification_params={
            "verticalAlign": vertical_align,
            "playbackRate": playback_rate,
            "brightness": brightness,
            "loop": loop,
            "muted": muted,
            "autoplay": autoplay,
            "playsInline": plays_inline,
            "removeOnEnded": remove_on_ended,
            "controls": controls,
            "replay": replay,
            "visibilityScope": "game",
        },
    )
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
    tasks = _current_log_tasks.get()
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
    invoking_tool_name = get_entry_point_name() or "unknown_tool"
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
    if not callback:
        logger.info(f"DEBUG: No callback found for key '{key}'")
        return None

    logger.info(f"DEBUG: Found callback for key '{key}', executing with context...")
    current_ctx = get_context()
    context_token = None
    if current_ctx:
        context_token = _current_context.set(current_ctx.model_copy(update={
            "client_log_func": bound_client_log,
            "entry_point_name": "click_callback",
        }))
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
        if context_token is not None:
            _current_context.reset(context_token)
