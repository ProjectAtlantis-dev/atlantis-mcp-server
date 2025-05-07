import contextvars
import inspect # Ensure inspect is imported
from typing import Callable, Optional

# --- Context Variables ---
# client_log_func: Holds the partially bound client_log function for the current request
_client_log_var: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar("_client_log_var", default=None)
# request_id: The unique ID of the current MCP request
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_request_id_var", default=None)
# client_id: The ID of the client that initiated the current request
_client_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_client_id_var", default=None)
# log_seq_num: A counter for log messages within the current request context
_log_seq_num_var: contextvars.ContextVar[int] = contextvars.ContextVar("_log_seq_num_var", default=0)

# entry_point_name: The name of the top-level dynamic function called by the request
_entry_point_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_entry_point_name_var", default=None)

# --- Accessor Functions ---

def client_log(message: str, level: str = "INFO"):
    """Sends a log message back to the requesting client for the current context.
    Includes a sequence number and automatically determines the calling function name.
    Also includes the original entry point function name.
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
            print(f"WARNING: Could not inspect caller frame for client_log: {inspect_err}")

        try:
            # Get current sequence number and increment it for the next call
            current_seq = _log_seq_num_var.get()
            _log_seq_num_var.set(current_seq + 1)

            # Call the underlying function, passing seq num, caller name, AND entry point name.
            # NOTE: log_func (utils.client_log) needs update to accept entry_point_name.
            log_func(
                message,
                level=level,
                seq_num=current_seq,
                logger_name=caller_name, # The immediate caller
                entry_point_name=entry_point_name # The original function called
            )
        except Exception as e:
            print(f"ERROR: Failed during client_log call (after inspect): {e}")
    else:
        print(f"WARNING: client_log called but no logger in context. Message: {message}")

# --- Other Accessors ---
def get_request_id() -> Optional[str]:
    """Returns the request_id for the current context."""
    return _request_id_var.get()

def get_client_id() -> Optional[str]:
    """Returns the client_id for the current context."""
    return _client_id_var.get()

# --- Setter Functions (primarily for internal use by dynamic_function_manager) ---

def set_context(client_log_func: Callable, request_id: str, client_id: str, entry_point_name: str):
    """Sets the context variables for the current execution scope.
    Returns tokens to be used for resetting.
    """
    client_log_token = _client_log_var.set(client_log_func)
    request_id_token = _request_id_var.set(request_id)
    client_id_token = _client_id_var.set(client_id)
    log_seq_num_token = _log_seq_num_var.set(0)
    # Set the entry point name context
    entry_point_token = _entry_point_name_var.set(entry_point_name)
    return (client_log_token, request_id_token, client_id_token, log_seq_num_token, entry_point_token)

def reset_context(tokens):
    """Resets the context variables using the provided tokens."""
    client_log_token, request_id_token, client_id_token, log_seq_num_token, entry_point_token = tokens
    _client_log_var.reset(client_log_token)
    _request_id_var.reset(request_id_token)
    _client_id_var.reset(client_id_token)
    _log_seq_num_var.reset(log_seq_num_token)
    # Reset the entry point name context
    _entry_point_name_var.reset(entry_point_token)
