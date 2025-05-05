import contextvars
from typing import Callable, Optional

# --- Context Variables ---
# Define context variables with default values (None or a placeholder)
# The actual values will be set per-request.

# client_log: A callable that sends logs back to the requesting client
_client_log_var: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar("_client_log_var", default=None)

# request_id: The unique ID for the current MCP request
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_request_id_var", default=None)

# client_id: The identifier for the client making the request
_client_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_client_id_var", default=None)

# --- Accessor Functions ---
# Functions that dynamic code can import and call to get the current context.

def client_log(message: str, level: str = "INFO"):
    """Sends a log message back to the requesting client for the current context.
    Calls the underlying log function directly; async dispatch is handled internally.
    """
    log_func = _client_log_var.get()
    if log_func:
        try:
            # Call the function directly. It's synchronous but handles
            # its own async task creation internally (in utils.client_log).
            log_func(message) # Or log_func(message, level=level) if needed
        except Exception as e:
            # Catch potential errors in the synchronous part of log_func
            # or if log_func itself raises an unexpected error.
            print(f"ERROR: Failed during client_log call: {e}")
    else:
        print(f"WARNING: client_log called but no logger in context. Message: {message}")

def get_request_id() -> Optional[str]:
    """Gets the request ID for the current context."""
    return _request_id_var.get()

def get_client_id() -> Optional[str]:
    """Gets the client ID for the current context."""
    return _client_id_var.get()

# --- Setter Functions (primarily for internal use by dynamic_manager) ---

def set_context(client_log_func: Callable, request_id: str, client_id: str):
    """Sets the context variables for the current execution scope.
    Returns tokens to be used for resetting.
    """
    # In Python 3.7+, ContextVar.set() returns a Token for resetting
    client_log_token = _client_log_var.set(client_log_func)
    request_id_token = _request_id_var.set(request_id)
    client_id_token = _client_id_var.set(client_id)
    return (client_log_token, request_id_token, client_id_token)

def reset_context(tokens):
    """Resets the context variables using the provided tokens."""
    client_log_token, request_id_token, client_id_token = tokens
    _client_log_var.reset(client_log_token)
    _request_id_var.reset(request_id_token)
    _client_id_var.reset(client_id_token)
