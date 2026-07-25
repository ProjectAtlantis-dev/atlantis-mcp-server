"""Read-only access to selected Atlantis runtime context."""

import inspect
from typing import Any, Callable, Dict

import atlantis as _atlantis


_ALLOWED_METHODS: Dict[str, Callable[[], Any]] = {
    "get_caller": _atlantis.get_caller,
    "get_caller_shell_path": _atlantis.get_caller_shell_path,
    "get_default_owner": _atlantis.get_default_owner,
    "get_exec_shell_path": _atlantis.get_exec_shell_path,
    "get_request_id": _atlantis.get_request_id,
    "get_script_folder": _atlantis.get_script_folder,
    "get_session_key": _atlantis.get_session_key,
    "get_terminal_key": _atlantis.get_terminal_key,
    "get_user_game_id": _atlantis.get_user_game_id,
}


@public
@visible
async def atlantis(method: str = "") -> Any:
    """Call an allowlisted read-only Atlantis runtime getter.

    Example: @atlantis get_user_game_id
    """
    method = str(method or "").strip()
    if not method:
        return {
            "usage": "@atlantis <method>",
            "methods": sorted(_ALLOWED_METHODS),
        }
    if method.endswith("()"):
        method = method[:-2].strip()
    if method.startswith("atlantis."):
        method = method[len("atlantis.") :].strip()

    func = _ALLOWED_METHODS.get(method)
    if func is None:
        raise ValueError(
            f"Unsupported Atlantis runtime method: {method!r}. "
            f"Allowed methods: {', '.join(sorted(_ALLOWED_METHODS))}"
        )

    result = func()
    if inspect.isawaitable(result):
        return await result
    return result
