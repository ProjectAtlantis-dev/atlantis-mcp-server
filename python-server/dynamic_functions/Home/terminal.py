"""Terminals — one user-facing shell, placed at one location.

A *terminal* is identified by `(session_key, shell_path)`. A user can have
several shells open in the same session; each is its own terminal and can be
parked at a different location. A terminal is just a viewing point — chat
identity lives on the session's `chat_slot`, not on the terminal.

In-memory only. Terminals die with the shell they belong to, which is the
right lifetime: there's no useful "reconnect to my old camera" story when the
shell process itself is gone.
"""

import atlantis
from typing import Any, Dict, List, Optional, Tuple


# (session_key, shell_path) -> { "location": str }
_terminals: Dict[Tuple[str, str], Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def _current_key() -> Optional[Tuple[str, str]]:
    sk = atlantis.get_session_key()
    shell = atlantis.get_exec_shell_path() or atlantis.get_caller_shell_path()
    if not sk or not shell:
        return None
    return (sk, shell)


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def get_terminal_location(session_key: str = "", shell_path: str = "") -> Optional[str]:
    """Where is this terminal currently placed? Defaults to the calling terminal."""
    if not session_key or not shell_path:
        key = _current_key()
        if key is None:
            return None
    else:
        key = (session_key, shell_path)
    rec = _terminals.get(key)
    return rec.get("location") if rec else None


def set_terminal_location(location: str) -> None:
    """Place the calling terminal at `location`. No-op if there's no terminal context."""
    key = _current_key()
    if key is None:
        return
    _terminals.setdefault(key, {})["location"] = location


def list_terminals() -> List[Dict[str, Any]]:
    """Snapshot of every live terminal."""
    rows: List[Dict[str, Any]] = []
    for (sk, shell), rec in _terminals.items():
        parts = sk.split(":", 1)
        rows.append({
            "session_key": sk,
            "user_game_id": parts[0] if len(parts) > 0 else "",
            "caller_sid": parts[1] if len(parts) > 1 else "",
            "shell_path": shell,
            "location": rec.get("location", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@visible
async def terminal_move(game_key: str, location: str = "") -> Dict[str, Any]:
    """Place this terminal at `location`. If `location` is empty, the terminal
    is moved to the calling user's current chat_slot position (the most common
    'put me where my character is' case)."""
    from dynamic_functions.Home.common import require_game_dir
    from dynamic_functions.Home.location import _load_location, position_get, _set_location_background
    from dynamic_functions.Home.session import require_session

    require_game_dir(game_key)

    if not location:
        s = require_session()
        chat_slot = s.get("chat_slot")
        if not chat_slot:
            raise ValueError("No location given and this session has no chat_slot.")
        location = position_get(game_key, chat_slot) or ""
        if not location:
            raise ValueError(f"chat_slot '{chat_slot}' has no position yet.")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")

    set_terminal_location(location)
    await _set_location_background(location, dest)
    return {"location": location, **(_current_key_dict())}


@visible
async def terminal_show() -> Dict[str, Any]:
    """Show this terminal's identity and current location."""
    info = _current_key_dict()
    info["location"] = get_terminal_location() or ""
    await atlantis.client_data("Terminal", [info])
    return info


@visible
async def terminal_list() -> List[Dict[str, Any]]:
    """List all live terminals across all sessions."""
    rows = list_terminals()
    await atlantis.client_data("Terminals", rows)
    return rows


def _current_key_dict() -> Dict[str, Any]:
    sk = atlantis.get_session_key() or ""
    parts = sk.split(":", 1)
    return {
        "session_key": sk,
        "user_game_id": parts[0] if len(parts) > 0 else "",
        "caller_sid": parts[1] if len(parts) > 1 else "",
        "shell_path": atlantis.get_exec_shell_path() or atlantis.get_caller_shell_path() or "",
    }
