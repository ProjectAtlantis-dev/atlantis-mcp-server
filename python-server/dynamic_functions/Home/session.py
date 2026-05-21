"""Per-session state, keyed by atlantis.get_session_key() = f"{user_game_id}:{caller_sid}".

One binding per session:
  - chat_slot:   sid → whose mouth my typed text comes out of

Viewing location ("where am I looking right now") is a per-terminal thing,
not per-session — see `terminal.py`. One session can have several shells open
and each is its own terminal placed at its own location.

In-process only — dies on restart, which matches session lifetime.
"""

import atlantis
from typing import Any, Dict, List, Optional


_state: Dict[str, Dict[str, Any]] = {}


def _slot() -> Dict[str, Any]:
    sk = atlantis.get_session_key()
    if not sk:
        return {}
    return _state.setdefault(sk, {})


def require_session() -> Dict[str, Any]:
    """Return the current session slot or raise if none exists."""
    sk = atlantis.get_session_key()
    if not sk:
        raise RuntimeError("No session.")
    s = _state.get(sk)
    if s is None:
        raise RuntimeError(f"Session not set: {sk}")
    return s


# --- chat_slot ---------------------------------------------------------

def set_chat_slot(sid: str) -> None:
    _slot()["chat_slot"] = sid


def chat_slot_claimed(sid: str) -> bool:
    """Is some live session currently claiming this sid as its chat_slot?"""
    return any(s.get("chat_slot") == sid for s in _state.values())


# --- room resolution ---------------------------------------------------

def get_session_room(game_key: str) -> str:
    """Resolve this session's chat room from its chat_slot's position. Raises if unset."""
    s = require_session()
    chat_slot = s.get("chat_slot")
    if not chat_slot:
        raise RuntimeError("Session has no chat_slot.")
    from dynamic_functions.Home.location import position_get
    loc = position_get(game_key, chat_slot)
    if not loc:
        raise RuntimeError(f"chat_slot '{chat_slot}' has no position.")
    return loc


# --- introspection -----------------------------------------------------

@visible
async def session_show() -> Dict[str, Any]:
    """Show the current session's identity — user_game_id, caller sid,
    caller shell path, and the slot the user is currently typing as."""
    sk = atlantis.get_session_key()
    info: Dict[str, Any] = {
        "session_key": sk or "",
        "user_game_id": atlantis.get_user_game_id(),
        "caller_sid": atlantis.get_caller() or "",
        "caller_shell_path": atlantis.get_caller_shell_path() or "",
        "exec_shell_path": atlantis.get_exec_shell_path() or "",
        "request_id": atlantis.get_request_id() or "",
        "chat_slot": "",
    }
    if sk and sk in _state:
        info["chat_slot"] = _state[sk].get("chat_slot", "") or ""
    from dynamic_functions.Home.terminal import get_terminal_location
    info["terminal_location"] = get_terminal_location() or ""
    await atlantis.client_data("Session", [info])
    return info


@visible
def session_list(game_key: str = "") -> List[Dict[str, Any]]:
    """List live sessions and their chat_slot. Pass game_key to scope to one game."""
    rows: List[Dict[str, Any]] = []

    target_uid: Optional[int] = None
    if game_key:
        import json, os
        from dynamic_functions.Home.common import game_dir
        meta_path = os.path.join(game_dir(game_key), "game.json")
        try:
            with open(meta_path) as f:
                target_uid = json.load(f).get("user_game_id")
        except (OSError, ValueError):
            target_uid = None

    for sk, s in _state.items():
        parts = sk.split(":", 1)
        user_game_id = parts[0] if len(parts) > 0 else ""
        caller_sid = parts[1] if len(parts) > 1 else ""

        if target_uid is not None and str(target_uid) != user_game_id:
            continue

        rows.append({
            "session_key": sk,
            "user_game_id": user_game_id,
            "caller_sid": caller_sid,
            "chat_slot": s.get("chat_slot"),
        })
    return rows
