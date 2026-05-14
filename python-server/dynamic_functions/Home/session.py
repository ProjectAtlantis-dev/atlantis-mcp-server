"""Per-session state, keyed by atlantis.get_session_key() = f"{user_game_id}:{caller_sid}".

Two independent bindings per session:
  - chat_slot:   sid → whose mouth my typed text comes out of
  - camera:      {location, follow_sid} → what this terminal is looking at;
                 if follow_sid is set, location auto-resolves to that character's position.

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


# --- chat_slot ---------------------------------------------------------

def set_chat_slot(sid: str) -> None:
    _slot()["chat_slot"] = sid


def chat_slot_claimed(sid: str) -> bool:
    """Is some live session currently claiming this sid as its chat_slot?"""
    return any(s.get("chat_slot") == sid for s in _state.values())


# --- camera ------------------------------------------------------------

def set_camera_location(location: str) -> None:
    """Park this terminal's camera at a specific location. Clears any follow."""
    s = _slot()
    s["camera_location"] = location
    s.pop("camera_follow", None)


def set_camera_follow(sid: str) -> None:
    """Make this terminal's camera follow a character's position."""
    s = _slot()
    s["camera_follow"] = sid
    s.pop("camera_location", None)


# --- introspection -----------------------------------------------------

@visible
def session_list(game_key: str = "") -> List[Dict[str, Any]]:
    """List live sessions and their chat_slot / camera state. Pass game_key to scope to one game."""
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

        follow_sid = s.get("camera_follow")
        followed_at = None
        if follow_sid and game_key:
            from dynamic_functions.Home.location import position_get
            followed_at = position_get(game_key, follow_sid)

        rows.append({
            "session_key": sk,
            "user_game_id": user_game_id,
            "caller_sid": caller_sid,
            "chat_slot": s.get("chat_slot"),
            "camera_location": s.get("camera_location"),
            "camera_follow": follow_sid,
            "camera_resolved": s.get("camera_location") or followed_at,
        })
    return rows
