"""Per-session state, keyed by atlantis.get_session_key() = f"{user_game_id}:{caller_sid}:{caller_shell_path}".

Two independent bindings per session:
  - chat_slot:   sid → whose mouth my typed text comes out of
  - camera:      {location, follow_sid} → what this terminal is looking at;
                 if follow_sid is set, location auto-resolves to that character's position.

In-process only — dies on restart, which matches session lifetime.
"""

import atlantis
from typing import Any, Dict, List, Optional, Set


_state: Dict[str, Dict[str, Any]] = {}


def _key() -> Optional[str]:
    return atlantis.get_session_key()


def _own_sid() -> Optional[str]:
    return atlantis.get_caller()


def current_game_key() -> Optional[str]:
    """Resolve this session's game_key (the on-disk uuid) from the session's user_game_id.

    Scans Data/games/*/game.json once per call; cheap relative to chat/render workloads.
    Returns None if no game.json matches the current user_game_id.
    """
    import json, os
    uid = atlantis.get_user_game_id()
    if uid is None:
        return None
    from dynamic_functions.Home.common import home_path
    games_dir = home_path("Data", "games")
    if not os.path.isdir(games_dir):
        return None
    for name in os.listdir(games_dir):
        meta_path = os.path.join(games_dir, name, "game.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if meta.get("user_game_id") == uid:
            return meta.get("key") or name
    return None


def _slot() -> Dict[str, Any]:
    sk = _key()
    if not sk:
        return {}
    return _state.setdefault(sk, {})


# --- chat_slot ---------------------------------------------------------

def get_chat_slot() -> Optional[str]:
    """Whose mouth my typed text comes out of. Defaults to my own sid."""
    sk = _key()
    if sk and sk in _state and "chat_slot" in _state[sk]:
        return _state[sk]["chat_slot"]
    return _own_sid()


def set_chat_slot(sid: str) -> None:
    _slot()["chat_slot"] = sid


def all_claimed_chat_slots() -> Set[str]:
    """Every sid currently claimed by some session's chat_slot."""
    return {s["chat_slot"] for s in _state.values() if "chat_slot" in s}


def chat_slot_claimed(sid: str) -> bool:
    return sid in all_claimed_chat_slots()


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


def get_camera_location() -> str:
    """Resolve this terminal's current view: either the parked location, the followed character's
    location, or empty if nothing is set."""
    sk = _key()
    s = _state.get(sk or "", {})
    if "camera_location" in s:
        return s["camera_location"]
    follow_sid = s.get("camera_follow")
    if not follow_sid:
        return ""
    gk = current_game_key()
    if not gk:
        return ""
    from dynamic_functions.Home.location import position_get
    return position_get(gk, follow_sid) or ""


def get_camera_follow() -> Optional[str]:
    return _slot().get("camera_follow")


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
        parts = sk.split(":", 2)
        user_game_id = parts[0] if len(parts) > 0 else ""
        caller_sid = parts[1] if len(parts) > 1 else ""
        shell_path = parts[2] if len(parts) > 2 else ""

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
            "caller_shell_path": shell_path,
            "chat_slot": s.get("chat_slot"),
            "camera_location": s.get("camera_location"),
            "camera_follow": follow_sid,
            "camera_resolved": s.get("camera_location") or followed_at,
        })
    return rows
