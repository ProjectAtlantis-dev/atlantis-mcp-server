"""Per-role interaction history.

Stored at {game_dir}/interactions/{role}/{other_sid}.json. Keyed by role POV
so any bot currently playing a role shares the role's accumulated memory of
each other participant.
"""

import os
from datetime import datetime
from typing import Any, Dict

from dynamic_functions.Home.common import _read_json, _write_json, require_game_dir, _safe_id


def _interactions_dir(game_key: str, role: str) -> str:
    return os.path.join(require_game_dir(game_key), "interactions", _safe_id(role, "role"))


def _interaction_path(game_key: str, role: str, other_sid: str) -> str:
    return os.path.join(_interactions_dir(game_key, role), f"{_safe_id(other_sid, 'sid')}.json")


def read_interaction(game_key: str, role: str, other_sid: str) -> Dict[str, Any]:
    """Return the role's recorded history with other_sid (empty record if none)."""
    record = _read_json(_interaction_path(game_key, role, other_sid), None)
    if not record:
        return {
            "role": role,
            "other_sid": other_sid,
            "first_name": "",
            "count": 0,
            "first_interaction_at": "",
            "last_interaction_at": "",
        }
    return record


def record_interaction(game_key: str, role: str, other_sid: str, first_name: str = "") -> Dict[str, Any]:
    """Increment the role's interaction count with other_sid and stamp the time."""
    record = read_interaction(game_key, role, other_sid)
    now = datetime.now().isoformat(timespec="seconds")
    record["count"] = int(record.get("count") or 0) + 1
    if not record.get("first_interaction_at"):
        record["first_interaction_at"] = now
    record["last_interaction_at"] = now
    if first_name and not record.get("first_name"):
        record["first_name"] = first_name
    _write_json(_interaction_path(game_key, role, other_sid), record)
    return record
