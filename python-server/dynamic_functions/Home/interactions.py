"""Per-bot interaction history.

Stored at {game_dir}/interactions/{bot_sid}/{other_sid}.json. Memory belongs
to the bot, so a bot keeps its own accumulated history of each participant
regardless of which role it's currently playing.
"""

import os
from datetime import datetime
from typing import Any, Dict

from dynamic_functions.Home.common import _read_json, _write_json, require_game_dir, _safe_id


def _interactions_dir(game_key: str, bot_sid: str) -> str:
    return os.path.join(require_game_dir(game_key), "interactions", _safe_id(bot_sid, "sid"))


def _interaction_path(game_key: str, bot_sid: str, other_sid: str) -> str:
    return os.path.join(_interactions_dir(game_key, bot_sid), f"{_safe_id(other_sid, 'sid')}.json")


def read_interaction(game_key: str, bot_sid: str, other_sid: str) -> Dict[str, Any]:
    """Return the bot's recorded history with other_sid (empty record if none)."""
    record = _read_json(_interaction_path(game_key, bot_sid, other_sid), None)
    if not record:
        return {
            "bot_sid": bot_sid,
            "other_sid": other_sid,
            "first_name": "",
            "count": 0,
            "first_interaction_at": "",
            "last_interaction_at": "",
        }
    return record


def record_interaction(game_key: str, bot_sid: str, other_sid: str, first_name: str = "") -> Dict[str, Any]:
    """Increment the bot's interaction count with other_sid and stamp the time."""
    record = read_interaction(game_key, bot_sid, other_sid)
    now = datetime.now().isoformat(timespec="seconds")
    record["count"] = int(record.get("count") or 0) + 1
    if not record.get("first_interaction_at"):
        record["first_interaction_at"] = now
    record["last_interaction_at"] = now
    if first_name and not record.get("first_name"):
        record["first_name"] = first_name
    _write_json(_interaction_path(game_key, bot_sid, other_sid), record)
    return record
