"""Game event log.

Append-only event store persisted at Data/games/{game_key}/events.json.
Temporary home until we move to a real DB.

Event types:
    chat        — someone spoke in a room
    bot_reply   — a bot responded (LLM output)
    move        — a character moved between rooms
    enter       — a character entered the game for the first time
    tool_call   — a bot invoked a tool during its turn
    system      — misc system-level note
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.common import _read_json, _write_json, require_game_dir

import atlantis


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _events_path(game_key: str) -> str:
    return os.path.join(require_game_dir(game_key), "events.json")


def _load_events(game_key: str) -> List[Dict[str, Any]]:
    return _read_json(_events_path(game_key), []) or []


def _save_events(game_key: str, events: List[Dict[str, Any]]) -> None:
    _write_json(_events_path(game_key), events)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def event_create(
    game_key: str,
    event_type: str,
    sid: str = "",
    location: str = "",
    content: str = "",
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a new event and return it.

    Parameters
    ----------
    game_key : str
        Which game this event belongs to.
    event_type : str
        One of: chat, bot_reply, move, enter, tool_call, system.
    sid : str
        The character / player that caused the event.
    location : str
        Room / location name where it happened.
    content : str
        Human-readable payload (the message text, tool name, etc.).
    data : dict, optional
        Arbitrary extra data (tool args, model used, etc.).
    """
    event: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "type": event_type,
        "sid": sid,
        "location": location,
        "content": content,
        "ts": datetime.now().isoformat(timespec="milliseconds"),
    }
    if data:
        event["data"] = data

    events = _load_events(game_key)
    events.append(event)
    _save_events(game_key, events)
    return event


@visible
async def event_list(
    game_key: str,
    event_type: str = "",
    sid: str = "",
    location: str = "",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List game events, newest first.

    Optional filters narrow by type, sid, or location. Returns at most
    `limit` events (default 50).
    """
    events = _load_events(game_key)

    if event_type:
        events = [e for e in events if e.get("type") == event_type]
    if sid:
        events = [e for e in events if e.get("sid") == sid]
    if location:
        events = [e for e in events if e.get("location") == location]

    events = list(reversed(events))[:limit]

    await atlantis.client_data("Events", events)
    return events
