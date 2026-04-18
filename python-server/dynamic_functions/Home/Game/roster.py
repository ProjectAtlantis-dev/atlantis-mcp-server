"""Runtime roster assignments.

Each user's game has a private roster.json. The roster records which bot is
assigned to each static role for that game.
"""

import json
import logging
import os
from typing import Optional, Dict, Any, List

from dynamic_functions.Home.Game.common import game_data_dir
from dynamic_functions.Home.Game.roles import get_role

logger = logging.getLogger("mcp_server")

_ROSTER_FILE = "roster.json"


def _roster_path(game_id: str, user_sid: Optional[str] = None, *, create_dir: bool = False) -> str:
    return os.path.join(game_data_dir(game_id, user_sid=user_sid, create=create_dir), _ROSTER_FILE)


def _load_roster(game_id: str, user_sid: Optional[str] = None) -> List[Dict[str, Any]]:
    path = _roster_path(game_id, user_sid)
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid game roster data in {path}: expected a list")

    return data


def _save_roster(game_id: str, roster: List[Dict[str, Any]], user_sid: Optional[str] = None) -> None:
    path = _roster_path(game_id, user_sid, create_dir=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(roster, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def assign_role(game_id: str, *, id: str, title: str, location: str,
                bot: str, requiresCheckin: bool = False,
                user_sid: Optional[str] = None, **extra) -> Dict[str, Any]:
    """Assign a bot to a role in a user's private game roster."""
    roster_entry = {
        "id": id,
        "title": title,
        "location": location,
        "bot": bot,
        "requiresCheckin": requiresCheckin,
        **extra,
    }

    roster = [
        existing
        for existing in _load_roster(game_id, user_sid)
        if existing.get("id") != id
        and existing.get("bot") != bot
        and existing.get("location") != location
    ]
    roster.append(roster_entry)
    _save_roster(game_id, roster, user_sid)

    logger.info(f"Game {game_id}: assigned bot='{bot}' to role='{id}' location='{location}'")
    return roster_entry


def get_role_for_bot(game_id: str, bot_sid: str, user_sid: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Given a user's game and bot sid, return the assigned role."""
    for roster_entry in _load_roster(game_id, user_sid):
        if roster_entry.get("bot") == bot_sid:
            return roster_entry
    return None


def get_role_for_location(game_id: str, location: str, user_sid: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Given a user's game and location, return the assigned role."""
    for roster_entry in _load_roster(game_id, user_sid):
        if roster_entry.get("location") == location:
            return roster_entry
    return None


@visible
async def roster_list(game_id: Optional[str] = None, user_sid: Optional[str] = None) -> List[Dict[str, Any]]:
    """List role assignments for the current caller/game, or an explicit user/game."""
    return _load_roster(game_id, user_sid)


@visible
async def roster_assign(role_id: str, bot_sid: str,
                        game_id: Optional[str] = None,
                        user_sid: Optional[str] = None) -> Dict[str, Any]:
    """Assign a bot to a static role in the current caller/game roster."""
    role = {**get_role(role_id), "bot": bot_sid}
    return assign_role(game_id, user_sid=user_sid, **role)


def clear_game(game_id: str, user_sid: Optional[str] = None) -> None:
    """Clean up this game's roster data."""
    path = _roster_path(game_id, user_sid)
    if os.path.exists(path):
        os.remove(path)
        game_dir = os.path.dirname(path)
        try:
            os.rmdir(game_dir)
        except OSError:
            pass
    logger.info(f"Game {game_id}: roster cleared")
