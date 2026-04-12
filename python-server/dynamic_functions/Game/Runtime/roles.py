"""Role registry — the join between bot, location, and procedures.

Game callbacks create roles at runtime. Nothing is static.
In-memory, keyed by game_id. DB table later.
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("mcp_server")

# In-memory rosters keyed by game_id
_game_rosters: Dict[str, List[Dict[str, Any]]] = {}


def create_role(game_id: str, *, id: str, title: str, location: str,
                bot: str, requiresCheckin: bool = False) -> Dict[str, Any]:
    """Create a role in a game's roster. Called by scenario game callbacks."""
    role = {
        "id": id,
        "title": title,
        "location": location,
        "bot": bot,
        "requiresCheckin": requiresCheckin,
    }
    _game_rosters.setdefault(game_id, []).append(role)
    logger.info(f"Game {game_id}: created role '{id}' -> bot='{bot}' location='{location}'")
    return role


def get_role_for_bot(game_id: str, bot_sid: str) -> Optional[Dict[str, Any]]:
    """Given a game and bot sid, return the role they're assigned to (or None)."""
    for role in _game_rosters.get(game_id, []):
        if role.get("bot") == bot_sid:
            return role
    return None


def get_role_for_location(game_id: str, location: str) -> Optional[Dict[str, Any]]:
    """Given a game and location, return the role assigned there (or None)."""
    for role in _game_rosters.get(game_id, []):
        if role.get("location") == location:
            return role
    return None


def clear_game(game_id: str) -> None:
    """Clean up a game's roster when the game ends."""
    _game_rosters.pop(game_id, None)
    logger.info(f"Game {game_id}: roster cleared")
