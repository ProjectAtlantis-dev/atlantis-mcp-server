"""Atlantis game — player movement between locations.

Positions are persisted in Data/{game_id}/positions.json.
New players must start in AtlantisLobby before they can go anywhere else.
Movement is only allowed along connects_to edges defined in Locations/*.json.
"""

import atlantis
import json
import logging
import os
from typing import Any, Dict, Optional

from dynamic_functions.Data.main import (
    get_player_position,
    set_player_position,
    ensure_location_data,
)
from dynamic_functions.Home.game_common import _load_characters

logger = logging.getLogger("mcp_server")

LOCATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Locations")
DEFAULT_LOCATION = "AtlantisLobby"


# =========================================================================
# Location graph
# =========================================================================

def _load_location(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(LOCATIONS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _connects_to(location_name: str) -> list[str]:
    loc = _load_location(location_name)
    if not loc:
        return []
    return loc.get("connects_to", [])


# =========================================================================
# Helpers
# =========================================================================

async def _set_location_background(location_data: Dict[str, Any]) -> None:
    """Set the client background to the location's image, if one exists."""
    image_name = location_data.get("image")
    if not image_name:
        return
    image_path = os.path.join(LOCATIONS_DIR, image_name)
    if os.path.exists(image_path):
        await atlantis.set_background(image_path)
    else:
        logger.warning(f"[Atlantis] Location image not found: {image_path}")


# =========================================================================
# Core logic
# =========================================================================

def _find_character(sid: str, is_bot: bool) -> dict:
    """Look up a character by sid and verify its isBot flag."""
    for ch in _load_characters():
        if ch.get("sid") == sid:
            if ch.get("isBot", True) != is_bot:
                kind = "bot" if is_bot else "human"
                raise ValueError(f"Character {sid!r} is not a {kind}")
            return ch
    kind = "character_bot()" if is_bot else "character_human()"
    raise ValueError(f"No character found for sid: {sid!r}. Register with {kind} first.")


async def _move_to(sid: str, location: str, is_bot: bool) -> str:
    """Shared movement logic for bot and human characters."""
    location = location or ""
    _find_character(sid, is_bot)

    game_id = atlantis.get_game_id()
    if not game_id:
        raise ValueError("No active game in context")
    if not sid:
        raise ValueError("sid is required")

    current = get_player_position(game_id, sid)

    # New player — drop them into the default lobby
    if current is None:
        location = location or DEFAULT_LOCATION
        if location != DEFAULT_LOCATION:
            raise ValueError(
                f"New players must start in {DEFAULT_LOCATION} "
                f"before moving to {location}"
            )
        dest = _load_location(location)
        if not dest:
            raise ValueError(f"Unknown location: {location}")
        desc = dest.get("description", location)
        set_player_position(game_id, sid, location)
        ensure_location_data(game_id, location)
        await _set_location_background(dest)
        await atlantis.client_log(f"🏛️ {sid} has entered {desc} for the first time")
        logger.info(f"[Atlantis] New player {sid} entered {DEFAULT_LOCATION}")
        return location

    if not location:
        raise ValueError("location is required for players who have already entered")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")

    desc = dest.get("description", location)

    # Already there
    if current == location:
        await atlantis.client_log(f"📍 {sid} is already in {desc}")
        return location

    # Check adjacency
    reachable = _connects_to(current)
    if location not in reachable:
        raise ValueError(
            f"Cannot reach {location} from {current}. "
            f"Reachable: {reachable}"
        )

    # Move
    current_desc = (_load_location(current) or {}).get("description", current)
    set_player_position(game_id, sid, location)
    ensure_location_data(game_id, location)
    await _set_location_background(dest)
    await atlantis.client_log(f"🚶 {sid} moved from {current_desc} to {desc}")
    logger.info(f"[Atlantis] {sid} moved from {current} to {location}")
    return location


@visible
async def move_bot(sid: str, location: str = "") -> str:
    """Move a bot character to a location. Returns the destination.

    sid must be a registered bot character (via character_bot()).
    For first-time entry, omit location to spawn in the default lobby.
    """
    return await _move_to(sid, location, is_bot=True)


@visible
async def move_human(sid: str, location: str = "") -> str:
    """Move a human character to a location. Returns the destination.

    sid must be a registered human character (via character_human()).
    For first-time entry, omit location to spawn in the default lobby.
    """
    return await _move_to(sid, location, is_bot=False)


