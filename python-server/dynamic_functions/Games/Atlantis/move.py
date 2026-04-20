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
# Core logic
# =========================================================================

@visible
async def move_to(sid: str, location: str) -> None:
    """Move a player to a location.

    game_id is obtained from the current atlantis context.

    Raises:
        ValueError: if any argument is missing, location is unknown,
                    player hasn't been through the lobby, or the
                    destination isn't reachable from current position.
    """
    game_id = atlantis.get_game_id()
    if not game_id:
        raise ValueError("No active game in context")
    if not sid:
        raise ValueError("sid is required")
    if not location:
        raise ValueError("location is required")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")

    current = get_player_position(game_id, sid)

    desc = dest.get("description", location)

    # New player — must enter lobby first
    if current is None:
        if location != DEFAULT_LOCATION:
            raise ValueError(
                f"New players must start in {DEFAULT_LOCATION} "
                f"before moving to {location}"
            )
        set_player_position(game_id, sid, location)
        ensure_location_data(game_id, location)
        await atlantis.client_log(f"🏛️ {sid} has entered {desc} for the first time")
        logger.info(f"[Atlantis] New player {sid} entered {DEFAULT_LOCATION}")
        return

    # Already there
    if current == location:
        await atlantis.client_log(f"📍 {sid} is already in {desc}")
        return

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
    await atlantis.client_log(f"🚶 {sid} moved from {current_desc} to {desc}")
    logger.info(f"[Atlantis] {sid} moved from {current} to {location}")


