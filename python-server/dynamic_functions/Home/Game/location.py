"""Shared location/room primitives.

A Game defines its locations in locations.json (id, background, adjacency).
Per-player state lives in the user's game dir as state.json.
"""

import atlantis
import json
import os
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.Game.common import game_data_dir


_STATE_FILE = "state.json"
_LOCATIONS_FILE = "locations.json"


def load_locations(game_dir: str) -> List[Dict[str, Any]]:
    """Load the location graph for a game from <game_dir>/locations.json."""
    path = os.path.join(game_dir, _LOCATIONS_FILE)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of location definitions")
    return data


def find_location(game_dir: str, location_id: str) -> Optional[Dict[str, Any]]:
    for loc in load_locations(game_dir):
        if loc.get("id") == location_id:
            return loc
    return None


def _state_path(game_id: Optional[str] = None, user_sid: Optional[str] = None) -> str:
    return os.path.join(game_data_dir(game_id, user_sid, create=True), _STATE_FILE)


def _read_state(game_id: Optional[str] = None, user_sid: Optional[str] = None) -> Dict[str, Any]:
    path = _state_path(game_id, user_sid)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_state(state: Dict[str, Any], game_id: Optional[str] = None, user_sid: Optional[str] = None) -> None:
    path = _state_path(game_id, user_sid)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def current_location(game_id: Optional[str] = None, user_sid: Optional[str] = None) -> Optional[str]:
    return _read_state(game_id, user_sid).get("location")


async def enter_location(location_id: str, game_dir: str, *,
                         game_id: Optional[str] = None,
                         user_sid: Optional[str] = None) -> Dict[str, Any]:
    """Unconditionally place the player in location_id. Sets background, writes state."""
    loc = find_location(game_dir, location_id)
    if not loc:
        raise ValueError(f"Unknown location: {location_id}")

    background = loc.get("background")
    if background:
        await atlantis.set_background(os.path.join(game_dir, background))

    state = _read_state(game_id, user_sid)
    state["location"] = location_id
    _write_state(state, game_id, user_sid)
    return loc


async def move_to(location_id: str, game_dir: str, *,
                  game_id: Optional[str] = None,
                  user_sid: Optional[str] = None) -> Dict[str, Any]:
    """Move to an adjacent location. Raises if not reachable from current."""
    current = current_location(game_id, user_sid)
    if current:
        current_loc = find_location(game_dir, current)
        adjacent = (current_loc or {}).get("adjacent", [])
        if location_id not in adjacent:
            raise ValueError(f"'{location_id}' is not adjacent to '{current}'")
    return await enter_location(location_id, game_dir, game_id=game_id, user_sid=user_sid)
