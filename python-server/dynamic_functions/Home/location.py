"""Location listing, position queries, and movement tools."""

import atlantis
import base64
import json
import os
from typing import Any, List, Dict

from dynamic_functions.Data.main import get_positions, get_players_at
from dynamic_functions.Home.common import (
    _locations_dir, location_thumb, _ensure_thumb, move_character, _default_location, GAMES_DIR,
)
from dynamic_functions.Home.character import _load_characters


def _get_current_game():
    from dynamic_functions.Home.main import _get_current_game
    return _get_current_game()


def _thumb_for_image(image_path: str) -> str:
    """Return thumbnail path for a location image, without relying on current game."""
    if not image_path or not os.path.isfile(image_path):
        return ''
    return _ensure_thumb(image_path)


def _collect_locations(locations_dir: str) -> List[Dict[str, str]]:
    """Scan a single Locations/ directory and return location entries."""
    locations = []
    if not os.path.isdir(locations_dir):
        return locations
    for fname in sorted(os.listdir(locations_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(locations_dir, fname), 'r') as f:
            data = json.load(f)
        image_data = ''
        image_file = data.get('image', '')
        if image_file:
            thumb = _thumb_for_image(os.path.join(locations_dir, image_file))
            if thumb:
                ext = os.path.splitext(thumb)[1].lower()
                mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext.lstrip('.'), 'jpeg')
                with open(thumb, 'rb') as img:
                    b64 = base64.b64encode(img.read()).decode('ascii')
                image_data = f'data:image/{mime};base64,{b64}'
        locations.append({
            'name': data.get('name', fname[:-5]),
            'description': data.get('description', data.get('name', fname[:-5])),
            'connects_to': data.get('connects_to', []),
            'image': image_data,
        })
    return locations


@visible
async def location_list() -> List[Dict[str, str]]:
    """List all locations with their name and image (base64-encoded).

    If a game is set, lists locations for that game only.
    Otherwise lists locations across all games.
    """
    game_name = _get_current_game()

    if game_name:
        return _collect_locations(_locations_dir())

    # No game set — scan all games, prefix with game name
    locations = []
    if os.path.isdir(GAMES_DIR):
        for gname in sorted(os.listdir(GAMES_DIR)):
            lpath = os.path.join(GAMES_DIR, gname, "Locations")
            for loc in _collect_locations(lpath):
                locations.append({'game': gname, **loc})
    return locations


@visible
async def position_list() -> List[Dict[str, str]]:
    """Show current player positions for the active game."""
    game = _get_current_game()
    if not game:
        raise ValueError("No game set. Call game_set() first.")
    game_id = atlantis.get_game_id()
    if not game_id:
        raise ValueError("No active game_id in context")
    positions = get_positions(game_id)
    return [{"sid": sid, "location": loc} for sid, loc in positions.items()]


@visible
def position_query(location: str) -> List[Dict[Any, Any]]:
    """Return all characters at a given location.

    Each entry includes id, sid, role, isBot, displayName, and location.
    """
    from dynamic_functions.Home.common import _load_bot_config

    game_id = atlantis.get_game_id()
    if not game_id:
        raise ValueError("No active game in context")

    sids_at = get_players_at(game_id, location)
    characters = _load_characters()
    result = []
    for ch in characters:
        if ch["sid"] not in sids_at:
            continue
        entry = dict(ch)
        entry["location"] = location
        if ch.get("isBot", True):
            loaded = _load_bot_config(ch["sid"])
            entry["displayName"] = loaded[0].get("displayName", ch["sid"]) if loaded else ch["sid"]
        else:
            entry["displayName"] = ch.get("humanName", ch["sid"])
        result.append(entry)
    return result


@visible
async def move_bot(sid: str, location: str = "") -> str:
    """Move a bot character to a location in the active game.

    sid must be a registered bot character (via character_bot()).
    Location is optional for first-time entry (spawns in default lobby).
    Call game_set() first to lock this server to a game.
    """
    return await move_character(sid, location or "", is_bot=True, set_bg=False)


@visible
async def move_human(sid: str, location: str = "") -> str:
    """Move a human character to a location in the active game.

    sid must be a registered human character (via character_human()).
    Location is optional for first-time entry (spawns in default lobby).
    Call game_set() first to lock this server to a game.
    """
    return await move_character(sid, location or "", is_bot=False, set_bg=False)


@visible
async def go(location: str = "") -> str:
    """Move the caller's character to a location in the active game.

    Uses the caller's identity as the sid (must be registered via character_self()/character_human()).
    Location is optional for first-time entry (spawns in default lobby).
    Call game_set() first to lock this server to a game.
    Updates the background image to match the new location.
    """
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    return await move_character(sid, location or _default_location(), is_bot=False, set_bg=True)
