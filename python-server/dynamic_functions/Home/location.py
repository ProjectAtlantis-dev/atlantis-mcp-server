"""Location tools"""

import atlantis
import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.character import (
    _find_character, _load_characters,
)
from dynamic_functions.Home.common import _ensure_thumb, home_path, require_game_dir

logger = logging.getLogger("mcp_server")


# =========================================================================
# Directory helpers
# =========================================================================

def _locations_dir() -> str:
    return home_path("Game", "Locations")


# =========================================================================
# Location loading
# =========================================================================

def _load_location(name: str) -> Optional[Dict[str, Any]]:
    """Load a location by name"""
    path = os.path.join(_locations_dir(), f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _connects_to(location_name: str) -> List[str]:
    loc = _load_location(location_name)
    if not loc:
        return []
    return loc.get("connects_to", [])


def _default_location() -> str:
    """Get the default location"""
    loc_dir = _locations_dir()
    for fname in os.listdir(loc_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(loc_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("default"):
            return data.get("name", fname[:-5])
    raise RuntimeError(f"No default location found in {loc_dir}")


def _entry_location(game_key: str, sid: str, is_bot: bool) -> str:
    """Get the first location for a character based on the character's role."""
    from dynamic_functions.Home.role import role_default_location

    character = _find_character(game_key, sid, is_bot)
    role = character.get("role", "")
    if role:
        role_location = role_default_location(role)
        if role_location:
            return role_location
    return _default_location()


# =========================================================================
# Thumbnails (location-specific)
# =========================================================================

def location_thumb(loc_name: str) -> str:
    """Get a location thumbnail path"""
    logger.info(f"[thumb] location_thumb called: {loc_name!r}")
    loc = _load_location(loc_name)
    if not loc:
        logger.warning(f"[thumb] location not found: {loc_name!r}")
        return ""
    image_file = loc.get("image", "")
    if not image_file:
        logger.warning(f"[thumb] no image for location: {loc_name!r}")
        return ""
    image_path = os.path.join(_locations_dir(), image_file)
    logger.info(f"[thumb] location {loc_name!r} -> {image_path}")
    if not os.path.isfile(image_path):
        logger.warning(f"[thumb] image file missing: {image_path}")
        return ""
    return _ensure_thumb(image_path)


# =========================================================================
# Positions — {sid: location_name}, persisted per game
# =========================================================================

def _positions_path(game_key: str) -> str:
    return os.path.join(require_game_dir(game_key), "positions.json")


def get_positions(game_key: str) -> Dict[str, str]:
    """Get all player positions"""
    path = _positions_path(game_key)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_positions(game_key: str, positions: Dict[str, str]) -> None:
    """Save all player positions"""
    path = _positions_path(game_key)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2)
    os.replace(tmp, path)


@visible
def position_get(game_key: str, sid: str) -> Optional[str]:
    """Get a player's location"""
    require_game_dir(game_key)
    return get_positions(game_key).get(sid)


def set_player_position(game_key: str, sid: str, location: str) -> None:
    """Set a player's location"""
    positions = get_positions(game_key)
    positions[sid] = location
    set_positions(game_key, positions)


def get_players_at(game_key: str, location: str) -> List[str]:
    """List players at a location"""
    return [s for s, loc in get_positions(game_key).items() if loc == location]


# =========================================================================
# Background
# =========================================================================

async def _set_location_background(location_data: Dict[str, Any]) -> None:
    """Set the location background"""
    image_name = location_data.get("image")
    if not image_name:
        return
    image_path = os.path.join(_locations_dir(), image_name)
    if os.path.exists(image_path):
        await atlantis.set_background(image_path)
    else:
        logger.warning(f"Location image not found: {image_path}")


# =========================================================================
# Movement
# =========================================================================

async def move_character(game_key: str, sid: str, location: str, is_bot: bool) -> str:
    """Move a bot or human character"""
    location = location or ""
    character = _find_character(game_key, sid, is_bot)
    human_name = character.get("humanName", "")
    display = f"{human_name} ({sid})" if human_name else sid

    if not sid:
        raise ValueError("sid is required")

    require_game_dir(game_key)
    current = position_get(game_key, sid)

    # New characters start in their configured entry location.
    if current is None:
        entry_location = _entry_location(game_key, sid, is_bot)
        location = location or entry_location
        if location != entry_location:
            raise ValueError(
                f"New characters must start in {entry_location} "
                f"before moving to {location}"
            )
        dest = _load_location(location)
        if not dest:
            raise ValueError(f"Unknown location: {location}")
        desc = dest.get("description", location)
        set_player_position(game_key, sid, location)
        await atlantis.client_log(f"\U0001f3db\ufe0f New player {display} has entered {desc} for the first time")
        await atlantis.client_description(
            "Someone has entered.",
            location=location,
        )
        logger.info(f"[game] New player {sid} entered {entry_location}")
        # Track the moving character's own camera; refresh the background only
        # for the client whose caller is actually moving themselves.
        from dynamic_functions.Home.camera import camera_set
        camera_set(game_key, location, sid)
        if atlantis.get_caller() == sid:
            await _set_location_background(dest)
        return location


    if not location:
        raise ValueError("location is required for players who have already entered")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")

    desc = dest.get("description", location)

    # No-op when already there
    if current == location:
        await atlantis.client_log(f"\U0001f4cd {display} is already in {desc}")
        return location

    # Enforce adjacency
    reachable = _connects_to(current)
    if location not in reachable:
        raise ValueError(
            f"Cannot reach {location} from {current}. "
            f"Reachable: {reachable}"
        )

    # Apply movement
    current_desc = (_load_location(current) or {}).get("description", current)
    set_player_position(game_key, sid, location)
    await atlantis.client_log(f"\U0001f6b6 {display} moved from {current_desc} to {desc}")
    await atlantis.client_description(
        "Someone has entered.",
        location=location,
    )
    logger.info(f"[game] {sid} moved from {current} to {location}")
    # Track the moving character's own camera; refresh the background only
    # for the client whose caller is actually moving themselves.
    from dynamic_functions.Home.camera import camera_set
    camera_set(game_key, location, sid)
    if atlantis.get_caller() == sid:
        await _set_location_background(dest)
    return location


# =========================================================================
# Helper for location_list thumbnail encoding
# =========================================================================

def _thumb_for_image(image_path: str) -> str:
    """Get a thumbnail path for a location image"""
    if not image_path or not os.path.isfile(image_path):
        return ''
    return _ensure_thumb(image_path)


def _collect_locations(locations_dir: str) -> List[Dict[str, str]]:
    """List locations from a directory"""
    locations = []
    if not os.path.isdir(locations_dir):
        return locations
    for entry in sorted(os.listdir(locations_dir)):
        if not entry.endswith('.json'):
            continue
        json_path = os.path.join(locations_dir, entry)
        if not os.path.isfile(json_path):
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        image_data = ''
        image_file = data.get('image', '')
        mtimes = [os.path.getmtime(json_path)]
        if image_file:
            image_path = os.path.join(locations_dir, image_file)
            if os.path.isfile(image_path):
                mtimes.append(os.path.getmtime(image_path))
            thumb = _thumb_for_image(image_path)
            if thumb:
                ext = os.path.splitext(thumb)[1].lower()
                mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext.lstrip('.'), 'jpeg')
                with open(thumb, 'rb') as img:
                    b64 = base64.b64encode(img.read()).decode('ascii')
                image_data = f'data:image/{mime};base64,{b64}'
        locations.append({
            'name': data.get('name', entry[:-5]),
            'description': data.get('description', data.get('name', entry[:-5])),
            'connects_to': data.get('connects_to', []),
            'image': image_data,
            'updated': datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M'),
        })
    return locations


# =========================================================================
# Visible tools
# =========================================================================

@visible
async def location_list() -> List[Dict[str, str]]:
    """List locations"""
    return _collect_locations(_locations_dir())


@visible
async def position_list(game_key: str) -> List[Dict[str, str]]:
    """List player positions"""
    require_game_dir(game_key)
    positions = get_positions(game_key)
    return [{"sid": sid, "location": loc} for sid, loc in positions.items()]


@visible
def position_query(game_key: str, location: str) -> List[Dict[Any, Any]]:
    """List characters at a location"""
    from dynamic_functions.Home.common import _load_bot_config

    require_game_dir(game_key)
    sids_at = get_players_at(game_key, location)
    characters = _load_characters(game_key)
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
async def move_bot(game_key: str, sid: str, location: str = "") -> str:
    """Move a bot character"""
    require_game_dir(game_key)
    return await move_character(game_key, sid, location or "", is_bot=True)


@visible
async def move_human(game_key: str, sid: str, location: str = "") -> str:
    """Move a human character"""
    require_game_dir(game_key)
    return await move_character(game_key, sid, location or "", is_bot=False)


@visible
async def go(game_key: str, location: str = "") -> str:
    """Move the caller's character"""
    require_game_dir(game_key)

    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    return await move_character(game_key, sid, location or "", is_bot=False)


@visible
async def look(game_key: str, location: str = "") -> str:
    """Move the camera to a location"""
    from dynamic_functions.Home.camera import camera_set

    require_game_dir(game_key)

    if not location:
        sid = atlantis.get_caller()
        if sid:
            location = position_get(game_key, sid) or ""
    if not location:
        raise ValueError("No location specified and caller has no current position.")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")

    camera_set(game_key, location)
    await _set_location_background(dest)
    return location

