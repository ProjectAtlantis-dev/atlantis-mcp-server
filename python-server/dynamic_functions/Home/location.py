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

def _location_dir(name: str) -> str:
    return os.path.join(_locations_dir(), name)


def _load_location(name: str) -> Optional[Dict[str, Any]]:
    """Load a location by name (folder name is the identifier)"""
    path = os.path.join(_location_dir(name), "config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _connects_to(location_name: str) -> List[str]:
    loc = _load_location(location_name)
    if not loc:
        return []
    return loc.get("connects_to", [])


def _child_locations(location_name: str) -> List[str]:
    """Names of locations whose parent is `location_name`."""
    loc_dir = _locations_dir()
    if not os.path.isdir(loc_dir):
        return []
    children: List[str] = []
    for entry in os.listdir(loc_dir):
        cfg = os.path.join(loc_dir, entry, "config.json")
        if not os.path.isfile(cfg):
            continue
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (data.get("parent") or "") == location_name:
            children.append(entry)
    return children


def _is_leaf(location_name: str) -> bool:
    """A location is a valid move target only if it has no children (containers aren't standable)."""
    return not _child_locations(location_name)


def _require_leaf(location_name: str) -> None:
    if not _is_leaf(location_name):
        children = _child_locations(location_name)
        raise ValueError(
            f"{location_name} is a container, not a place you can stand. "
            f"It contains: {', '.join(children)}."
        )


def _default_location() -> str:
    """Get the default location (must be a leaf — containers aren't valid)"""
    loc_dir = _locations_dir()
    for entry in os.listdir(loc_dir):
        cfg = os.path.join(loc_dir, entry, "config.json")
        if not os.path.isfile(cfg):
            continue
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("default"):
            _require_leaf(entry)
            return entry
    raise RuntimeError(f"No default location found in {loc_dir}")


def _entry_location(game_key: str, sid: str) -> str:
    """Get the first location for a character based on the character's role."""
    from dynamic_functions.Home.role import role_default_location

    character = _find_character(sid)
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
    image_path = os.path.join(_location_dir(loc_name), image_file)
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

async def _set_location_background(loc_name: str, location_data: Dict[str, Any]) -> None:
    """Set the location background"""
    image_name = location_data.get("image")
    if not image_name:
        return
    image_path = os.path.join(_location_dir(loc_name), image_name)
    if os.path.exists(image_path):
        await atlantis.set_background(image_path)
    else:
        logger.warning(f"Location image not found: {image_path}")


# =========================================================================
# Movement
# =========================================================================

@visible
async def character_move(game_key: str, location: str = "", sid: str = "") -> str:
    """Move a character to a location. sid defaults to the caller."""
    require_game_dir(game_key)

    if not sid:
        sid = atlantis.get_caller() or ""
    if not sid:
        raise ValueError("Unable to determine character to move (no sid and no caller).")

    location = location or ""
    character = _find_character(sid)
    display_name = character.get("displayName", sid)
    display = f"{display_name} ({sid})" if display_name != sid else sid

    current = position_get(game_key, sid)

    # New characters start in their configured entry location.
    if current is None:
        entry_location = _entry_location(game_key, sid)
        location = location or entry_location
        if location != entry_location:
            raise ValueError(
                f"New characters must start in {entry_location} "
                f"before moving to {location}"
            )
        dest = _load_location(location)
        if not dest:
            raise ValueError(f"Unknown location: {location}")
        _require_leaf(location)
        desc = dest.get("displayName", location)
        set_player_position(game_key, sid, location)
        await atlantis.client_log(f"\U0001f3db\ufe0f New player {display} has entered {desc} for the first time")
        await atlantis.client_description(
            "Someone has entered.",
            location=location,
        )
        logger.info(f"[game] New player {sid} entered {entry_location}")
        if atlantis.get_caller() == sid:
            await _set_location_background(location, dest)
        return location


    if not location:
        raise ValueError("location is required for players who have already entered")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")
    _require_leaf(location)

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
    current_desc = (_load_location(current) or {}).get("displayName", current)
    set_player_position(game_key, sid, location)
    await atlantis.client_log(f"\U0001f6b6 {display} moved from {current_desc} to {desc}")
    await atlantis.client_description(
        "Someone has entered.",
        location=location,
    )
    logger.info(f"[game] {sid} moved from {current} to {location}")
    if atlantis.get_caller() == sid:
        await _set_location_background(dest)
    return location


# =========================================================================
# Visible tools
# =========================================================================

def _location_rows() -> List[Dict[str, str]]:
    """Pure data: list locations. No client side effects."""
    locations_dir = _locations_dir()
    if not os.path.isdir(locations_dir):
        return []
    locations: List[Dict[str, str]] = []
    for entry in sorted(os.listdir(locations_dir)):
        entry_dir = os.path.join(locations_dir, entry)
        if not os.path.isdir(entry_dir) or entry.startswith('.') or entry == '__pycache__':
            continue
        json_path = os.path.join(entry_dir, 'config.json')
        if not os.path.isfile(json_path):
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        name = entry  # folder name is the identifier
        image_data = ''
        image_file = data.get('image', '')
        mtimes = [os.path.getmtime(json_path)]
        if image_file:
            image_path = os.path.join(entry_dir, image_file)
            if os.path.isfile(image_path):
                mtimes.append(os.path.getmtime(image_path))
                thumb = _ensure_thumb(image_path)
                if thumb:
                    ext = os.path.splitext(thumb)[1].lower().lstrip('.')
                    mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext, 'jpeg')
                    with open(thumb, 'rb') as img:
                        b64 = base64.b64encode(img.read()).decode('ascii')
                    image_data = f'data:image/{mime};base64,{b64}'
        locations.append({
            'name': name,
            'displayName': data.get('displayName', name),
            'parent': data.get('parent') or '',
            'connects_to': data.get('connects_to', []),
            'description': data.get('description', ''),
            'image': image_data,
            'updated': datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M'),
        })
    return locations


def compose_setting(location_name: str) -> str:
    """Walk from the root down to `location_name` and concatenate descriptions.

    Returns one paragraph per level (root first), so the prompt reads
    outer-context → inner-context. Empty string if the location has no
    description and no ancestors with one.
    """
    chain: List[str] = []
    seen: set = set()
    current = location_name
    while current and current not in seen:
        seen.add(current)
        loc = _load_location(current)
        if not loc:
            break
        chain.append(current)
        current = loc.get("parent") or ""
    parts: List[str] = []
    for name in reversed(chain):
        loc = _load_location(name) or {}
        desc = (loc.get("description") or "").strip()
        if desc:
            parts.append(desc)
    return "\n\n".join(parts)


@visible
async def location_list() -> List[Dict[str, str]]:
    """List locations"""
    locations = _location_rows()
    await atlantis.client_data("Locations", locations, column_formatter={
        "description": {"maxWidth": "80ch"},
    })
    return locations


@visible
def position_query(game_key: str, location: str) -> List[Dict[Any, Any]]:
    """List characters at a location"""
    require_game_dir(game_key)
    sids_at = get_players_at(game_key, location)
    result = []
    for ch in _load_characters():
        if ch["sid"] not in sids_at:
            continue
        entry = dict(ch)
        entry["location"] = location
        result.append(entry)
    return result


@visible
async def camera_look(game_key: str, location: str = "") -> str:
    """Park this terminal's camera at a location (or the caller's current position)."""
    from dynamic_functions.Home.session import set_camera_location

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

    set_camera_location(location)
    await _set_location_background(location, dest)
    return location


@visible
async def camera_follow(game_key: str, sid: str) -> str:
    """Make this terminal's camera follow a character's position."""
    from dynamic_functions.Home.session import set_camera_follow

    require_game_dir(game_key)
    if not sid:
        raise ValueError("sid is required")
    set_camera_follow(sid)
    loc = position_get(game_key, sid) or ""
    if loc:
        dest = _load_location(loc)
        if dest:
            await _set_location_background(loc, dest)
    return loc

