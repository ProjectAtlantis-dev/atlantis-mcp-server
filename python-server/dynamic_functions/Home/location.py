"""Location listing, position queries, movement tools, and all underlying location infrastructure."""

import atlantis
import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.character import (
    _current_game_name, _find_character, _load_characters,
)
from dynamic_functions.Home.common import _ensure_thumb, GAMES_DIR

logger = logging.getLogger("mcp_server")


# =========================================================================
# Directory helpers
# =========================================================================

def _locations_dir() -> str:
    return os.path.join(GAMES_DIR, "Locations")


# =========================================================================
# Location loading
# =========================================================================

def _load_location(name: str) -> Optional[Dict[str, Any]]:
    """Load a location JSON by name, or return None."""
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
    """Return the name of the default (lobby) location for the current game."""
    loc_dir = _locations_dir()
    for fname in os.listdir(loc_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(loc_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("default"):
            return data.get("name", fname[:-5])
    raise RuntimeError(f"No default location found in {loc_dir}")


# =========================================================================
# Thumbnails (location-specific)
# =========================================================================

def location_thumb(loc_name: str) -> str:
    """Return the filesystem path to a location's thumbnail image.

    Auto-generates the thumb if it doesn't exist. Returns empty string
    if the location has no image.
    """
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

def _positions_path() -> str:
    from dynamic_functions.Home.character import game_data_dir
    return os.path.join(game_data_dir(), "positions.json")


def get_positions() -> Dict[str, str]:
    """Return the full sid -> location dict for the active game."""
    path = _positions_path()
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_positions(positions: Dict[str, str]) -> None:
    """Persist the full positions dict for the active game."""
    path = _positions_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2)
    os.replace(tmp, path)


@visible
def position_get(sid: str) -> Optional[str]:
    """Return a player's current location in the active game, or None."""
    return get_positions().get(sid)


def set_player_position(sid: str, location: str) -> None:
    """Set a player's location and persist."""
    positions = get_positions()
    positions[sid] = location
    set_positions(positions)


def get_players_at(location: str) -> List[str]:
    """Return list of sids at a location in the active game."""
    return [s for s, loc in get_positions().items() if loc == location]


# =========================================================================
# Background
# =========================================================================

async def _set_location_background(location_data: Dict[str, Any]) -> None:
    """Set the client background to the location's image, if one exists."""
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

async def move_character(sid: str, location: str, is_bot: bool) -> str:
    """Shared movement logic for bot and human characters.

    New players must start in the default lobby before moving elsewhere.
    Movement is only allowed along connects_to edges defined in Locations/*.json.
    """
    location = location or ""
    character = _find_character(sid, is_bot)
    human_name = character.get("humanName", "")
    display = f"{human_name} ({sid})" if human_name else sid

    if not sid:
        raise ValueError("sid is required")

    game_name = _current_game_name()
    current = position_get(sid)

    # New player — drop them into the default lobby
    if current is None:
        default_location = _default_location()
        location = location or default_location
        if location != default_location:
            raise ValueError(
                f"New players must start in {default_location} "
                f"before moving to {location}"
            )
        dest = _load_location(location)
        if not dest:
            raise ValueError(f"Unknown location: {location}")
        desc = dest.get("description", location)
        set_player_position(sid, location)
        await atlantis.client_log(f"\U0001f3db\ufe0f New player {display} has entered {desc} for the first time")
        await atlantis.client_description(
            "Someone has entered.",
            location=location,
        )
        logger.info(f"[{game_name}] New player {sid} entered {default_location}")
        # If the owner moved, update the camera and background
        if atlantis.is_owner(sid):
            from dynamic_functions.Home.camera import camera_set
            camera_set(location)
            await _set_location_background(dest)
        return location


    if not location:
        raise ValueError("location is required for players who have already entered")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")

    desc = dest.get("description", location)

    # Already there
    if current == location:
        await atlantis.client_log(f"\U0001f4cd {display} is already in {desc}")
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
    set_player_position(sid, location)
    await atlantis.client_log(f"\U0001f6b6 {display} moved from {current_desc} to {desc}")
    await atlantis.client_description(
        "Someone has entered.",
        location=location,
    )
    logger.info(f"[{game_name}] {sid} moved from {current} to {location}")
    # If the owner moved, update the camera and background
    if atlantis.is_owner(sid):
        from dynamic_functions.Home.camera import camera_set
        camera_set(location)
        await _set_location_background(dest)
    return location


# =========================================================================
# Helper for location_list thumbnail encoding
# =========================================================================

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
        json_path = os.path.join(locations_dir, fname)
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
            'name': data.get('name', fname[:-5]),
            'description': data.get('description', data.get('name', fname[:-5])),
            'connects_to': data.get('connects_to', []),
            'image': image_data,
            'updated': datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M'),
        })
    return locations


def _get_current_game():
    from dynamic_functions.Home.main import _get_current_game
    return _get_current_game()


# =========================================================================
# Visible tools
# =========================================================================

@visible
async def location_list() -> List[Dict[str, str]]:
    """List all locations with their name and image (base64-encoded).

    If no game session is active, still lists the static location definitions.
    """
    game_name = _get_current_game()

    if game_name:
        return _collect_locations(_locations_dir())

    return _collect_locations(_locations_dir())


@visible
async def position_list() -> List[Dict[str, str]]:
    """Show current player positions for the active game."""
    game = _get_current_game()
    if not game:
        raise ValueError("No active game session.")
    positions = get_positions()
    return [{"sid": sid, "location": loc} for sid, loc in positions.items()]


@visible
def position_query(location: str) -> List[Dict[Any, Any]]:
    """Return all characters at a given location.

    Each entry includes id, sid, role, isBot, displayName, and location.
    """
    from dynamic_functions.Home.common import _load_bot_config

    _get_current_game()  # guard: requires an active game session
    sids_at = get_players_at(location)
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
    Requires an active game session.
    """
    return await move_character(sid, location or "", is_bot=True)


@visible
async def move_human(sid: str, location: str = "") -> str:
    """Move a human character to a location in the active game.

    sid must be a registered human character (via character_human()).
    Location is optional for first-time entry (spawns in default lobby).
    Requires an active game session.
    """
    return await move_character(sid, location or "", is_bot=False)


@visible
async def go(location: str = "") -> str:
    """Move the caller's character to a location in the active game.

    Uses the caller's identity as the sid (must be registered via character_self()/character_human()).
    Location is optional for first-time entry (spawns in default lobby).
    Requires an active game session.
    """
    if not _get_current_game():
        raise ValueError("No active game session.")

    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    return await move_character(sid, location or _default_location(), is_bot=False)


@visible
async def look(location: str = "") -> str:
    """Move the camera to a location, setting the background image.

    The camera is a per-game concept — it determines the background
    image independent of where any characters are.
    If no location is given, uses the caller's current position.
    """
    from dynamic_functions.Home.camera import set_camera

    _get_current_game()  # guard: requires an active game session

    if not location:
        sid = atlantis.get_caller()
        if sid:
            location = position_get(sid) or ""
    if not location:
        raise ValueError("No location specified and caller has no current position.")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")

    set_camera(location)
    await _set_location_background(dest)
    return location
