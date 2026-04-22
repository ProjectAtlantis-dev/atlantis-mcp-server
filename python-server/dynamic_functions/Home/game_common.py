"""Shared helpers for game callbacks — bot config loading and spawning."""

import atlantis
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dynamic_functions.Data.main import game_dir, get_player_position, set_player_position

logger = logging.getLogger("mcp_server")

GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "Games")


def _bots_dir() -> str:
    from dynamic_functions.Home.main import _get_current_game
    return os.path.join(GAMES_DIR, _get_current_game(), "Bots")


def _locations_dir() -> str:
    from dynamic_functions.Home.main import _get_current_game
    return os.path.join(GAMES_DIR, _get_current_game(), "Locations")


def game_data_dir(game_id: Optional[str] = None, *, create: bool = True) -> str:
    """Return the data directory for the current game."""
    actual_game_id = game_id if game_id is not None else atlantis.get_game_id()
    if not actual_game_id:
        raise RuntimeError("game_data_dir requires an active game")
    return game_dir(actual_game_id, create=create)


def _load_bot_config(bot_sid: str, bots_dir: Optional[str] = None) -> Optional[Tuple[Dict[str, Any], str]]:
    if bots_dir is None:
        bots_dir = _bots_dir()
    """Find config.json for a bot by sid under bots_dir. Returns (config, folder_name) or None."""
    for entry in os.listdir(bots_dir):
        config_path = os.path.join(bots_dir, entry, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("sid") == bot_sid:
                cfg["_botDir"] = os.path.join(bots_dir, entry)
                return cfg, entry
    return None


def _available_bot_sids(bots_dir: Optional[str] = None) -> List[str]:
    """Return all known bot sids from Bots/ config files."""
    if bots_dir is None:
        bots_dir = _bots_dir()
    sids = []
    if not os.path.isdir(bots_dir):
        return sids
    for entry in os.listdir(bots_dir):
        config_path = os.path.join(bots_dir, entry, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                if "sid" in cfg:
                    sids.append(cfg["sid"])
            except (json.JSONDecodeError, OSError):
                pass
    return sorted(sids)


# =========================================================================
# Characters — stored in Data/<game_id>/characters.json
# =========================================================================


def _current_game_name() -> str:
    """Return the current game name, but only if it has been locked via game_set()."""
    from dynamic_functions.Home.main import _get_current_game
    name = _get_current_game()
    if not name:
        raise RuntimeError("No game locked. Call game_set() first.")
    return name


def _find_game_dir() -> str:
    """Resolve the current game's definition folder under Games/."""
    name = _current_game_name()
    path = os.path.join(GAMES_DIR, name)
    if not os.path.isdir(path):
        raise RuntimeError(f"Game folder not found: {name}")
    return path


def _characters_path() -> str:
    return os.path.join(game_data_dir(), "characters.json")


def _load_characters() -> List[Dict[str, Any]]:
    path = _characters_path()
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Invalid characters.json: expected a list")
    return data


def _save_characters(characters: List[Dict[str, Any]]) -> None:
    path = _characters_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)

@visible
def role_list() -> List[str]:
    """Return available role names (subfolder names under Games/<game>/Roles/)."""
    roles_dir = os.path.join(_find_game_dir(), "Roles")
    if not os.path.isdir(roles_dir):
        return []
    return sorted(
        d for d in os.listdir(roles_dir)
        if os.path.isdir(os.path.join(roles_dir, d))
        and not d.startswith(".")
        and d != "__pycache__"
    )


def _validate_role(role: str) -> None:
    """Raise if role folder doesn't exist under the current game."""
    roles_dir = os.path.join(_find_game_dir(), "Roles")
    if not os.path.isdir(os.path.join(roles_dir, role)):
        raise ValueError(f"Role folder not found: {role}")


def _upsert_character(sid: str, role: str, is_bot: bool, human_name: str = "") -> str:
    """Shared upsert logic for bot and human characters. Returns the UUID."""
    _validate_role(role)
    characters = _load_characters()

    record: Dict[str, Any] = {"isBot": is_bot}
    if is_bot:
        record["sid"] = sid
    else:
        record["sid"] = sid
        record["humanName"] = human_name
    record["role"] = role

    for ch in characters:
        if ch.get("sid") == sid:
            ch.update(record)
            _save_characters(characters)
            logger.info(f"Updated character {sid}: role={role} isBot={is_bot}")
            return sid

    characters.append(record)
    _save_characters(characters)
    logger.info(f"Created character {sid}: role={role} isBot={is_bot}")
    return sid


@visible
def character_bot(sid: str, role: str) -> str:
    """Assign a bot character. Returns the UUID.

    sid must match a bot in Bots/. Role must be a folder under Games/<game>/Roles/.
    If a character with this sid exists, updates it. Otherwise creates a new entry.
    """
    if _load_bot_config(sid) is None:
        raise ValueError(f"Unknown bot sid: {sid!r}. Must match a bot in Bots/ (e.g. {_available_bot_sids()})")
    return _upsert_character(sid, role, is_bot=True)


@visible
def character_human(role: str, human_name: str) -> str:
    """Assign a human character. Returns the UUID.

    Uses the caller's identity as the sid. human_name is their display name.
    Role must be a folder under Games/<game>/Roles/.
    If a character with this sid exists, updates it. Otherwise creates a new entry.
    """
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    if not human_name or not human_name.strip():
        raise ValueError("human_name is required for human characters")
    return _upsert_character(sid, role, is_bot=False, human_name=human_name.strip())

@visible
def character_list() -> List[Dict[str, Any]]:
    """Return all characters for the current game.

    Each entry includes id, sid, role, isBot, and a resolved displayName.
    Bot characters pull displayName from Bots/ config; human characters
    use humanName.
    """
    characters = _load_characters()
    result = []
    for ch in characters:
        entry = dict(ch)
        if ch.get("isBot", True):
            loaded = _load_bot_config(ch["sid"])
            entry["displayName"] = loaded[0].get("displayName", ch["sid"]) if loaded else ch["sid"]
        else:
            entry["displayName"] = ch.get("humanName", ch["sid"])
        result.append(entry)
    return result


@visible
def position_query(location: str) -> List[Dict[str, Any]]:
    """Return all characters at a given location.

    Each entry includes id, sid, role, isBot, displayName, and location.
    """
    from dynamic_functions.Data.main import get_players_at

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


# =========================================================================
# Bot spawning
# =========================================================================

# =========================================================================
# Movement
# =========================================================================


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


async def move_character(sid: str, location: str, is_bot: bool, set_bg: bool = True) -> str:
    """Shared movement logic for bot and human characters.

    Positions are persisted in Data/{game_id}/positions.json.
    New players must start in the default lobby before moving elsewhere.
    Movement is only allowed along connects_to edges defined in Locations/*.json.
    """
    location = location or ""
    _find_character(sid, is_bot)

    game_id = atlantis.get_game_id()
    if not game_id:
        raise ValueError("No active game in context")
    if not sid:
        raise ValueError("sid is required")

    game_name = _current_game_name()
    current = get_player_position(game_id, sid)

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
        set_player_position(game_id, sid, location)
        if set_bg:
            await _set_location_background(dest)
        await atlantis.client_log(f"\U0001f3db\ufe0f {sid} has entered {desc} for the first time")
        logger.info(f"[{game_name}] New player {sid} entered {default_location}")
        return location

    if not location:
        raise ValueError("location is required for players who have already entered")

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")

    desc = dest.get("description", location)

    # Already there
    if current == location:
        await atlantis.client_log(f"\U0001f4cd {sid} is already in {desc}")
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
    if set_bg:
        await _set_location_background(dest)
    await atlantis.client_log(f"\U0001f6b6 {sid} moved from {current_desc} to {desc}")
    logger.info(f"[{game_name}] {sid} moved from {current} to {location}")
    return location


# =========================================================================
# Bot spawning
# =========================================================================


async def spawn_bot(bot_sid: str, bots_dir: Optional[str] = None) -> None:
    """Spawn a bot: show their face image and announce them."""
    if bots_dir is None:
        bots_dir = _bots_dir()
    loaded = _load_bot_config(bot_sid, bots_dir)
    if not loaded:
        logger.warning(f"No config.json found for bot sid: {bot_sid}")
        return
    cfg, folder = loaded

    display_name = cfg.get("displayName", folder)

    bot_dir = os.path.join(bots_dir, folder)
    face_candidates = [f for f in os.listdir(bot_dir) if "face" in f.lower() and f.lower().endswith((".jpg", ".png", ".webp"))]
    if face_candidates:
        face_path = os.path.join(bot_dir, face_candidates[0])
        await atlantis.client_image(face_path)
        logger.info(f"Spawned {display_name}: showed face image")

    # Say hello as a chat message so the bot shows up in the transcript.
    greeting = cfg.get("greeting", f"Hi, I'm {display_name}.")
    stream_id = await atlantis.stream_start(bot_sid, display_name)
    await atlantis.stream(greeting, stream_id)
    await atlantis.stream_end(stream_id)
