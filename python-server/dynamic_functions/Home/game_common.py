"""Shared helpers for game callbacks — bot config loading and spawning."""

import atlantis
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dynamic_functions.Data.main import game_dir

logger = logging.getLogger("mcp_server")

BOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "Bots")


def game_data_dir(game_id: Optional[str] = None, *, create: bool = True) -> str:
    """Return the data directory for the current game."""
    actual_game_id = game_id if game_id is not None else atlantis.get_game_id()
    if not actual_game_id:
        raise RuntimeError("game_data_dir requires an active game")
    return game_dir(actual_game_id, create=create)


def _load_bot_config(bot_sid: str, bots_dir: str = BOTS_DIR) -> Optional[Tuple[Dict[str, Any], str]]:
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


def _available_bot_sids(bots_dir: str = BOTS_DIR) -> List[str]:
    """Return all known bot sids from Bots/ config files."""
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

GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "Games")


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
def character_human(sid: str, role: str, human_name: str) -> str:
    """Assign a human character. Returns the UUID.

    sid is a unique identifier for the human player. human_name is their display name.
    Role must be a folder under Games/<game>/Roles/.
    If a character with this sid exists, updates it. Otherwise creates a new entry.
    """
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

async def spawn_bot(bot_sid: str, bots_dir: str = BOTS_DIR) -> None:
    """Spawn a bot: show their face image and announce them."""
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
