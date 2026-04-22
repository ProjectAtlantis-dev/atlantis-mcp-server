"""Character management — creation, listing, roles, and queries."""

import atlantis
import json
import logging
import os
from typing import Any, Dict, List, Optional

from dynamic_functions.Data.main import game_dir

logger = logging.getLogger("mcp_server")

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


def game_data_dir(game_id: Optional[str] = None, *, create: bool = True) -> str:
    """Return the data directory for the current game."""
    actual_game_id = game_id if game_id is not None else atlantis.get_game_id()
    if not actual_game_id:
        raise RuntimeError("game_data_dir requires an active game")
    return game_dir(actual_game_id, create=create)


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
    from dynamic_functions.Home.common import _load_bot_config, _available_bot_sids
    if _load_bot_config(sid) is None:
        raise ValueError(f"Unknown bot sid: {sid!r}. Must match a bot in Bots/ (e.g. {_available_bot_sids()})")
    return _upsert_character(sid, role, is_bot=True)


@visible
def character_human(sid: str, role: str, human_name: str) -> str:
    """Assign a human character. Returns the UUID.

    sid identifies the human. human_name is their display name.
    Role must be a folder under Games/<game>/Roles/.
    If a character with this sid exists, updates it. Otherwise creates a new entry.
    """
    if not sid:
        raise ValueError("sid is required for human characters")
    if not human_name or not human_name.strip():
        raise ValueError("human_name is required for human characters")
    return _upsert_character(sid, role, is_bot=False, human_name=human_name.strip())


@visible
def character_self(role: str, human_name: str) -> str:
    """Assign a human character using the caller's identity as the sid. Returns the UUID.

    human_name is the caller's display name. Role must be a folder under Games/<game>/Roles/.
    If a character with this sid exists, updates it. Otherwise creates a new entry.
    """
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    return character_human(sid, role, human_name)


@visible
def character_list() -> List[Dict[str, Any]]:
    """Return all characters for the current game.

    Each entry includes id, sid, role, isBot, and a resolved displayName.
    Bot characters pull displayName from Bots/ config; human characters
    use humanName.
    """
    from dynamic_functions.Home.common import _load_bot_config
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
