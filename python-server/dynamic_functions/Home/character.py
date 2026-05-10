"""Character tools"""

import atlantis
import json
import logging
import os
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.common import require_game_dir
from dynamic_functions.Home.role import _validate_role

logger = logging.getLogger("mcp_server")


def _require_active_game(game_key: Optional[str] = None) -> str:
    """Internal: activate game_key when given, else fall back to session.

    Visible/public tools must pass game_key explicitly. Internal helpers
    (move_character, etc.) call without args to use the already-pinned session.
    """
    if game_key is not None:
        from dynamic_functions.Home.game import activate_game
        return activate_game(game_key)
    from dynamic_functions.Home.game import require_game_key
    gk = require_game_key()
    require_game_dir(gk)
    return gk


def game_data_dir(game_key: Optional[str] = None) -> str:
    """Get the game data directory"""
    if game_key is None:
        from dynamic_functions.Home.game import require_game_key
        actual_game_key = require_game_key()
    else:
        actual_game_key = game_key
    return require_game_dir(actual_game_key)


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
    """Find a character by sid"""
    for ch in _load_characters():
        if ch.get("sid") == sid:
            if ch.get("isBot", True) != is_bot:
                kind = "bot" if is_bot else "human"
                raise ValueError(f"Character {sid!r} is not a {kind}")
            return ch
    kind = "character_bot()" if is_bot else "character_self() or character_human()"
    raise ValueError(f"No character found for sid: {sid!r}. Register role with {kind} first.")


async def _upsert_character(sid: str, role: str, is_bot: bool, human_name: str = "") -> None:
    """Create or update a character"""
    _validate_role(role)
    characters = _load_characters()

    record: Dict[str, Any] = {"isBot": is_bot}
    if is_bot:
        record["sid"] = sid
    else:
        record["sid"] = sid
        record["humanName"] = human_name
    record["role"] = role

    is_self = (not is_bot) and atlantis.get_caller() == sid
    if is_bot:
        subject, verb = f"Bot {sid}", "is"
    elif is_self:
        subject, verb = "You", "are"
    else:
        subject, verb = f"User {sid}", "is"
    suffix = f" named {human_name}" if (human_name and not is_bot) else ""
    message = f"{subject} {verb} now roleplaying as {role}{suffix}"

    for ch in characters:
        if ch.get("sid") == sid:
            ch.update(record)
            _save_characters(characters)
            await atlantis.client_log(message)
            return

    characters.append(record)
    _save_characters(characters)
    await atlantis.client_log(message)


@visible
async def character_bot(game_key: str, sid: str, role: str) -> None:
    """Assign a role to a bot"""
    _require_active_game(game_key)
    from dynamic_functions.Home.common import _load_bot_config, _available_bot_sids
    if _load_bot_config(sid) is None:
        raise ValueError(f"Unknown bot sid: {sid!r}. Must match a bot in Bots/ (e.g. {_available_bot_sids()})")
    await _upsert_character(sid, role, is_bot=True)


@visible
async def character_human(game_key: str, sid: str, role: str, human_name: str) -> None:
    """Assign a role to a human"""
    _require_active_game(game_key)
    if not sid:
        raise ValueError("sid is required for human characters")
    if not human_name or not human_name.strip():
        raise ValueError("human_name is required for human characters")
    await _upsert_character(sid, role, is_bot=False, human_name=human_name.strip())


@visible
async def character_self(game_key: str, role: str, human_name: str) -> None:
    """Assign a role to the caller"""
    _require_active_game(game_key)
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    await character_human(game_key, sid, role, human_name)


@visible
def character_list(game_key: str) -> List[Dict[str, Any]]:
    """List game characters"""
    _require_active_game(game_key)
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
