"""Character tools"""

import atlantis
import json
import logging
import os
from typing import Any, Dict, List

from dynamic_functions.Home.common import require_game_dir
from dynamic_functions.Home.role import _validate_role

logger = logging.getLogger("mcp_server")


def _characters_path(game_key: str) -> str:
    return os.path.join(require_game_dir(game_key), "characters.json")


def _load_characters(game_key: str) -> List[Dict[str, Any]]:
    path = _characters_path(game_key)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Invalid characters.json: expected a list")
    return data


def _save_characters(game_key: str, characters: List[Dict[str, Any]]) -> None:
    path = _characters_path(game_key)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _find_character(game_key: str, sid: str) -> dict:
    """Find a character by sid"""
    for ch in _load_characters(game_key):
        if ch.get("sid") == sid:
            return ch
    raise ValueError(f"No character found for sid: {sid!r}. Register with character_set() first.")


def is_bot_driven(game_key: str, sid: str) -> bool:
    """A character is bot-driven iff its sid has a Bots/ config AND no live session has claimed its chat slot."""
    from dynamic_functions.Home.common import _load_bot_config
    from dynamic_functions.Home.session import chat_slot_claimed
    if _load_bot_config(sid) is None:
        return False
    return not chat_slot_claimed(sid)

@visible
async def character_assign(game_key: str, sid: str, role: str, display_name: str = "") -> None:
    """Register a character (or update an existing one).

    The (sid x role) specialization prompt lives at Game/Bots/<sid>/<role>.md
    and is loaded at chat time — it's not stored on the character record.
    """
    require_game_dir(game_key)
    _validate_role(role)
    if not sid:
        raise ValueError("sid is required")

    from dynamic_functions.Home.common import _load_bot_config
    loaded = _load_bot_config(sid)
    if loaded is None:
        caller = atlantis.get_caller()
        if sid != caller:
            raise ValueError(f"Invalid sid {sid!r}: must be a bot sid or match the caller")
        if not display_name.strip():
            raise ValueError("display_name is required when sid matches the caller")
    if not display_name.strip():
        display_name = loaded[0].get("displayName", sid) if loaded else sid
    display_name = display_name.strip()

    characters = _load_characters(game_key)
    record = {"sid": sid, "role": role, "displayName": display_name}
    for ch in characters:
        if ch.get("sid") == sid:
            ch.update(record)
            _save_characters(game_key, characters)
            await atlantis.client_log(f"{display_name} ({sid}) is now roleplaying as {role}")
            return
    characters.append(record)
    _save_characters(game_key, characters)
    await atlantis.client_log(f"{display_name} ({sid}) is now roleplaying as {role}")


@visible
def character_list(game_key: str) -> List[Dict[str, Any]]:
    """List game characters with their current positions (blank if unplaced)."""
    require_game_dir(game_key)
    from dynamic_functions.Home.location import get_positions
    positions = get_positions(game_key)
    return [
        {**ch, "location": positions.get(ch["sid"], "")}
        for ch in _load_characters(game_key)
    ]
