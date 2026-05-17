"""Character tools

Characters are derived from the filesystem: a `Characters/<sid>/<Role>/prompt.md`
file declares that bot <sid> plays role <Role>. There is no per-game character
registry — the (sid x role) binding is a static asset, not runtime state.
"""

import atlantis
import logging
import os
from typing import Any, Dict, List

from dynamic_functions.Home.common import (
    _load_bot_config,
    home_path,
    require_game_dir,
)

logger = logging.getLogger("mcp_server")


def _valid_roles() -> set:
    roles_dir = home_path("Game", "Roles")
    if not os.path.isdir(roles_dir):
        return set()
    return {
        entry for entry in os.listdir(roles_dir)
        if os.path.isdir(os.path.join(roles_dir, entry))
        and not entry.startswith(".") and entry != "__pycache__"
    }


def _load_characters() -> List[Dict[str, Any]]:
    """Derive characters from the filesystem.

    Each `Game/Characters/<sid>/<Role>/prompt.md` declares one character:
    bot <sid> playing role <Role>. The binding is a pure asset — it does not
    vary per game. The sid must resolve to a bot folder (for the display
    name and driving model); the role must resolve to a role folder.
    """
    roles = _valid_roles()
    characters_dir = home_path("Game", "Characters")
    if not os.path.isdir(characters_dir):
        return []
    characters: List[Dict[str, Any]] = []
    for sid in sorted(os.listdir(characters_dir)):
        sid_dir = os.path.join(characters_dir, sid)
        if not os.path.isdir(sid_dir) or sid.startswith(".") or sid == "__pycache__":
            continue
        loaded = _load_bot_config(sid)
        if not loaded:
            continue
        cfg, _ = loaded
        display_name = cfg.get("displayName", sid)
        for role in sorted(os.listdir(sid_dir)):
            role_dir = os.path.join(sid_dir, role)
            if not os.path.isdir(role_dir) or role not in roles:
                continue
            if not os.path.isfile(os.path.join(role_dir, "prompt.md")):
                continue
            characters.append({
                "sid": sid,
                "role": role,
                "displayName": display_name,
                "prompt": load_character_prompt(sid, role),
            })
    return characters


def load_character_prompt(sid: str, role: str) -> str:
    """Read this character's prompt — the (sid x role) join.

    Stored at Game/Characters/<sid>/<role>/prompt.md. Empty string if missing.
    """
    path = os.path.join(home_path("Game", "Characters", sid, role), "prompt.md")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _find_character(sid: str) -> dict:
    """Find a character by sid"""
    for ch in _load_characters():
        if ch["sid"] == sid:
            return ch
    raise ValueError(f"No character found for sid: {sid!r}. Add a prompt at Characters/{sid}/<Role>/prompt.md.")


def is_bot_driven(sid: str) -> bool:
    """A character is bot-driven iff its sid has a Bots/ config AND no live session has claimed its chat slot."""
    from dynamic_functions.Home.session import chat_slot_claimed
    if _load_bot_config(sid) is None:
        return False
    return not chat_slot_claimed(sid)


def _character_rows(game_key: str) -> List[Dict[str, Any]]:
    """Pure data: characters with current positions. No client side effects."""
    require_game_dir(game_key)
    from dynamic_functions.Home.location import get_positions
    positions = get_positions(game_key)
    return [
        {**ch, "location": positions.get(ch["sid"], "")}
        for ch in _load_characters()
    ]


@visible
async def character_list(game_key: str) -> List[Dict[str, Any]]:
    """List game characters with their current positions (blank if unplaced)."""
    characters = _character_rows(game_key)
    await atlantis.client_data("Characters", characters, column_formatter={
        "prompt": {"type": "markdown"},
    })
    return characters
