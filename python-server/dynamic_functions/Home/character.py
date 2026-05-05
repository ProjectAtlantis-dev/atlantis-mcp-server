"""Character management — creation, listing, roles, and queries."""

import atlantis
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.common import require_game_dir

logger = logging.getLogger("mcp_server")

GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "Games")


def _current_game_name() -> str:
    """Return the current game name, but only if a game session is active."""
    from dynamic_functions.Home.main import _get_current_game
    name = _get_current_game()
    if not name:
        raise RuntimeError("No active game session.")
    return name


def _find_game_dir() -> str:
    """Resolve the game definition folder."""
    _current_game_name()
    path = GAMES_DIR
    if not os.path.isdir(path):
        raise RuntimeError(f"Game folder not found: {path}")
    return path


def game_data_dir(game_id: Optional[str] = None) -> str:
    """Return the data directory for the current game."""
    actual_game_id = game_id if game_id is not None else atlantis.get_game_id()
    if not actual_game_id:
        raise RuntimeError("game_data_dir requires an active game")
    return require_game_dir(actual_game_id)


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
    kind = "character_bot()" if is_bot else "character_self() or character_human()"
    raise ValueError(f"No character found for sid: {sid!r}. Register role with {kind} first.")


@visible
def _load_role_json(roles_dir: str, role_name: str) -> dict:
    """Read a role's role.json, returning {} if missing."""
    rjson = os.path.join(roles_dir, role_name, "role.json")
    if os.path.isfile(rjson):
        with open(rjson) as f:
            return json.load(f)
    return {}


def _collect_roles(roles_dir: str) -> List[Dict[str, str]]:
    """Scan a single Roles/ directory and return role entries."""
    roles = []
    if not os.path.isdir(roles_dir):
        return roles
    for d in sorted(os.listdir(roles_dir)):
        role_dir = os.path.join(roles_dir, d)
        if os.path.isdir(role_dir) and not d.startswith(".") and d != "__pycache__":
            rdata = _load_role_json(roles_dir, d)
            mtimes = [os.path.getmtime(os.path.join(role_dir, f)) for f in os.listdir(role_dir)
                      if os.path.isfile(os.path.join(role_dir, f))]
            updated = datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M') if mtimes else ''
            roles.append({
                "name": d,
                "title": rdata.get("title", d),
                "greeting": rdata.get("greeting", ""),
                "updated": updated,
            })
    return roles


@visible
def role_list():
    """Return available roles.

    If a game session is active, returns roles for that session.
    Otherwise returns the static role definitions.
    """
    from dynamic_functions.Home.game import _get_current_game
    game_name = _get_current_game()

    if game_name:
        return _collect_roles(os.path.join(_find_game_dir(), "Roles"))

    return _collect_roles(os.path.join(GAMES_DIR, "Roles"))


def _validate_role(role: str) -> None:
    """Raise if role folder doesn't exist under the current game."""
    roles_dir = os.path.join(_find_game_dir(), "Roles")
    if not os.path.isdir(os.path.join(roles_dir, role)):
        raise ValueError(f"Role folder not found: {role}")


async def _upsert_character(sid: str, role: str, is_bot: bool, human_name: str = "") -> None:
    """Shared upsert logic for bot and human characters."""
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
async def character_bot(sid: str, role: str) -> None:
    """Assign a bot character.

    sid must match a bot in Bots/. Role must be a folder under Games/Roles/.
    If a character with this sid exists, updates it. Otherwise creates a new entry.
    """
    _current_game_name()  # guard: requires an active game session
    from dynamic_functions.Home.common import _load_bot_config, _available_bot_sids
    if _load_bot_config(sid) is None:
        raise ValueError(f"Unknown bot sid: {sid!r}. Must match a bot in Bots/ (e.g. {_available_bot_sids()})")
    await _upsert_character(sid, role, is_bot=True)


@visible
async def character_human(sid: str, role: str, human_name: str) -> None:
    """Assign a human character.

    sid identifies the human. human_name is their display name.
    Role must be a folder under Games/Roles/.
    If a character with this sid exists, updates it. Otherwise creates a new entry.
    """
    _current_game_name()  # guard: requires an active game session
    if not sid:
        raise ValueError("sid is required for human characters")
    if not human_name or not human_name.strip():
        raise ValueError("human_name is required for human characters")
    await _upsert_character(sid, role, is_bot=False, human_name=human_name.strip())


@visible
async def character_self(role: str, human_name: str) -> None:
    """Assign a human character using the caller's identity as the sid.

    human_name is the caller's display name. Role must be a folder under Games/Roles/.
    If a character with this sid exists, updates it. Otherwise creates a new entry.
    """
    _current_game_name()  # guard: requires an active game session
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    await character_human(sid, role, human_name)


@visible
def character_list() -> List[Dict[str, Any]]:
    """Return all characters for the current game.

    Each entry includes id, sid, role, isBot, and a resolved displayName.
    Bot characters pull displayName from Bots/ config; human characters
    use humanName.
    """
    _current_game_name()  # guard: requires an active game session
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
