"""Per-game bot tool inventories of server-supplied tool entries."""

import os
import re
from typing import Any, Dict, List

import atlantis

from .common import _read_json, _write_json
from .game import require_membership
from .tool import AtlantisSearchToolT


_TOOLS_DIRNAME = "tools"
_BOT_SID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _normalize_bot_sid(bot_sid: str) -> str:
    bot_sid = str(bot_sid or "").strip()
    if not bot_sid or not _BOT_SID_RE.fullmatch(bot_sid):
        raise ValueError(f"Invalid bot sid: {bot_sid!r}")
    return bot_sid


def _bot_tool_path(game_name: str, game_key: str, bot_sid: str) -> str:
    bot_sid = _normalize_bot_sid(bot_sid)
    return os.path.join(
        require_membership(game_name, game_key),
        _TOOLS_DIRNAME,
        f"{bot_sid}.json",
    )


def get_bot_tools(game_name: str, game_key: str, bot_sid: str) -> List[AtlantisSearchToolT]:
    """Load one bot's per-game inventory of server-supplied tool entries."""
    path = _bot_tool_path(game_name, game_key, bot_sid)
    raw = _read_json(path, None)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON array")
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path} contains a non-object tool entry: {entry!r}")
    return raw


def get_bot_tool_argument_overrides(
    game_name: str,
    game_key: str,
    bot_sid: str,
) -> Dict[str, Dict[str, Any]]:
    """Bind trusted turn context for bot tools without exposing it to the bot."""
    bot_sid = _normalize_bot_sid(bot_sid)
    return {
        "remember_visitor": {"game_name": game_name, "bot_sid": bot_sid},
        "bot_locate": {"game_name": game_name, "game_key": game_key, "bot_sid": bot_sid},
    }


def _initialize_bot_tool_files(
    game_name: str,
    game_key: str,
    roster: List[Dict[str, Any]],
) -> None:
    """Create missing per-game inventory files for AI bots without overwriting."""
    for row in roster:
        if row.get("ai") is not True:
            continue
        bot_sid = _normalize_bot_sid(str(row.get("bot_sid") or ""))
        path = _bot_tool_path(game_name, game_key, bot_sid)
        if os.path.exists(path):
            continue
        _write_json(path, [])


@public
async def bot_tool_list(game_name: str, game_key: str, bot_sid: str) -> List[AtlantisSearchToolT]:
    """List the tool entries in this bot's inventory for the game."""
    tools = get_bot_tools(game_name, game_key, bot_sid)
    await atlantis.client_data(f"{bot_sid} tools", tools)
    return tools
