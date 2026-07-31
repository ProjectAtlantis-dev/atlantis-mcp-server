"""Per-game bot tool inventories and canonical model-facing tool schemas."""

import os
import re
from typing import Any, Dict, List

import atlantis

from .common import _read_json, _write_json
from .game import require_membership
from .tool import AtlantisSearchToolT


_TOOLS_DIRNAME = "tools"
_BOT_SID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_TOOL_DEFINITIONS: Dict[str, AtlantisSearchToolT] = {
    "remember_visitor": {
        "tool_name": "remember_visitor",
        "tool_description": (
            "Remember a visitor's name after they directly tell you what it is. "
            "Call this before addressing the visitor by that name."
        ),
        "input_schema": (
            '{"type":"object","properties":{"name":{"type":"string",'
            '"description":"The name the visitor directly told you."}},'
            '"required":["name"]}'
        ),
    },
}

_DEFAULT_TOOL_NAMES: Dict[str, List[str]] = {
    "kitty": ["remember_visitor"],
}


def _normalize_bot_sid(bot_sid: str) -> str:
    bot_sid = str(bot_sid or "").strip()
    if not bot_sid or not _BOT_SID_RE.fullmatch(bot_sid):
        raise ValueError(f"Invalid bot sid: {bot_sid!r}")
    return bot_sid


def _bot_tool_path(game_key: str, bot_sid: str) -> str:
    bot_sid = _normalize_bot_sid(bot_sid)
    return os.path.join(
        require_membership(game_key),
        _TOOLS_DIRNAME,
        f"{bot_sid}.json",
    )


def _validate_tool_names(raw: Any, *, source: str) -> List[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{source} must be a JSON array")

    names: List[str] = []
    for value in raw:
        name = str(value or "").strip()
        if not name:
            raise ValueError(f"{source} contains an empty tool name")
        if name not in _TOOL_DEFINITIONS:
            raise ValueError(f"{source} references unknown bot tool {name!r}")
        if name not in names:
            names.append(name)
    return names


def bot_tool_names(game_key: str, bot_sid: str) -> List[str]:
    """Load one bot's authoritative tool-name inventory for a game."""
    path = _bot_tool_path(game_key, bot_sid)
    raw = _read_json(path, None)
    if raw is None:
        return []
    return _validate_tool_names(raw, source=path)


def get_bot_tools(game_key: str, bot_sid: str) -> List[AtlantisSearchToolT]:
    """Resolve one bot's per-game inventory to canonical tool schemas."""
    return [_TOOL_DEFINITIONS[name].copy() for name in bot_tool_names(game_key, bot_sid)]


def get_bot_tool_argument_overrides(
    game_key: str,
    bot_sid: str,
) -> Dict[str, Dict[str, Any]]:
    """Bind trusted turn context for enabled tools without exposing it to the bot."""
    bot_sid = _normalize_bot_sid(bot_sid)
    overrides: Dict[str, Dict[str, Any]] = {}
    for name in bot_tool_names(game_key, bot_sid):
        if name == "remember_visitor":
            overrides[name] = {"bot_sid": bot_sid}
    return overrides


def _initialize_bot_tool_files(
    game_key: str,
    roster: List[Dict[str, Any]],
) -> None:
    """Create missing per-game inventory files for AI bots without overwriting."""
    for row in roster:
        if row.get("ai") is not True:
            continue
        bot_sid = _normalize_bot_sid(str(row.get("bot_sid") or ""))
        path = _bot_tool_path(game_key, bot_sid)
        if os.path.exists(path):
            continue
        _write_json(path, list(_DEFAULT_TOOL_NAMES.get(bot_sid, [])))


@public
async def bot_tool_list(game_key: str, bot_sid: str) -> List[str]:
    """List the tools currently enabled for a bot in this game."""
    names = bot_tool_names(game_key, bot_sid)
    await atlantis.client_data(
        f"{bot_sid} tools",
        [{"tool": name} for name in names],
    )
    return names
