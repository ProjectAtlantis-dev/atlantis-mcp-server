"""Per-game bot tool inventories and canonical model-facing tool schemas."""

import json
import os
import re
from typing import Any, Dict, List

import atlantis

from .common import _read_json, _write_json
from .game import require_membership
from .tool import AtlantisSearchToolT


_TOOLS_DIRNAME = "tools"
_BOT_SID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_REMOVED_TOOL_NAMES = {"dir"}

# App folder holding the dispatched bot tools. Definitions carrying a tool_app
# are routed to a real dynamic function; search is handled inside the turn loop
# and is never addressed by path.
_TOOL_APP = "Chat"

_TOOL_DEFINITIONS: Dict[str, AtlantisSearchToolT] = {
    "search": {
        "tool_name": "search",
        "tool_description": (
            "Search for available tools and commands. Results are added to your "
            "tool list for this turn. Use this when you need a capability you "
            "do not currently have."
        ),
        "input_schema": (
            '{"type":"object","properties":{"query":{"type":"string",'
            '"description":"One or two words describing the capability to find."}},'
            '"required":["query"]}'
        ),
    },
    "remember_visitor": {
        "tool_name": "remember_visitor",
        "tool_app": _TOOL_APP,
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
    "kitty": ["search", "remember_visitor"],
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
        if name in _REMOVED_TOOL_NAMES:
            continue
        if name not in _TOOL_DEFINITIONS:
            raise ValueError(f"{source} references unknown bot tool {name!r}")
        if name not in names:
            names.append(name)
    return names


def _normalize_tool_name(tool_name: str) -> str:
    tool_name = str(tool_name or "").strip()
    if not tool_name:
        raise ValueError("tool_name required")
    if tool_name not in _TOOL_DEFINITIONS:
        raise ValueError(f"Unknown bot tool: {tool_name!r}")
    return tool_name


def bot_tool_names(game_key: str, bot_sid: str) -> List[str]:
    """Load one bot's authoritative tool-name inventory for a game."""
    path = _bot_tool_path(game_key, bot_sid)
    raw = _read_json(path, None)
    if raw is None:
        return []
    return _validate_tool_names(raw, source=path)


def _write_bot_tool_names(game_key: str, bot_sid: str, names: List[str]) -> None:
    path = _bot_tool_path(game_key, bot_sid)
    _write_json(path, _validate_tool_names(names, source=path))


def _bot_tool_signature(name: str) -> str:
    definition = _TOOL_DEFINITIONS[name]
    schema = json.loads(
        definition.get("input_schema", "")
        or '{"type":"object","properties":{}}'
    )
    properties = schema.get("properties", {})
    params = ",".join(
        f"{param_name}:{param_schema.get('type', 'string')}"
        for param_name, param_schema in properties.items()
    )
    return f"{name}({params})"


def _resolve_tool_definition(name: str) -> AtlantisSearchToolT:
    """Copy a canonical definition, stamping this remote onto dispatched tools."""
    definition = _TOOL_DEFINITIONS[name].copy()
    if not definition.get("tool_app"):
        return definition

    owner = atlantis.get_default_owner()
    remote = atlantis.get_server_info()["remote_name"]
    if not owner or not remote:
        raise RuntimeError(
            f"Cannot address bot tool {name!r}: remote identity is not resolved yet "
            f"(owner={owner!r} remote={remote!r})"
        )
    definition["remote_owner"] = owner
    definition["remote_name"] = remote
    return definition


def get_bot_tools(game_key: str, bot_sid: str) -> List[AtlantisSearchToolT]:
    """Resolve one bot's per-game inventory to canonical tool schemas."""
    return [_resolve_tool_definition(name) for name in bot_tool_names(game_key, bot_sid)]


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
async def bot_tool_list(game_key: str, bot_sid: str) -> List[Dict[str, Any]]:
    """List enabled tools with the exact descriptions and parameters shown to this bot."""
    names = bot_tool_names(game_key, bot_sid)
    rows = [
        {
            "tool": _bot_tool_signature(name),
            "description": _TOOL_DEFINITIONS[name].get("tool_description", ""),
        }
        for name in names
    ]
    await atlantis.client_data(
        f"{bot_sid} tools",
        rows,
    )
    return rows


@public
async def bot_tool_add(game_key: str, bot_sid: str, tool_name: str) -> List[str]:
    """Enable a canonical tool for a bot in the current game."""
    bot_sid = _normalize_bot_sid(bot_sid)
    tool_name = _normalize_tool_name(tool_name)
    names = bot_tool_names(game_key, bot_sid)
    if tool_name not in names:
        names.append(tool_name)
        _write_bot_tool_names(game_key, bot_sid, names)
    await atlantis.client_log(
        f"Enabled bot tool {tool_name!r} for {bot_sid!r} in game {game_key!r}"
    )
    return names


@public
async def bot_tool_remove(game_key: str, bot_sid: str, tool_name: str) -> List[str]:
    """Disable a canonical tool for a bot in the current game."""
    bot_sid = _normalize_bot_sid(bot_sid)
    tool_name = _normalize_tool_name(tool_name)
    names = bot_tool_names(game_key, bot_sid)
    if tool_name in names:
        names.remove(tool_name)
        _write_bot_tool_names(game_key, bot_sid, names)
    await atlantis.client_log(
        f"Disabled bot tool {tool_name!r} for {bot_sid!r} in game {game_key!r}"
    )
    return names
