import atlantis
import copy
import json
import os
import posixpath
import time as _t

from datetime import datetime, timezone

from dotenv import dotenv_values
from mcp.shared.exceptions import McpError
from openai import AsyncOpenAI, omit
from openai.types.chat import ChatCompletionFunctionToolParam
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple, cast

from .bot import load_bot, render_bot_prompt
from .common import _read_json, _write_json, silent_command, silently
from .runner import _game_init_script
from .tool import (
    logger,
    AtlantisSearchToolT, OpenAITool, ToolLookupInfo,
    _repair_json, _signature_name, coerce_args_to_schema, convert_discovery_rows,
    convert_folder_rows, convert_search_tools, ROBOT_FOLDER,
)
from utils import format_json_log


async def _close_streams(talk_id, think_id):
    """Close open stream IDs"""
    for sid in [think_id, talk_id]:
        if sid:
            try:
                await atlantis.stream_end(sid)
            except Exception as e:
                # Runs in run_turn's finally: report and keep closing, so the
                # turn's own exception is not replaced by this one.
                logger.exception(f"Failed to close stream {sid}")
                await atlantis.client_warning(
                    f"Failed to close stream {sid}: {type(e).__name__}: {e}",
                    level="ERROR",
                )


async def _record_telemetry(game_dir: str, record: Dict[str, Any]) -> None:
    """Log one API round's served model/provider/usage and append it to telemetry.jsonl."""
    usage = record["usage"] or {}
    summary = (
        f"telemetry: bot={record['bot_sid']!r} turn={record['turn']} "
        f"gen={record['generation_id']} model={record['model']!r} provider={record['provider']!r} "
        f"tokens={usage.get('prompt_tokens')}/{usage.get('completion_tokens')} "
        f"cost={usage.get('cost')} latency={record['latency_s']}s"
    )
    logger.info(summary)
    await silently(atlantis.client_log(summary))

    # A round cut off at the chunk limit never reaches the usage chunk
    expected = ["generation_id", "model", "provider"] + ([] if record["stream_aborted"] else ["usage"])
    missing = [k for k in expected if record[k] is None]
    if missing:
        logger.error(f"Telemetry missing {missing} for {record['bot_sid']!r} turn {record['turn']}")
        await atlantis.client_warning(
            f"API response for {record['bot_sid']!r} turn {record['turn']} did not report: {', '.join(missing)}",
            level="ERROR",
        )

    path = os.path.join(game_dir, "telemetry.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.exception(f"Failed to append telemetry to {path}")
        await atlantis.client_warning(
            f"Failed to append telemetry to {path}: {type(e).__name__}: {e}",
            level="ERROR",
        )


# Read on every turn so a key added during setup works without a restart.
_CHAT_ENV = Path(__file__).resolve().parent / ".env"


def _chat_env(name: str) -> str:
    """Return a setting from Chat/.env, falling back to the process environment."""
    value = dotenv_values(_CHAT_ENV).get(name) if _CHAT_ENV.is_file() else None
    return value or os.environ.get(name, "")


_DISCOVERY_TOOLS = ("search",)

# Each discovery call costs a full model round trip, so a bot that keeps
# guessing synonyms stalls the reply. Close discovery after this many calls.
_MAX_DISCOVERY_CALLS = 3


async def _search_rows(query: str) -> Any:
    """Run /search; a request no tool fits comes back as an empty list.

    The query travels in the data section, never the command line: chat text
    carries apostrophes, ';' and '|' that the cloud's shell parser would read
    as quoting and sequencing.
    """
    return await silent_command("/search", {"content": query})


# A zone is a game whose app has a Robot folder, named as ROBOT in
# <game_name>/init.py (e.g. "$Demo/Robot"). For a zone, Chat itself:
#   - remembers who the zone's bot has met, per game (visitors.json)
#   - tells the bot how the user types and which folder they are in
#   - runs the Robot's get_status(command_prefix, current_folder) every round
#   - loads the zone's own tools into every round, from one of two places:
#       1. the Robot has a Tools/ folder (Demo): only those. The app folder
#          holds what humans run, and the bot gets a curated set on top.
#       2. the Robot has no Tools/ folder (Setup): the app folder's own tools,
#          the same ones humans see there. Its logic is exposed in the open,
#          and Chat does not check how each tool is marked.
#     plus, in either mode, the folders its tool_folders returns, if it has
#     one (Setup names the Robot/Tools of each package it installs)
#   - runs the Robot's record_turn, if it has one, after each reply
# A game without a Robot (Greenland) gets none of this.
_ROBOT_TOOLS_FOLDER = "Tools"
_ROBOT_STATUS_TOOL = "get_status"
_ROBOT_AFTER_TURN_TOOL = "record_turn"
_ROBOT_TOOL_FOLDERS_TOOL = "tool_folders"
# In each zone game's folder: who its bot has met, {username: {lastSeen}}
_VISITORS_FILENAME = "visitors.json"


def _game_robot(game_name: str) -> Optional[str]:
    """The game's Robot folder term, or None when the game is not a zone"""
    robot = getattr(_game_init_script(game_name), "ROBOT", None)
    if robot is None:
        return None
    if not isinstance(robot, str) or robot[:1] not in ('%', '$', '~'):
        raise ValueError(f"Game {game_name!r} ROBOT must be an anchored folder term, got {robot!r}")
    return robot


async def _ls(folder_term: str) -> List[Dict[str, Any]]:
    rows = await silent_command("/ls", {"searchTerm": f"{folder_term}/*"})
    if not isinstance(rows, list):
        raise TypeError(f"/ls {folder_term}/* returned {type(rows).__name__}, expected a list")
    return rows


def _defines(rows: List[Dict[str, Any]], function_name: str) -> bool:
    """Whether /ls rows of a folder include the function function_name"""
    return any(row.get("name") and _signature_name(row["name"]) == function_name for row in rows)


async def _robot_has(robot: str, function_name: str) -> bool:
    """Whether the Robot folder itself defines function_name"""
    return _defines(await _ls(robot), function_name)


def _robot_app(robot: str) -> str:
    """The app folder term a Robot folder term sits in ($Setup/Robot -> $Setup)"""
    app, name = posixpath.split(robot)
    if name != ROBOT_FOLDER or not app:
        raise ValueError(f"Robot term must name an app's {ROBOT_FOLDER} folder, got {robot!r}")
    return app


async def _robot_tools(robot: Optional[str]) -> Tuple[List[OpenAITool], Dict[str, ToolLookupInfo]]:
    """The zone's bot tools, listed fresh: its Robot's Tools/ when it has one,
    else the app folder's own tools; then each folder its tool_folders names.
    A game without a Robot has none."""
    if robot is None:
        return [], {}
    robot_rows = await _ls(robot)
    # /ls names a subfolder in its dir field ("Tools/"), with no name
    if any(row.get("dir") == f"{_ROBOT_TOOLS_FOLDER}/" for row in robot_rows):
        folders = [f"{robot}/{_ROBOT_TOOLS_FOLDER}"]
    else:
        folders = [_robot_app(robot)]
    if _defines(robot_rows, _ROBOT_TOOL_FOLDERS_TOOL):
        named = await silent_command(f"{robot}/{_ROBOT_TOOL_FOLDERS_TOOL}")
        if not isinstance(named, list) or not all(isinstance(folder, str) for folder in named):
            raise TypeError(f"{robot}/{_ROBOT_TOOL_FOLDERS_TOOL} must return a list of folder terms, got {named!r}")
        folders += named

    openai_tools: List[OpenAITool] = []
    tool_lookup: Dict[str, ToolLookupInfo] = {}
    for folder_term in folders:
        tools, lookup = convert_folder_rows(await _ls(folder_term), folder_term)
        for tool in tools:
            name = tool["function"]["name"]
            if name in tool_lookup:
                raise ValueError(f"Tool {name!r} in {folder_term} clashes with {tool_lookup[name]['searchTerm']}")
            openai_tools.append(tool)
            tool_lookup[name] = lookup[name]
    return openai_tools, tool_lookup


def _caller_username() -> str:
    username = atlantis.get_caller()
    if not username:
        raise RuntimeError("No caller identity in this bot turn")
    return username


def _visitors_path(game_name: str, game_key: str) -> str:
    from .game import require_membership
    return os.path.join(require_membership(game_name, game_key), _VISITORS_FILENAME)


def _read_visitors(game_name: str, game_key: str) -> Dict[str, Any]:
    path = _visitors_path(game_name, game_key)
    visitors = _read_json(path, {})
    if not isinstance(visitors, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return visitors


async def _user_terminal() -> Tuple[str, str, str]:
    """(command prefix, current folder, how the user types) from the caller's
    shell env: ui.appMode is set when they type in the chat window, where a
    Multix command needs a leading /; a terminal takes it bare."""
    shell = atlantis.get_caller_shell_path()
    if not shell:
        raise RuntimeError("No caller shell; a zone's bot turn must run from the user's terminal")
    ui = await silent_command(f"/env peek {json.dumps(shell)} ui")
    path = await silent_command(f"/env peek {json.dumps(shell)} path")
    if not isinstance(ui, dict) or not isinstance(path, dict):
        raise TypeError(f"/env peek of shell {shell!r} returned {ui!r} and {path!r}, expected objects")
    folder = "/" + "/".join(path["current"])

    if ui.get("appMode") is True:
        return "/", folder, (
            "The user types in the chat window: plain text goes to the chat, a line starting "
            "with / runs a Multix command and a line starting with @ calls a tool, so every "
            "Multix command you give them starts with /.")
    return "", folder, (
        "The user types in a Multix terminal, not the chat window: a line runs as a Multix "
        "command exactly as typed and a line starting with @ calls a tool. Never tell them to "
        "put / in front of a command.")


async def _robot_status(game_name: str, game_key: str, robot: Optional[str]) -> Optional[Dict[str, str]]:
    """What the zone's bot must know at the start of every round, fetched
    fresh instead of trusting the bot to ask: who the user is and whether it
    has met them, how they type and where they are, then the Robot's own
    status. get_status runs every round, so it may write only what it
    derives once per user message (a Robot memoizing its choice of behavior)."""
    if robot is None:
        return None
    username = _caller_username()
    if username in _read_visitors(game_name, game_key):
        seen = f"You have already met {username} in this game; carry on from where you are."
    else:
        seen = f"You have not met {username}; introduce yourself."
    prefix, folder, typing = await _user_terminal()

    status_tool = f"{robot}/{_ROBOT_STATUS_TOOL}"
    status = await silent_command(status_tool, {"command_prefix": prefix, "current_folder": folder})
    if not isinstance(status, str):
        raise TypeError(f"{status_tool} must return a string, got {type(status).__name__}")
    lines = [f"The user is {username}. {seen}", f"{typing} Their current folder is {folder}.", status]
    return {'role': 'system', 'content': "Current status, fetched for this turn:\n" + "\n".join(lines)}


async def _robot_after_turn(game_name: str, game_key: str, robot: Optional[str]) -> None:
    """Once a zone bot's reply has gone out: remember that it has met the
    user in this game, then run the Robot's record_turn if it has one."""
    if robot is None:
        return
    username = _caller_username()
    visitors = _read_visitors(game_name, game_key)
    visitors[username] = {"lastSeen": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_json(_visitors_path(game_name, game_key), visitors)

    if not await _robot_has(robot, _ROBOT_AFTER_TURN_TOOL):
        return
    after_turn_tool = f"{robot}/{_ROBOT_AFTER_TURN_TOOL}"
    result = await silent_command(after_turn_tool)
    logger.info(f"{after_turn_tool}: {result}")


def _message_text(message: Dict[str, Any]) -> str:
    """Text of a message whose content is a string or a list of parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


# Spoken lines the pre-load search sees, so a bare "yes" or "it failed" still
# carries what it answers.
_PRELOAD_SEARCH_LINES = 3


def _recent_spoken_text(transcript: List[Dict[str, Any]], count: int) -> str:
    """Last `count` spoken user/assistant lines (no scene lines or tool calls),
    oldest first, flattened to one line for /search."""
    lines: List[str] = []
    for message in reversed(transcript):
        if message.get("role") not in ("user", "assistant"):
            continue
        text = " ".join(_message_text(message).split())
        if not text or text.startswith("[scene:"):
            continue
        lines.append(text)
        if len(lines) == count:
            break
    return " ".join(reversed(lines))


def _openrouter_payload_path(game_name: str, game_key: str, sid: str) -> str:
    """Return the per-game snapshot path for one bot's latest payload."""
    from .game import require_membership

    sid = str(sid or "").strip()
    if not sid:
        raise ValueError("sid required")
    load_bot(game_name, game_key, sid)
    return os.path.join(
        require_membership(game_name, game_key),
        "openrouter_payloads",
        f"{sid}.json",
    )


@public
async def openrouter_payload(game_name: str, sid: str) -> Dict[str, Any]:
    """Return this game's latest exact OpenRouter payload for a bot SID."""
    from .game import game_require_current

    game_key = game_require_current(game_name)
    path = _openrouter_payload_path(game_name, game_key, sid)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise FileNotFoundError(
            f"No OpenRouter payload has been submitted for {sid!r} in game {game_key!r}"
        )
    return payload


_SEARCH_TOOL: OpenAITool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Find tools in the system by keyword and add the matches to this "
            "reply. Use one or two words."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "One or two keywords"},
            },
            "required": ["query"],
        },
    },
}


def _open_discovery(
    openai_tools: List[OpenAITool],
    tool_lookup: Dict[str, ToolLookupInfo],
) -> None:
    """Offer search for this reply; nothing in a bot's inventory supplies it."""
    name = _SEARCH_TOOL["function"]["name"]
    if name in tool_lookup:
        raise ValueError(f"Bot inventory already declares the discovery tool {name!r}")
    openai_tools.append(copy.deepcopy(_SEARCH_TOOL))
    # Dispatch requires a lookup row; search itself is handled by _execute_discovery_tool.
    tool_lookup[name] = {"searchTerm": "/search", "filename": "", "functionName": name}


def _close_discovery(
    openai_tools: List[OpenAITool],
    tool_lookup: Dict[str, ToolLookupInfo],
) -> None:
    """Withdraw search for the rest of this reply."""
    openai_tools[:] = [
        tool for tool in openai_tools
        if tool["function"]["name"] not in _DISCOVERY_TOOLS
    ]
    for name in _DISCOVERY_TOOLS:
        tool_lookup.pop(name, None)


ToolBinder = Callable[[OpenAITool, ToolLookupInfo], None]


async def _cursor_values() -> Dict[str, Any]:
    """Cursor contents at the start of this turn."""
    await atlantis.client_command("/silent on")
    try:
        cursor = await atlantis.client_command("/cursor show")
    finally:
        await atlantis.client_command("/silent off")
    if not isinstance(cursor, dict):
        raise TypeError(f"/cursor show returned {type(cursor).__name__}, expected an object")
    return cursor


def _make_tool_binder(
    cursor: Dict[str, Any],
    tool_argument_overrides: Dict[str, Dict[str, Any]],
    tool_bindings: Dict[str, Dict[str, Any]],
) -> ToolBinder:
    """Close each tool over the values this turn supplies.

    A parameter the cursor holds, or a turn override names, is removed from the
    model-facing schema and recorded in tool_bindings (keyed by model-facing
    name) for execution to fill in. Overrides win over the cursor.
    """
    def bind(tool: OpenAITool, lookup: ToolLookupInfo) -> None:
        parameters = tool["function"]["parameters"]
        # A schema with no parameters may omit properties entirely
        properties = parameters.get("properties", {})
        bound = {k: v for k, v in cursor.items() if k in properties}
        bound.update(tool_argument_overrides.get(lookup["functionName"], {}))
        for param in bound:
            properties.pop(param, None)
        if "required" in parameters:
            parameters["required"] = [p for p in parameters["required"] if p not in bound]
        tool_bindings[tool["function"]["name"]] = bound

    return bind


async def _build_round_tools(
    *,
    robot: Optional[str],
    tools: List[AtlantisSearchToolT],
    tool_argument_overrides: Dict[str, Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    searched_queries: List[str],
    discovery_open: bool,
) -> Tuple[List[OpenAITool], Dict[str, ToolLookupInfo], Dict[str, Dict[str, Any]], ToolBinder, Optional[Dict[str, str]]]:
    """One round's tool list, rebuilt from scratch so a tool run last round
    cannot leave it stale: the bot's inventory, bound to the current cursor,
    plus what the recent conversation asks for and every search the bot has
    made this turn, rerun against the system as it is now. The bot's own
    tools come first, then its zone's, then search. Also returns a system
    note saying what the search of the recent conversation found, so the bot
    knows when nothing fits instead of guessing."""
    openai_tools, tool_lookup = convert_search_tools(tools)
    zone_tools, zone_lookup = await _robot_tools(robot)
    for tool in zone_tools:
        name = tool["function"]["name"]
        if name in tool_lookup:
            raise ValueError(f"Zone tool {name!r} clashes with a tool in the bot's inventory")
        openai_tools.append(tool)
        tool_lookup[name] = zone_lookup[name]
    cursor = await _cursor_values()
    tool_bindings: Dict[str, Dict[str, Any]] = {}
    bind = _make_tool_binder(cursor, tool_argument_overrides, tool_bindings)
    for tool in openai_tools:
        bind(tool, tool_lookup[tool["function"]["name"]])
    # Search is the turn's own tool, never bound
    if discovery_open:
        _open_discovery(openai_tools, tool_lookup)

    # Pre-load the tools the recent conversation asks for, so the bot can call
    # them this round instead of searching first
    recent = _recent_spoken_text(transcript, _PRELOAD_SEARCH_LINES)
    found_note = None
    if recent:
        found, _added = await _discover(recent, openai_tools, tool_lookup, bind)
        logger.info(f"Round search for {recent[:200]!r}: {found}")
        found_note = {'role': 'system', 'content': (
            f"Tools that fit what was just said, in your tools this round: {', '.join(found)}."
            if found else
            "No tool on the user's search path fits what was just said. If they asked "
            "for a tool, say plainly that there is none; never offer one you have not seen."
        )}
    for query in searched_queries:
        found, _added = await _discover(query, openai_tools, tool_lookup, bind)
        logger.info(f"Round search for {query[:200]!r}: {found}")
    return openai_tools, tool_lookup, tool_bindings, bind, found_note


async def _discover(
    value: str,
    openai_tools: List[OpenAITool],
    tool_lookup: Dict[str, ToolLookupInfo],
    bind: ToolBinder,
) -> Tuple[List[str], List[str]]:
    """Run /search and add newly discovered tools to this model turn.

    Returns the names of every callable tool found and a "name: description"
    line for each one newly added. Each added tool is bound first, so the
    model never sees or fills the parameters this turn supplies.
    """
    command = "/search"
    results = await _search_rows(value)
    logger.info(
        f"{command} {value[:200]!r} returned "
        f"{len(results) if isinstance(results, list) else type(results).__name__} rows:\n"
        f"{format_json_log(results)}"
    )
    if not isinstance(results, list):
        raise TypeError(f"{command} returned {type(results).__name__}, expected a list")

    # Rows the bot may not call (disconnected, Robot folders) count as nothing
    discovered_tools, discovered_lookup = convert_discovery_rows(results)
    added: List[str] = []
    for tool in discovered_tools:
        name = tool["function"]["name"]
        if name in tool_lookup:
            logger.info(f"{command}: skip {name!r}: already loaded this reply")
            continue
        bind(tool, discovered_lookup[name])
        openai_tools.append(tool)
        added.append(
            f"{name}: {tool['function'].get('description', '')}".rstrip()
        )
    for name, lookup in discovered_lookup.items():
        if name not in tool_lookup:
            tool_lookup[name] = lookup
    # Search never hands out Robot rows, but a match on a zone tool the bot
    # already holds is still a match
    matched = {_signature_name(row["tool"]) for row in results if row.get("tool")}
    held = [name for name, lookup in tool_lookup.items()
            if lookup["functionName"] in matched and name not in discovered_lookup]
    return [tool["function"]["name"] for tool in discovered_tools] + held, added


async def _execute_discovery_tool(
    tool_key: str,
    arguments: Dict[str, Any],
    openai_tools: List[OpenAITool],
    tool_lookup: Dict[str, ToolLookupInfo],
    bind: ToolBinder,
) -> str:
    """The bot's search tool: run it and say what it added to this turn."""
    if tool_key != "search":
        raise ValueError(f"Unsupported discovery tool: {tool_key!r}")
    argument_name = "query"
    value = str(arguments.get(argument_name) or "").strip()
    if not value:
        raise ValueError(f"{tool_key} requires {argument_name!r}")

    found, added = await _discover(value, openai_tools, tool_lookup, bind)
    if not found:
        return (
            f"Nothing on your search path provides {value!r}; tools outside it "
            f"may still exist. Do not retry with synonyms."
        )
    if not added:
        return f"{tool_key} found tools for {value!r}, but they were already loaded."
    return "Added tools to this turn:\n" + "\n".join(f"- {item}" for item in added)


@visible
async def execute_tool(search_term: str, arguments: Dict[str, Any] = {}) -> Any:
    """silent wrapper around client_command; search_term must already be anchored"""

    logger.info(f"TOOL: searchTerm='{search_term}' args={format_json_log(arguments)}")

    if search_term[:1] not in ('%', '$', '~', '@'):
        raise ValueError(f"Tool search term is not anchored: {search_term!r}")

    t0 = _t.monotonic()
    await atlantis.client_command("/silent on")
    try:
        tool_result = await atlantis.client_command(search_term, data=arguments)
    finally:
        await atlantis.client_command("/silent off")

    logger.info(f"TOOL {search_term} returned in {_t.monotonic() - t0:.2f}s: {str(tool_result)[:200]}")
    await atlantis.tool_result(search_term, tool_result)

    return tool_result


def _parse_tool_arguments(raw_args: str, tool_key: str) -> Dict[str, Any]:
    """Parse tool arguments JSON"""
    if not raw_args:
        return {}
    try:
        return json.loads(raw_args)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON for {tool_key}, attempting repair: {e}")
        repaired = _repair_json(raw_args)
        if repaired is not None:
            return repaired
        raise ValueError(f"Could not parse tool arguments as JSON: {e}")





@visible
async def run_turn(
    *,
    bot_sid: str,
    transcript: List[Dict[str, Any]],
    game_name: str,
    game_key: str,
    roster_names: Dict[str, str],
    system_prompt: Optional[str] = None,
    tools: Optional[List[AtlantisSearchToolT]] = None,
    tool_argument_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """Run a streaming tool-calling turn for a bot that has joined this game."""
    from .game import require_membership

    cfg = load_bot(game_name, game_key, bot_sid)
    if system_prompt is None:
        system_prompt = render_bot_prompt(game_name, game_key, bot_sid, roster_names)

    api_key_env = cfg["apiKeyEnv"]
    api_key = _chat_env(api_key_env) if api_key_env else ""
    base_url = cfg["baseUrl"] or None
    model = cfg["model"]
    routing = cfg["routing"]
    bot_display_name = str(roster_names.get(bot_sid) or "").strip()
    if not bot_display_name:
        raise ValueError(f"Bot {bot_sid!r} has no roster name in game {game_key!r}")

    if not api_key or not model:
        raise ValueError(f"Bot {bot_sid} missing model/api key: set {api_key_env} in Chat/.env")

    await silently(atlantis.client_log(
        f"run_turn start: bot_sid={bot_sid!r} display={bot_display_name!r} "
        f"model={model!r} transcript={len(transcript)} tools={len(tools or [])}"
    ))
    # Async, so the event loop keeps running while the model streams: other
    # tool calls are served, and an interrupt's notifications/cancelled can
    # cancel this task at any await, including the wait for the next chunk
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    stream_talk_id = None
    stream_think_id = None
    max_turns = 10
    discovery_calls = 0
    discovery_open = True
    # The bot's own searches this turn, rerun every round to rebuild its tools
    searched_queries: List[str] = []
    accumulated_text = ""
    robot = _game_robot(game_name)

    try:
        for turn_count in range(1, max_turns + 1):
            logger.info(f"=== TURN {turn_count}/{max_turns} === session_key={atlantis.get_session_key()}")
            # The last round must answer in text: tool results requested here
            # would never be read back by the model.
            final_round = turn_count == max_turns

            # Tools and status are rebuilt every round: the last round's tool
            # calls may have changed what exists and where the game stands
            openai_tools, tool_lookup, tool_bindings, bind, found_note = await _build_round_tools(
                robot=robot,
                tools=tools or [],
                tool_argument_overrides=tool_argument_overrides or {},
                transcript=transcript,
                searched_queries=searched_queries,
                discovery_open=discovery_open,
            )
            status_message = await _robot_status(game_name, game_key, robot)
            status_messages = [m for m in (status_message, found_note) if m]

            api_messages: List[Dict[str, Any]] = [
                {'role': 'system', 'content': system_prompt}
            ] + status_messages + transcript

            logger.info(f"Sending to {model}: {len(api_messages)} messages, {len(openai_tools)} tools")
            await silently(atlantis.client_log(
                f"run_turn api call: model={model!r} messages={len(api_messages)} tools={len(openai_tools)}"
            ))

            api_payload = {
                "model": model,
                "messages": api_messages,
                "tools": openai_tools,
                "provider": routing,
                "turn": turn_count,
            }
            try:
                _write_json(
                    os.path.join(os.path.dirname(__file__), "api_payload.json"),
                    api_payload,
                )
                _write_json(
                    _openrouter_payload_path(game_name, game_key, bot_sid),
                    api_payload,
                )
            except Exception as e:
                logger.exception("Failed to write API payload")
                await atlantis.client_warning(
                    f"Failed to write API payload for {bot_sid!r}: {type(e).__name__}: {e}",
                    level="ERROR",
                )

            # Tools this round offers the model, next to the transcript dumps
            tools_dump_file = os.path.join(require_membership(game_name, game_key), "tools.json")
            try:
                _write_json(tools_dump_file, {
                    "bot_sid": bot_sid,
                    "turn": turn_count,
                    "tools": openai_tools,
                    "tool_lookup": tool_lookup,
                })
            except Exception as e:
                logger.exception(f"Failed to write tool list to {tools_dump_file}")
                await atlantis.client_warning(
                    f"Failed to write tool list to {tools_dump_file}: {type(e).__name__}: {e}",
                    level="ERROR",
                )

            # Call LLM
            tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
            streamed_count = 0
            accumulated_text = ""

            t_api = _t.monotonic()
            stream = await client.chat.completions.create(
                model=model,
                messages=cast(Any, api_messages),
                # Chat's OpenAITool is the SDK's function tool with type typed as str
                tools=cast(List[ChatCompletionFunctionToolParam], openai_tools) if openai_tools else omit,
                tool_choice=("none" if final_round else "auto") if openai_tools else omit,
                stream=True,
                max_tokens=16000,
                stream_options={"include_usage": True},
                extra_body={"reasoning": {"effort": "low"}, "provider": routing},
            )
            logger.info(f"Stream opened in {_t.monotonic() - t_api:.2f}s")
            await silently(atlantis.client_log(f"run_turn stream opened: model={model!r}"))

            # What OpenRouter reports back: which generation, model and
            # provider served this round, and the usage/cost chunk at the end
            generation_id: Optional[str] = None
            served_model: Optional[str] = None
            served_provider: Optional[str] = None
            usage: Optional[Dict[str, Any]] = None
            stream_aborted = False

            async for chunk in stream:
                generation_id = generation_id or chunk.id
                served_model = served_model or chunk.model
                served_provider = served_provider or getattr(chunk, "provider", None)
                if chunk.usage is not None:
                    usage = chunk.usage.model_dump()

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Thinking content
                reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                if reasoning:
                    if not stream_think_id:
                        stream_think_id = await atlantis.stream_start(bot_sid, f"{bot_display_name} (thinking)")
                    await atlantis.stream(reasoning, stream_think_id)

                # Text content
                if delta.content:
                    if stream_think_id:
                        await atlantis.stream_end(stream_think_id)
                        stream_think_id = None

                    if not stream_talk_id:
                        stream_talk_id = await atlantis.stream_start(bot_sid, bot_display_name)

                    text = delta.content.lstrip() if streamed_count == 0 else delta.content
                    if text:
                        await atlantis.stream(text, stream_talk_id)
                        streamed_count += 1
                        accumulated_text += text

                        if streamed_count >= 512:
                            logger.warning("Aborting stream — chunk limit reached")
                            stream_aborted = True
                            await stream.close()
                            break

                # Tool call fragments
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        acc = tool_calls_accumulator.setdefault(tc.index, {'id': '', 'name': '', 'arguments': ''})
                        if tc.id:
                            acc['id'] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc['name'] += tc.function.name
                            if tc.function.arguments:
                                acc['arguments'] += tc.function.arguments

            logger.info(f"Stream done: turn={turn_count} chunks={streamed_count} tool_calls={len(tool_calls_accumulator)}")
            await silently(atlantis.client_log(
                f"run_turn stream done: turn={turn_count} chunks={streamed_count} "
                f"tool_calls={len(tool_calls_accumulator)}"
            ))

            await _record_telemetry(
                game_dir=require_membership(game_name, game_key),
                record={
                    "ts": _t.time(),
                    "bot_sid": bot_sid,
                    "turn": turn_count,
                    "generation_id": generation_id,
                    "requested_model": model,
                    "model": served_model,
                    "provider": served_provider,
                    "usage": usage,
                    "latency_s": round(_t.monotonic() - t_api, 3),
                    "stream_aborted": stream_aborted,
                },
            )

            # Stop when no tools are requested
            if not tool_calls_accumulator:
                break
            if final_round:
                raise RuntimeError(
                    f"{model} requested tools on the final round despite tool_choice='none': "
                    f"{[tc['name'] for tc in tool_calls_accumulator.values()]}"
                )

            # Close streams before tools
            await _close_streams(stream_talk_id, stream_think_id)
            stream_talk_id = None
            stream_think_id = None

            # Record assistant tool calls
            transcript.append({
                'role': 'assistant',
                'content': accumulated_text or None,
                'tool_calls': [
                    {'id': tc['id'], 'type': 'function', 'function': {'name': tc['name'], 'arguments': tc['arguments']}}
                    for tc in tool_calls_accumulator.values()
                ]
            })

            # Execute tool calls
            any_executed = False
            for tc in tool_calls_accumulator.values():
                try:
                    tool_key = tc['name']
                    if tool_key not in tool_lookup:
                        # Includes search after discovery closes mid-batch.
                        tool_result = (
                            f"No tool named {tool_key!r}, nothing was executed. "
                            f"Call one of these by its exact name: {', '.join(tool_lookup)}"
                        )
                        logger.error(f"Model requested undeclared tool: {tool_key}")
                        await atlantis.client_warning(
                            f"{bot_display_name} called undeclared tool {tool_key}, nothing was executed",
                            level="ERROR",
                        )
                        await atlantis.tool_result(tool_key, tool_result)
                        transcript.append({
                            'role': 'tool',
                            'tool_call_id': tc['id'],
                            'content': tool_result,
                        })
                        any_executed = True
                        continue
                    lookup_info = tool_lookup[tool_key]
                    search_term = lookup_info['searchTerm']
                    arguments = _parse_tool_arguments(tc['arguments'], tool_key)
                    # Values bound this round replace anything the model sent
                    arguments.update(tool_bindings.get(tool_key, {}))

                    # Coerce args to match schema types
                    for ot in openai_tools:
                        if ot['function']['name'] == tool_key:
                            schema = ot['function']['parameters']
                            if schema and arguments:
                                arguments = coerce_args_to_schema(arguments, schema)
                            break

                    if tool_key in _DISCOVERY_TOOLS:
                        discovery_calls += 1
                        if "query" not in arguments:
                            logger.error(f"{tool_key} called without 'query': {arguments!r}")
                            await atlantis.client_warning(
                                f"{bot_display_name} called {tool_key} "
                                f"({discovery_calls}/{_MAX_DISCOVERY_CALLS}) with no 'query' "
                                "argument, nothing was searched",
                                level="ERROR",
                            )
                            await silently(atlantis.client_data(f"{tool_key} arguments", arguments))
                            tool_result = (
                                f"{tool_key} needs a 'query' string, nothing was searched."
                            )
                            await atlantis.tool_result(search_term, tool_result)
                        else:
                            # Warnings show through silent mode, so this is a plain log.
                            await silently(
                                atlantis.client_log(
                                    f"{bot_display_name} is consulting {tool_key} "
                                    f"({discovery_calls}/{_MAX_DISCOVERY_CALLS})",
                                ),
                                atlantis.client_data(f"{tool_key} arguments", arguments),
                            )
                            tool_result = await _execute_discovery_tool(
                                tool_key,
                                arguments,
                                openai_tools,
                                tool_lookup,
                                bind,
                            )
                            searched_queries.append(str(arguments["query"]).strip())
                        if discovery_calls >= _MAX_DISCOVERY_CALLS:
                            discovery_open = False
                            _close_discovery(openai_tools, tool_lookup)
                            tool_result = (
                                f"{tool_result}\n\nTool discovery is now closed for "
                                f"this reply. Answer with what you have."
                            )
                            logger.info(
                                f"Discovery closed after {discovery_calls} calls"
                            )
                    else:
                        try:
                            tool_result = await execute_tool(
                                search_term=search_term,
                                arguments=arguments,
                            )
                        except McpError as e:
                            # The tool ran and reported failure: the bot gets its
                            # message as the result so it can address it. Our own bugs still raise.
                            tool_result = e.error.message
                            logger.error(f"Tool {tool_key} reported failure: {tool_result}")
                            await atlantis.client_warning(
                                f"{bot_display_name}'s call to {tool_key} failed: {tool_result}",
                                level="ERROR",
                            )
                            await atlantis.tool_result(search_term, tool_result)
                    transcript.append({
                        'role': 'tool',
                        'tool_call_id': tc['id'],
                        'content': str(tool_result) if tool_result else "No result"
                    })
                    any_executed = True
                except Exception as e:
                    logger.error(f"Tool {tc['name']} failed: {e}")
                    await atlantis.client_warning(
                        f"{bot_display_name}'s turn aborted, tool {tc['name']} failed: "
                        f"{type(e).__name__}: {e}",
                        level="ERROR",
                    )
                    raise RuntimeError(f"Tool call failed: {tc['name']} — {e}") from e

            if not any_executed:
                break

    finally:
        try:
            await _close_streams(stream_talk_id, stream_think_id)
        finally:
            # Also drops a stream left open by a cancelled or failed round
            await client.close()

    # Only a turn that actually said something counts as the bot having been there
    if accumulated_text:
        await _robot_after_turn(game_name, game_key, robot)

    await silently(atlantis.client_log(
        f"run_turn done: bot_sid={bot_sid!r} chars={len(accumulated_text or '')}"
    ))
    return accumulated_text or None

# Keep bot_turn as an alias for backward compat
bot_turn = run_turn
