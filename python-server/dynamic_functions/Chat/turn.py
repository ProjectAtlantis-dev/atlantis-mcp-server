import atlantis
import copy
import json
import os
import time as _t

from dotenv import dotenv_values
from mcp.shared.exceptions import McpError
from openai import OpenAI, omit
from openai.types.chat import ChatCompletionFunctionToolParam
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple, cast

from .bot import load_bot, render_bot_prompt
from .common import _read_json, _write_json, silent_command, silently
from .runner import _game_init_script
from .tool import (
    logger,
    AtlantisSearchToolT, OpenAITool, ToolLookupInfo,
    _repair_json, coerce_args_to_schema, convert_discovery_rows, convert_folder_rows,
    convert_search_tools,
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


# A zone's own bot tools live in its app's Robot/Tools folder; the app folder
# is named for the game. Every tool there is loaded into every bot round.
_ZONE_TOOLS_FOLDER = "Robot/Tools"


async def _zone_tools(game_name: str) -> Tuple[List[OpenAITool], Dict[str, ToolLookupInfo]]:
    """The zone's bot tools, listed fresh: a game without the folder has none."""
    folder_term = f"${game_name}/{_ZONE_TOOLS_FOLDER}"
    rows = await silent_command("/ls", {"searchTerm": f"{folder_term}/*"})
    if not isinstance(rows, list):
        raise TypeError(f"/ls {folder_term}/* returned {type(rows).__name__}, expected a list")
    return convert_folder_rows(rows, folder_term)


async def _game_status(game_name: str) -> Optional[Dict[str, str]]:
    """Run the game's status tool so every turn starts from fresh state,
    instead of trusting the bot to ask for it.

    A game names its tool as STATUS_TOOL in <game_name>/init.py (by convention
    the get_status in its app's Robot folder). It runs every round, so it must
    only read state; recording what the bot did belongs in AFTER_TURN_TOOL.
    A game without one is skipped.
    """
    status_tool = getattr(_game_init_script(game_name), "STATUS_TOOL", None)
    if not status_tool:
        logger.warning(f"Game {game_name!r} has no STATUS_TOOL in its init.py; turn runs without status")
        return None
    status = await silent_command(status_tool)
    if not isinstance(status, str):
        raise TypeError(f"{status_tool} must return a string, got {type(status).__name__}")
    return {'role': 'system', 'content': f"Current status, fetched for this turn by {status_tool}:\n{status}"}


async def _game_after_turn(game_name: str) -> None:
    """Run the game's AFTER_TURN_TOOL once a bot's reply has gone out, so the
    game records that turn (e.g. that the bot has now met the caller).

    Named in <game_name>/init.py like STATUS_TOOL. A game without one is skipped.
    """
    after_turn_tool = getattr(_game_init_script(game_name), "AFTER_TURN_TOOL", None)
    if not after_turn_tool:
        logger.warning(f"Game {game_name!r} has no AFTER_TURN_TOOL in its init.py; turn is not recorded")
        return
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
    game_name: str,
    tools: List[AtlantisSearchToolT],
    tool_argument_overrides: Dict[str, Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    searched_queries: List[str],
    discovery_open: bool,
) -> Tuple[List[OpenAITool], Dict[str, ToolLookupInfo], Dict[str, Dict[str, Any]], ToolBinder]:
    """One round's tool list, rebuilt from scratch so a tool run last round
    cannot leave it stale: the bot's inventory, bound to the current cursor,
    plus what the recent conversation asks for and every search the bot has
    made this turn, rerun against the system as it is now. The bot's own
    tools come first, then its zone's, then search."""
    openai_tools, tool_lookup = convert_search_tools(tools)
    zone_tools, zone_lookup = await _zone_tools(game_name)
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
    for query in ([recent] if recent else []) + searched_queries:
        loaded = await _execute_discovery_tool(
            "search", {"query": query}, openai_tools, tool_lookup, bind,
        )
        logger.info(f"Round search for {query[:200]!r}: {loaded}")
    return openai_tools, tool_lookup, tool_bindings, bind


async def _execute_discovery_tool(
    tool_key: str,
    arguments: Dict[str, Any],
    openai_tools: List[OpenAITool],
    tool_lookup: Dict[str, ToolLookupInfo],
    bind: ToolBinder,
) -> str:
    """Run search and add newly discovered tools to this model turn.

    Each added tool is bound first, so the model never sees or fills the
    parameters this turn supplies.
    """
    if tool_key != "search":
        raise ValueError(f"Unsupported discovery tool: {tool_key!r}")

    argument_name = "query"
    command = "/search"
    value = str(arguments.get(argument_name) or "").strip()
    if not value:
        raise ValueError(f"{tool_key} requires {argument_name!r}")

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
    if not discovered_tools:
        return (
            f"Nothing on your search path provides {value!r}; tools outside it "
            f"may still exist. Do not retry with synonyms."
        )
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
    client = OpenAI(api_key=api_key, base_url=base_url)
    stream_talk_id = None
    stream_think_id = None
    max_turns = 10
    discovery_calls = 0
    discovery_open = True
    # The bot's own searches this turn, rerun every round to rebuild its tools
    searched_queries: List[str] = []
    accumulated_text = ""

    try:
        for turn_count in range(1, max_turns + 1):
            logger.info(f"=== TURN {turn_count}/{max_turns} === session_key={atlantis.get_session_key()}")
            # The last round must answer in text: tool results requested here
            # would never be read back by the model.
            final_round = turn_count == max_turns

            # Tools and status are rebuilt every round: the last round's tool
            # calls may have changed what exists and where the game stands
            openai_tools, tool_lookup, tool_bindings, bind = await _build_round_tools(
                game_name=game_name,
                tools=tools or [],
                tool_argument_overrides=tool_argument_overrides or {},
                transcript=transcript,
                searched_queries=searched_queries,
                discovery_open=discovery_open,
            )
            status_message = await _game_status(game_name)
            status_messages = [status_message] if status_message else []

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
            stream = client.chat.completions.create(
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

            for chunk in stream:
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
        await _close_streams(stream_talk_id, stream_think_id)

    # Only a turn that actually said something counts as the bot having been there
    if accumulated_text:
        await _game_after_turn(game_name)

    await silently(atlantis.client_log(
        f"run_turn done: bot_sid={bot_sid!r} chars={len(accumulated_text or '')}"
    ))
    return accumulated_text or None

# Keep bot_turn as an alias for backward compat
bot_turn = run_turn
