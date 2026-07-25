import atlantis
import json
import os
import time as _t

from openai import OpenAI
from typing import List, Dict, Any, Optional, cast

from .bot import bot_roster_name, load_bot, render_bot_prompt
from .tool import (
    logger,
    AtlantisSearchToolT, OpenAITool, ToolLookupInfo,
    _repair_json, coerce_args_to_schema, convert_search_tools,
)
from utils import format_json_log


async def _close_streams(talk_id, think_id):
    """Close open stream IDs"""
    for sid in [think_id, talk_id]:
        if sid:
            try:
                await atlantis.stream_end(sid)
            except Exception as e:
                logger.warning(f"Failed to close stream {sid}: {e}")

@visible
async def execute_tool(search_term: str, arguments: Dict[str, Any] = {}) -> Any:
    """silent wrapper around client_command"""

    logger.info(f"TOOL: searchTerm='{search_term}' args={format_json_log(arguments)}")

    t0 = _t.monotonic()
    await atlantis.client_command("/silent on")
    tool_result = await atlantis.client_command(f"%{search_term}", data=arguments)
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
    system_prompt: Optional[str] = None,
    roster_names: Optional[Dict[str, str]] = None,
    tools: Optional[List[AtlantisSearchToolT]] = None,
) -> Optional[str]:
    """Run a streaming tool-calling turn. Loads bot config from bot_sid."""
    cfg = load_bot(bot_sid)
    if system_prompt is None:
        system_prompt = render_bot_prompt(bot_sid, roster_names)

    api_key_env = cfg["apiKeyEnv"]
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    base_url = cfg["baseUrl"] or None
    model = cfg["model"]
    bot_display_name = bot_roster_name(bot_sid, roster_names)

    if not api_key or not model:
        raise ValueError(f"Bot {bot_sid} missing model/api key (env={api_key_env})")

    await atlantis.client_log(
        f"run_turn start: bot_sid={bot_sid!r} display={bot_display_name!r} "
        f"model={model!r} transcript={len(transcript)} tools={len(tools or [])}"
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    openai_tools, tool_lookup = convert_search_tools(tools or [])
    stream_talk_id = None
    stream_think_id = None
    max_turns = 10
    accumulated_text = ""

    try:
        for turn_count in range(1, max_turns + 1):
            logger.info(f"=== TURN {turn_count}/{max_turns} === session_key={atlantis.get_session_key()}")

            api_messages: List[Dict[str, Any]] = [
                {'role': 'system', 'content': system_prompt}
            ] + transcript

            logger.info(f"Sending to {model}: {len(api_messages)} messages, {len(openai_tools)} tools")
            await atlantis.client_log(
                f"run_turn api call: model={model!r} messages={len(api_messages)} tools={len(openai_tools)}"
            )

            api_dump_file = os.path.join(os.path.dirname(__file__), "api_payload.json")
            try:
                with open(api_dump_file, "w") as f:
                    json.dump(
                        {
                            "model": model,
                            "messages": api_messages,
                            "tools": openai_tools,
                            "turn": turn_count,
                        },
                        f,
                        indent=2,
                        default=str,
                    )
            except Exception as e:
                logger.warning(f"Failed to write API payload: {e}")

            # Call LLM
            tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
            streamed_count = 0
            accumulated_text = ""

            t_api = _t.monotonic()
            stream = client.chat.completions.create(
                model=model,
                messages=cast(Any, api_messages),
                tools=openai_tools if openai_tools else None,  # type: ignore[arg-type]
                tool_choice=cast(Any, "auto" if openai_tools else None),
                stream=True,
                max_tokens=16000,
                extra_body={"reasoning": {"effort": "low"}},
            )
            logger.info(f"Stream opened in {_t.monotonic() - t_api:.2f}s")
            await atlantis.client_log(f"run_turn stream opened: model={model!r}")

            for chunk in stream:
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
            await atlantis.client_log(
                f"run_turn stream done: turn={turn_count} chunks={streamed_count} "
                f"tool_calls={len(tool_calls_accumulator)}"
            )

            # Stop when no tools are requested
            if not tool_calls_accumulator:
                break

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
                    lookup_info = tool_lookup[tool_key]
                    search_term = lookup_info['searchTerm']
                    arguments = _parse_tool_arguments(tc['arguments'], tool_key)

                    # Coerce args to match schema types
                    for ot in openai_tools:
                        if ot['function']['name'] == tool_key:
                            schema = ot['function']['parameters']
                            if schema and arguments:
                                arguments = coerce_args_to_schema(arguments, schema)
                            break

                    tool_result = await execute_tool(
                        search_term=search_term,
                        arguments=arguments,
                    )
                    transcript.append({
                        'role': 'tool',
                        'tool_call_id': tc['id'],
                        'content': str(tool_result) if tool_result else "No result"
                    })
                    any_executed = True
                except Exception as e:
                    logger.error(f"Tool {tc['name']} failed: {e}")
                    raise RuntimeError(f"Tool call failed: {tc['name']} — {e}") from e

            if not any_executed:
                break

    finally:
        await _close_streams(stream_talk_id, stream_think_id)

    await atlantis.client_log(
        f"run_turn done: bot_sid={bot_sid!r} chars={len(accumulated_text or '')}"
    )
    return accumulated_text or None

# Keep bot_turn as an alias for backward compat
bot_turn = run_turn
