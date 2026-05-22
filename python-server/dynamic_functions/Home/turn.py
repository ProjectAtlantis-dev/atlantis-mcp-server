import atlantis
import json
import os
import time as _t

from openai import OpenAI
from typing import List, Dict, Any, Optional, cast

from dynamic_functions.Home.chat_common import (
    logger,
    OpenAITool, ToolLookupInfo,
    _repair_json, coerce_args_to_schema,
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
    """Execute a tool call via Atlantis client_command."""

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

async def run_turn(
    *,
    api_key: str,
    base_url: Optional[str],
    model: str,
    bot_sid: str,
    bot_display_name: str,
    system_prompt: str,
    transcript: List[Dict[str, Any]],
    tools: List[str] = [],
) -> Optional[str]:
    """Run a streaming tool-calling turn"""

    client = OpenAI(api_key=api_key, base_url=base_url)
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

            logger.info(f"Sending to {model}: {len(api_messages)} messages, {len(converted_tools)} tools")

            # Call LLM
            tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
            streamed_count = 0
            accumulated_text = ""

            t_api = _t.monotonic()
            stream = client.chat.completions.create(
                model=model,
                messages=cast(Any, api_messages),
                tools=converted_tools if converted_tools else None,  # type: ignore[arg-type]
                tool_choice=cast(Any, "auto" if converted_tools else None),
                stream=True,
                max_tokens=16000,
                extra_body={"reasoning": {"effort": "low"}},
            )
            logger.info(f"Stream opened in {_t.monotonic() - t_api:.2f}s")

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
                    arguments = _parse_tool_arguments(tc['arguments'], tool_key)

                    # Coerce args to match schema types
                    for ct in converted_tools:
                        if ct['function']['name'] == tool_key:
                            schema = ct['function']['parameters']
                            if schema and arguments:
                                arguments = coerce_args_to_schema(arguments, schema)
                            break

                    tool_result = await execute_tool(
                        search_term=lookup_info['searchTerm'],
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

    return accumulated_text or None

@visible
async def bot_turn(
    *,
    bot_sid: str,
    system_prompt: str,
    transcript: List[Dict[str, Any]],  # Atlantis chat transcript (not OpenAI messages)
    tools: List[str] = [],             # Atlantis search terms, e.g. ["**Home**bot_list"]
) -> Optional[str]:
    """High-level wrapper: loads bot config for bot_sid and delegates to run_turn."""
    from dynamic_functions.Home.common import _load_bot_config

    loaded = _load_bot_config(bot_sid)
    if not loaded:
        raise ValueError(f"No bot config for bot {bot_sid}")
    cfg, _folder = loaded

    api_key_env = cfg.get("apiKeyEnv", "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    base_url = cfg.get("baseUrl", "") or None
    model = cfg.get("model", "")
    bot_display_name = cfg.get("displayName", bot_sid)

    if not api_key or not model:
        raise ValueError(f"Bot {bot_sid} missing model/api key (env={api_key_env})")

    return await run_turn(
        api_key=api_key,
        base_url=base_url,
        model=model,
        bot_sid=bot_sid,
        bot_display_name=bot_display_name,
        system_prompt=system_prompt,
        transcript=transcript,
        tools=tools,
    )
