import atlantis
import json
import os
import time as _t

from openai import OpenAI
from typing import List, Dict, Any, Optional, cast

from dynamic_functions.Home.bot_common import (
    logger,
    TranscriptToolT, ToolLookupInfo, ToolSchemaT,
    _repair_json, coerce_args_to_schema, convert_tools_for_llm,
    handle_dir_tool, handle_search_tool,
)
from dynamic_functions.Data.todo import handle_todo_tool
from utils import format_json_log


async def _close_streams(talk_id, think_id):
    """Close any open stream IDs, ignoring errors."""
    for sid in [think_id, talk_id]:
        if sid:
            try:
                await atlantis.stream_end(sid)
            except Exception as e:
                logger.warning(f"Failed to close stream {sid}: {e}")


async def _execute_tool(
    tool_key: str,
    arguments: Dict[str, Any],
    call_id: str,
    converted_tools: List[TranscriptToolT],
    tool_lookup: Dict[str, ToolLookupInfo],
    transcript: List[Dict[str, Any]],
    allowed_apps: Optional[List[str]] = None,
) -> bool:
    """Execute a single tool call and append result to transcript. Returns True if handled."""

    # Pseudo-tools
    if tool_key == 'dir':
        name = arguments.get('name', '')
        logger.info(f"DIR: name='{name}'")
        summary, _, _ = await handle_dir_tool(name, converted_tools, tool_lookup)
        transcript.append({'role': 'tool', 'tool_call_id': call_id, 'content': summary})
        return True

    if tool_key == 'search':
        query = arguments.get('query', '')
        logger.info(f"SEARCH: query='{query}'")
        summary, _, _ = await handle_search_tool(query, converted_tools, tool_lookup, allowed_apps=allowed_apps)
        transcript.append({'role': 'tool', 'tool_call_id': call_id, 'content': summary})
        return True

    if tool_key == 'todo':
        result = await handle_todo_tool(arguments)
        transcript.append({'role': 'tool', 'tool_call_id': call_id, 'content': result})
        return True

    # Real tool via lookup
    if tool_key not in tool_lookup:
        raise ValueError(f"Unknown tool: {tool_key} (available: {list(tool_lookup.keys())})")

    lookup_info = tool_lookup[tool_key]
    search_term = lookup_info['searchTerm']
    function_name = lookup_info['functionName']

    logger.info(f"TOOL: {tool_key} searchTerm='{search_term}' args={format_json_log(arguments)}")

    # Coerce args to match schema
    for ct in converted_tools:
        if ct['function']['name'] == tool_key:
            tool_schema = ct['function']['parameters']
            if tool_schema and arguments:
                arguments = coerce_args_to_schema(arguments, tool_schema)
            break

    t0 = _t.monotonic()
    await atlantis.client_command("/silent on")
    tool_result = await atlantis.client_command(f"%{search_term}", data=arguments)
    await atlantis.client_command("/silent off")

    logger.info(f"TOOL {tool_key} returned in {_t.monotonic() - t0:.2f}s: {str(tool_result)[:200]}")
    await atlantis.tool_result(function_name, tool_result)

    transcript.append({
        'role': 'tool',
        'tool_call_id': call_id,
        'content': str(tool_result) if tool_result else "No result"
    })
    return True


def _parse_tool_arguments(raw_args: str, tool_key: str) -> Dict[str, Any]:
    """Parse tool call arguments JSON, attempting repair if needed."""
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
    client: OpenAI,
    model: str,
    bot_sid: str,
    bot_display_name: str,
    system_prompt: str,
    transcript: List[Dict[str, Any]],
    converted_tools: List[TranscriptToolT],
    tool_lookup: Dict[str, ToolLookupInfo],
    sessionId: str,
    requestId: str,
    allowed_apps: Optional[List[str]] = None,
) -> Optional[str]:
    """Stream a multi-turn LLM conversation with tool calls. Returns accumulated text."""

    stream_talk_id = None
    stream_think_id = None
    max_turns = 10
    accumulated_text = ""

    try:
        for turn_count in range(1, max_turns + 1):
            logger.info(f"=== TURN {turn_count}/{max_turns} === session={sessionId}")

            api_messages: List[Dict[str, Any]] = [
                {'role': 'system', 'content': system_prompt}
            ] + transcript

            logger.info(f"Sending to {model}: {len(api_messages)} messages, {len(converted_tools)} tools")

            # Debug dump
            api_dump_file = os.path.join(os.path.dirname(__file__), 'api_payload.json')
            try:
                with open(api_dump_file, 'w') as f:
                    json.dump({'model': model, 'messages': api_messages, 'tools': converted_tools, 'turn': turn_count}, f, indent=2, default=str)
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

            # No tool calls — we're done
            if not tool_calls_accumulator:
                break

            # Close streams before tool execution
            await _close_streams(stream_talk_id, stream_think_id)
            stream_talk_id = None
            stream_think_id = None

            # Record assistant message with tool calls
            transcript.append({
                'role': 'assistant',
                'content': accumulated_text or None,
                'tool_calls': [
                    {'id': tc['id'], 'type': 'function', 'function': {'name': tc['name'], 'arguments': tc['arguments']}}
                    for tc in tool_calls_accumulator.values()
                ]
            })

            # Execute each tool call
            any_executed = False
            for tc in tool_calls_accumulator.values():
                try:
                    arguments = _parse_tool_arguments(tc['arguments'], tc['name'])
                    await _execute_tool(tc['name'], arguments, tc['id'], converted_tools, tool_lookup, transcript, allowed_apps=allowed_apps)
                    any_executed = True
                except Exception as e:
                    logger.error(f"Tool {tc['name']} failed: {e}")
                    raise RuntimeError(f"Tool call failed: {tc['name']} — {e}") from e

            if not any_executed:
                break

    finally:
        await _close_streams(stream_talk_id, stream_think_id)

    return accumulated_text or None
