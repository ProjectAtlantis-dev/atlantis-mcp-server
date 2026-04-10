import atlantis
import json
import os
import time as _t

from openai import OpenAI
from typing import List, Dict, Any, Optional, cast

from dynamic_functions.Bot.Runtime.common import (
    logger,
    TranscriptToolT, ToolLookupInfo, ToolSchemaT,
    _repair_json, coerce_args_to_schema, convert_tools_for_llm,
    handle_dir_tool, handle_search_tool,
)
from dynamic_functions.Misc.todo import handle_todo_tool
from utils import format_json_log


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
) -> Optional[str]:
    """Stream a multi-turn LLM conversation with tool calls. Returns accumulated text."""

    streamTalkId = None
    streamThinkId = None
    max_turns = 10
    turn_count = 0
    accumulated_text = ""

    try:
        while turn_count < max_turns:
            turn_count += 1
            logger.info(f"=== TURN {turn_count}/{max_turns} === session={sessionId} request={requestId}")

            if turn_count == 1:
                await atlantis.owner_log(f"Attempting to call OpenRouter: {model}")

            # Build full message list: system prompt + transcript
            api_messages: List[Dict[str, Any]] = [
                {'role': 'system', 'content': system_prompt}
            ] + transcript

            logger.info(f"=== SENDING TO OPENROUTER (GLM) (turn {turn_count}) ===")
            logger.info(f"Messages: {len(api_messages)} entries")
            logger.info(f"Tools: {len(converted_tools)} entries")
            logger.info(f"Tool names: {[t['function']['name'] for t in converted_tools]}")

            # Dump full API payload for debugging (clobbers each turn)
            api_dump_file = os.path.join(os.path.dirname(__file__), 'api_payload.json')
            try:
                with open(api_dump_file, 'w') as f:
                    json.dump({
                        'model': model,
                        'messages': api_messages,
                        'tools': converted_tools,
                        'turn': turn_count,
                    }, f, indent=2, default=str)
                logger.info(f"API payload written to {api_dump_file}")
            except Exception as e:
                logger.warning(f"Failed to write API payload: {e}")

            event_count = 0
            streamed_count = 0
            max_stream_chunks = 512
            tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
            tool_call_made = False
            accumulated_text = ""

            logger.info(f">>> Calling OpenRouter ({model})...")
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

            logger.info(f"<<< OpenRouter stream opened in {_t.monotonic() - t_api:.2f}s, reading chunks...")
            if turn_count == 1:
                await atlantis.owner_log("OpenRouter API call successful, starting stream")

            for chunk in stream:
                event_count += 1

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Handle reasoning/thinking content
                reasoning_content = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                if reasoning_content:
                    if not streamThinkId:
                        streamThinkId = await atlantis.stream_start(bot_sid, f"{bot_display_name} (thinking)")
                        logger.info(f"Think stream started with ID: {streamThinkId}")
                    await atlantis.stream(reasoning_content, streamThinkId)

                # Handle text content
                if delta.content:
                    # Close thinking stream before streaming text
                    if streamThinkId:
                        await atlantis.stream_end(streamThinkId)
                        streamThinkId = None

                    if not streamTalkId:
                        streamTalkId = await atlantis.stream_start(bot_sid, bot_display_name)
                        logger.info(f"Talk stream started with ID: {streamTalkId}")

                    text = delta.content
                    content_to_send = text.lstrip() if streamed_count == 0 else text

                    if content_to_send:
                        await atlantis.stream(content_to_send, streamTalkId)
                        streamed_count += 1
                        accumulated_text += content_to_send

                        if streamed_count >= max_stream_chunks:
                            logger.warning(f"Aborting stream after {streamed_count} chunks")
                            break

                # Accumulate tool calls (streamed in fragments)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {'id': '', 'name': '', 'arguments': ''}
                        if tc_delta.id:
                            tool_calls_accumulator[idx]['id'] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_accumulator[idx]['name'] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_accumulator[idx]['arguments'] += tc_delta.function.arguments

            logger.info(f"Stream complete: turn={turn_count} events={event_count} text_chunks={streamed_count} tool_calls={len(tool_calls_accumulator)} session={sessionId}")

            # Execute accumulated tool calls if any
            if tool_calls_accumulator:
                # Close any open streams before executing tools so the next
                # turn's text starts a fresh chat bubble.
                if streamTalkId:
                    await atlantis.stream_end(streamTalkId)
                    streamTalkId = None
                if streamThinkId:
                    await atlantis.stream_end(streamThinkId)
                    streamThinkId = None

                logger.info(f"Executing {len(tool_calls_accumulator)} tool calls")

                # Add assistant message with tool_calls to transcript (OpenAI format)
                assistant_tool_calls = [
                    {
                        'id': tc['id'],
                        'type': 'function',
                        'function': {'name': tc['name'], 'arguments': tc['arguments']}
                    }
                    for tc in tool_calls_accumulator.values()
                ]
                transcript.append({
                    'role': 'assistant',
                    'content': accumulated_text or None,
                    'tool_calls': assistant_tool_calls
                })

                # Execute each tool and append result message
                for tc in tool_calls_accumulator.values():
                    call_id = tc['id']
                    tool_key = tc['name']

                    try:
                        arguments = json.loads(tc['arguments']) if tc['arguments'] else {}
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON for {tool_key}, attempting repair: {e}")
                        repaired = _repair_json(tc['arguments']) if tc['arguments'] else None
                        if repaired is not None:
                            logger.info(f"🔧 Repaired JSON for {tool_key}")
                            arguments = repaired
                        else:
                            logger.error(f"❌ JSON repair failed for {tool_key}: {e}")
                            transcript.append({
                                'role': 'tool',
                                'tool_call_id': call_id,
                                'content': f"ERROR: Could not parse your tool arguments as JSON: {e}. Please retry with valid JSON (double-quoted keys and strings)."
                            })
                            tool_call_made = True
                            continue

                    # Handle dir pseudo-tool
                    if tool_key == 'dir':
                        name = arguments.get('name', '')
                        logger.info(f"=== DIR TOOL CALLED: name='{name}' ===")
                        summary, converted_tools, tool_lookup = await handle_dir_tool(
                            name, converted_tools, tool_lookup
                        )
                        transcript.append({
                            'role': 'tool',
                            'tool_call_id': call_id,
                            'content': summary
                        })
                        tool_call_made = True
                        continue

                    # Handle search pseudo-tool
                    if tool_key == 'search':
                        query = arguments.get('query', '')
                        logger.info(f"=== SEARCH TOOL CALLED: query='{query}' ===")
                        summary, converted_tools, tool_lookup = await handle_search_tool(
                            query, converted_tools, tool_lookup
                        )
                        transcript.append({
                            'role': 'tool',
                            'tool_call_id': call_id,
                            'content': summary
                        })
                        tool_call_made = True
                        continue

                    # Handle todo pseudo-tool
                    if tool_key == 'todo':
                        logger.info(f"=== TODO TOOL CALLED ===")
                        result = await handle_todo_tool(arguments)
                        transcript.append({
                            'role': 'tool',
                            'tool_call_id': call_id,
                            'content': result
                        })
                        tool_call_made = True
                        continue

                    if tool_key not in tool_lookup:
                        logger.error(f"\x1b[91m🚨 UNKNOWN TOOL KEY: '{tool_key}' not in tool_lookup!\x1b[0m")
                        logger.error(f"\x1b[91m  Available keys: {list(tool_lookup.keys())}\x1b[0m")
                        transcript.append({
                            'role': 'tool',
                            'tool_call_id': call_id,
                            'content': f"Error: Unknown tool: {tool_key}"
                        })
                        continue

                    lookup_info = tool_lookup[tool_key]
                    search_term = lookup_info['searchTerm']
                    function_name = lookup_info['functionName']

                    logger.info(f">>> EXECUTING TOOL: {tool_key} (call_id={call_id})")
                    logger.info(f"    searchTerm='{search_term}' function='{function_name}'")
                    logger.info(f"    args={format_json_log(arguments)}")

                    try:
                        # Look up the tool's schema from converted_tools for argument coercion
                        tool_schema = None
                        for ct in converted_tools:
                            if ct['function']['name'] == tool_key:
                                tool_schema = ct['function']['parameters']
                                break

                        if tool_schema and arguments:
                            arguments = coerce_args_to_schema(arguments, tool_schema)
                            logger.info(f"    post-coercion args={format_json_log(arguments)}")

                        t0 = _t.monotonic()
                        logger.info("    /silent on")
                        await atlantis.client_command("/silent on")
                        logger.info(f"    %{search_term}")
                        tool_result = await atlantis.client_command(f"%{search_term}", data=arguments)
                        logger.info("    /silent off")
                        await atlantis.client_command("/silent off")
                        elapsed = _t.monotonic() - t0

                        logger.info(f"<<< TOOL {tool_key} RETURNED in {elapsed:.2f}s — result: {str(tool_result)[:200]}")
                        await atlantis.tool_result(function_name, tool_result)

                        transcript.append({
                            'role': 'tool',
                            'tool_call_id': call_id,
                            'content': str(tool_result) if tool_result else "No result"
                        })
                        tool_call_made = True

                    except Exception as e:
                        logger.error(f"<<< TOOL {tool_key} FAILED: {e}")
                        transcript.append({
                            'role': 'tool',
                            'tool_call_id': call_id,
                            'content': f"Error: {str(e)}"
                        })

                logger.info("Tool calls executed, continuing conversation")
                logger.info(f"tool_lookup keys ({len(tool_lookup)}): {list(tool_lookup.keys())}")

            # Exit if no tool calls were made
            if not tool_call_made:
                logger.info(f"No tool calls — conversation complete (session={sessionId})")
                break
            else:
                logger.info(f"Tool calls executed — continuing to turn {turn_count + 1} (session={sessionId})")
                tool_calls_accumulator = {}

        # End of while loop
        logger.info(f"Conversation complete after {turn_count} turns")
        if streamThinkId:
            await atlantis.stream_end(streamThinkId)
            logger.info("Think stream ended successfully")
        if streamTalkId:
            await atlantis.stream_end(streamTalkId)
            logger.info("Talk stream ended successfully")

    except Exception as e:
        logger.error(f"ERROR calling OpenRouter: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full exception:", exc_info=True)

        error_details = str(e)
        err_body = getattr(e, 'body', None)
        err_response = getattr(e, 'response', None)
        if err_body:
            logger.error(f"Error body: {err_body}")
            error_details = f"{error_details} | Body: {err_body}"
        if err_response:
            try:
                logger.error(f"Response status: {err_response.status_code}")
                logger.error(f"Response text: {err_response.text}")
                error_details = f"{error_details} | Status: {err_response.status_code} | Response: {err_response.text}"
            except:
                pass

        await atlantis.owner_log(f"Error calling OpenRouter: {error_details}")
        for sid in [streamThinkId, streamTalkId]:
            if sid:
                try:
                    await atlantis.stream_end(sid)
                except:
                    pass
        raise

    return accumulated_text or None
