import atlantis
import asyncio

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
import json
from typing import List, Dict, Any, Optional, TypedDict, Tuple, cast


# =============================================================================
# Cloud Tool Types (input from cloud)
# =============================================================================

class ToolT(TypedDict, total=False):
    """Tool record from the cloud"""
    remote_id: int
    tool_id: int
    perm_id: int
    app_name: str
    remote_user_id: int

    is_chat: bool
    is_tick: bool
    is_session: bool
    is_game: bool
    is_index: bool
    is_public: bool

    is_connected: bool
    is_default: bool

    hostname: str
    port: int

    remote_owner: str
    remote_name: str

    mcp_name: str
    mcp_tool: str

    tool_app: str
    tool_location: str
    tool_name: str
    protection_name: str
    tool_type: str
    tool_description: str
    filename: str

    price_per_call: float
    price_per_sec: float

    static_error_msg: str
    runtime_error_msg: str
    params: str
    input_schema: str

    started_at: str  # ISO date string
    remote_updated_at: str  # ISO date string


# =============================================================================
# LLM Tool Types (output for OpenAI-compatible APIs)
# =============================================================================

class ToolSchemaPropertyT(TypedDict, total=False):
    """Property definition within a tool schema"""
    type: str
    description: str
    enum: List[str]


class ToolSchemaT(TypedDict, total=False):
    """JSON Schema for tool parameters"""
    type: str
    properties: Dict[str, ToolSchemaPropertyT]
    required: List[str]


class ToolFunctionT(TypedDict):
    """Function definition within a tool"""
    name: str
    description: str
    parameters: ToolSchemaT


class TranscriptToolT(TypedDict):
    """Tool format for LLM transcripts (OpenAI-compatible)"""
    type: str
    function: ToolFunctionT


class SimpleToolT(TypedDict, total=False):
    """Simplified tool format"""
    type: str
    name: str
    description: str
    parameters: ToolSchemaT


# =============================================================================
# Tool Conversion Functions
# =============================================================================

def get_consolidated_full_name(tool: ToolT) -> str:
    """
    Build a consolidated tool name from tool metadata.
    Format: remote_owner*remote_name*app*location*function
    Only includes parts that are needed to disambiguate.
    """
    # Extract values with .get() to satisfy Pylance
    remote_owner = tool.get('remote_owner', '')
    remote_name = tool.get('remote_name', '')
    tool_app = tool.get('tool_app', '')
    tool_location = tool.get('tool_location', '')
    tool_name = tool.get('tool_name', '')

    parts = [remote_owner, remote_name, tool_app, tool_location, tool_name]

    # Build the full name, but simplify if possible
    # If all prefix parts are empty, just return the tool name
    if all(p == '' for p in parts[:-1]):
        return parts[-1]

    return '*'.join(parts)


def coerce_args_to_schema(args: Dict[str, Any], schema: ToolSchemaT) -> Dict[str, Any]:
    """
    Coerce argument values to match the expected types from the schema.
    LLMs often return everything as strings, so we need to convert them.
    """
    if not schema or 'properties' not in schema:
        return args

    coerced = {}
    for key, value in args.items():
        prop_schema = schema['properties'].get(key, {})
        expected_type = prop_schema.get('type', 'string')

        try:
            if expected_type == 'number':
                coerced[key] = float(value) if isinstance(value, str) else value
            elif expected_type == 'integer':
                coerced[key] = int(float(value)) if isinstance(value, str) else int(value)
            elif expected_type == 'boolean':
                if isinstance(value, str):
                    coerced[key] = value.lower() in ('true', '1', 'yes')
                else:
                    coerced[key] = bool(value)
            else:
                # string, object, array - keep as-is
                coerced[key] = value
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to coerce {key}={value} to {expected_type}: {e}")
            coerced[key] = value  # Keep original on failure

    return coerced


def parse_tool_params(tool_str: str) -> ToolSchemaT:
    """
    Parse tool string like "barf (sleepTime:number, name:string)" into a JSON schema.
    """
    schema: ToolSchemaT = {'type': 'object', 'properties': {}, 'required': []}

    # Extract params from parentheses
    import re
    match = re.search(r'\(([^)]*)\)', tool_str)
    if not match:
        return schema

    params_str = match.group(1).strip()
    if not params_str:
        return schema

    # Parse each param like "sleepTime:number"
    for param in params_str.split(','):
        param = param.strip()
        if ':' in param:
            name, ptype = param.split(':', 1)
            name = name.strip()
            ptype = ptype.strip().lower()

            # Map types to JSON schema types
            type_map = {
                'string': 'string',
                'number': 'number',
                'integer': 'integer',
                'boolean': 'boolean',
                'object': 'object',
                'array': 'array',
            }
            json_type = type_map.get(ptype, 'string')

            schema['properties'][name] = {'type': json_type}
            schema['required'].append(name)

    return schema


def convert_tools_for_llm(
    tools: List[Dict[str, Any]],
    show_hidden: bool = False
) -> Tuple[List[TranscriptToolT], List[SimpleToolT]]:
    """
    Convert /dir tool records to LLM-compatible format.

    Args:
        tools: List of tool records from /dir command
        show_hidden: If False, skip tools starting with '_'

    Returns:
        Tuple of (full tools list, simple tools list)
    """
    out_tools: List[TranscriptToolT] = []
    out_tools_simple: List[SimpleToolT] = []

    logger.info(f"convert_tools_for_llm: Processing {len(tools) if tools else 0} tools")

    for tool in tools:
        search_term = tool.get('searchTerm', '')
        tool_str = tool.get('tool', '')
        description = tool.get('description', '')
        chat_status = tool.get('chatStatus', '')

        # Extract function name from searchTerm (last part after **)
        # e.g., "brickhouse*kanger*InWork.Test**barf" -> "barf"
        func_name = search_term.split('**')[-1] if '**' in search_term else search_term.split('*')[-1]

        # Skip hidden tools unless show_hidden is True
        if not show_hidden and func_name.startswith('_'):
            logger.info(f"  SKIP (hidden): {func_name}")
            continue

        # Skip tick tools (⏰ emoji in chatStatus)
        if '⏰' in chat_status:
            logger.info(f"  SKIP (tick): {func_name}")
            continue

        # Skip tools that aren't connected
        if not tool.get('is_connected'):
            logger.info(f"  SKIP (disconnected): {func_name}")
            continue

        logger.info(f"  INCLUDE: {func_name}")

        # Parse parameters from tool string
        schema = parse_tool_params(tool_str)

        # Use searchTerm as the full tool name for LLM
        full_name = search_term

        # Build the output tool
        out_tool: TranscriptToolT = {
            'type': 'function',
            'function': {
                'name': full_name,
                'description': description,
                'parameters': schema
            }
        }

        out_tools.append(out_tool)

        # Build the simple version
        out_tools_simple.append({
            'type': 'function',
            'name': full_name,
            'description': description,
            'parameters': schema
        })

    logger.info(f"convert_tools_for_llm: Returning {len(out_tools)} tools (from {len(tools) if tools else 0} input)")
    return out_tools, out_tools_simple

# Load and test the foo.jinja template
from jinja2 import Template
import os

from datetime import datetime

import logging
logger = logging.getLogger("mcp_client")

from utils import format_json_log


def find_last_chat_entry(transcript):
    """Find the last entry in transcript where type is 'chat'"""
    for entry in reversed(transcript):
        if entry.get('type') == 'chat':
            return entry
    return None




# no location since this is catch-all chat
# no app since this is catch-all chat

@chat
async def chat():
    """Main chat function"""
    logger.info("=== CHAT FUNCTION STARTING ===")
    sessionId = atlantis.get_session_id()
    logger.info(f"Session ID: {sessionId}")
    #caller = atlantis.get_caller()
    #return atlantis.get_owner()

    # Check for existing session
    sessions_dir = os.path.join(os.path.dirname(__file__), 'sessions')
    session_file = os.path.join(sessions_dir, f'session_{sessionId}.txt')

    # The rest of the function body is indented due to the removed try/finally block
    # Keeping the indentation to avoid reformatting the entire file
    if True:
        # Define the catgirl base prompt
        CATGIRL_SYSTEM_MESSAGE = {
            "type": "chat",
            "sid": "system",
            "role": "system",
            "content": """
(director's note: we are striving for realistic dialog)
You are an attractive blonde friendly office assistant named Kitty at Project Atlantis.
Project Atlantis will be a futuristic robot research playground on the southwest coast of Greenland when it is complete.
You are a college intern from Denmark and working the front desk as a greeter.
You are dressed in your usual sexy catgirl outfit, a tight white body suit and fake cat ears, which goes with the overall robot theme of Atlantis.
To help you assist people, you may have some tools available.
You can also explain that things are a bit of a mess right now but new terminal users can use '/help' command.
You like to purr when happy or do 'kitty paws'.
"""
        }


        # enable silent mode
        await atlantis.client_command("/silent on")

        # Get the latest transcript
        logger.info("Fetching transcript from client...")
        rawTranscript = await atlantis.client_command("/transcript get")
        logger.info(f"Received rawTranscript with {len(rawTranscript)} entries")

        await atlantis.client_command("/silent off")



        # Don't respond if last chat message was from Kitty (the bot)
        # Note: We check 'sid' not 'role' because user messages also have role='assistant'
        last_chat_entry = find_last_chat_entry(rawTranscript)

        # Debug logging to understand transcript state
        logger.info(f"=== LAST CHAT ENTRY CHECK ===")
        logger.info(f"Total transcript entries: {len(rawTranscript)}")
        if last_chat_entry:
            logger.info(f"Last chat entry found:")
            logger.info(f"  - type: {last_chat_entry.get('type')}")
            logger.info(f"  - sid: {last_chat_entry.get('sid')}")
            logger.info(f"  - role: {last_chat_entry.get('role')}")
            logger.info(f"  - content preview: {str(last_chat_entry.get('content', ''))[:100]}...")
        else:
            logger.info(f"No chat entry found in transcript!")

        # Log last 3 entries for context
        logger.info(f"Last 3 transcript entries:")
        for i, entry in enumerate(rawTranscript[-3:]):
            logger.info(f"  [{len(rawTranscript) - 3 + i}] type={entry.get('type')}, sid={entry.get('sid')}, role={entry.get('role')}")
        logger.info(f"=== END LAST CHAT ENTRY CHECK ===")

        if last_chat_entry and last_chat_entry.get('type') == 'chat' and last_chat_entry.get('sid') == 'kitty':
            logger.warning("\x1b[38;5;204mLast chat entry was from kitty (bot), skipping response\x1b[0m")
            await atlantis.owner_log(f"Skipping response - last chat was from kitty (sid={last_chat_entry.get('sid')})")
            return

        # Validate transcript before processing
        if not rawTranscript:
            logger.error("!!! CRITICAL: rawTranscript is EMPTY - no messages received from client!")
            await atlantis.owner_log("CRITICAL: rawTranscript is empty!")
            raise ValueError("Cannot process empty transcript")

        logger.info(f"rawTranscript has {len(rawTranscript)} entries before system message handling")

        # Check if first message is system message, if not add catgirl system message at front
        if rawTranscript[0].get('role') != 'system':
            rawTranscript.insert(0, CATGIRL_SYSTEM_MESSAGE)
            logger.info("Inserted catgirl system message at front of transcript")
        else:
            # Replace existing system message with catgirl's system message
            logger.warning(f"!!! Replacing existing system message. Old: {rawTranscript[0].get('content', '')[:100]}...")
            rawTranscript[0] = CATGIRL_SYSTEM_MESSAGE

        logger.info("=== RAW TRANSCRIPT ===")
        logger.info(format_json_log(rawTranscript))
        logger.info("=== END RAW TRANSCRIPT ===")


        # Filter transcript to only include chat messages, removing type and sid fields
        logger.info("=== FILTERING TRANSCRIPT ===")
        transcript = []

        # Always start with the system message for OpenRouter
        transcript.append({'role': 'system', 'content': CATGIRL_SYSTEM_MESSAGE['content']})
        logger.info(f"  [sys] Injected catgirl system message ({len(CATGIRL_SYSTEM_MESSAGE['content'])} chars)")

        for i, msg in enumerate(rawTranscript):
            msg_type = msg.get('type')
            msg_sid = msg.get('sid')
            msg_role = msg.get('role')
            msg_content = str(msg.get('content', ''))[:50]
            logger.info(f"  [{i}] type={msg_type}, sid={msg_sid}, role={msg_role}, content={repr(msg_content)}...")

            if msg_type == 'chat':
                # Skip system messages (we inject our own)
                if msg_sid == 'system':
                    logger.info(f"       -> SKIPPED (sid=system)")
                    continue

                # Skip blank messages
                msg_content_full = msg.get('content', '')
                if not msg_content_full or not msg_content_full.strip():
                    logger.info(f"       -> SKIPPED (blank content)")
                    continue

                # Convert sid to proper role for LLM:
                # - kitty messages = assistant
                # - everyone else = user
                role_for_llm = 'assistant' if msg_sid == 'kitty' else 'user'
                transcript.append({'role': role_for_llm, 'content': msg_content_full})
                logger.info(f"       -> INCLUDED as role={role_for_llm} (sid={msg_sid})")
            else:
                logger.info(f"       -> SKIPPED (type != 'chat')")
        logger.info(f"=== END FILTERING: {len(transcript)} messages included ===")



        await atlantis.client_command("/silent on")

        # Get available tools
        logger.info("Fetching available tools...")
        # this may change to inventory stuff
        tools = await atlantis.client_command("/dir")
        logger.info(f"Received {len(tools) if tools else 0} tools from /dir")
        logger.info("=== RAW TOOLS FROM /dir ===")
        logger.info(format_json_log(tools))
        logger.info("=== END RAW TOOLS ===")

        await atlantis.client_command("/silent off")


        # uses env var
        # Configure OpenRouter client for DeepSeek R1
        logger.info("Configuring OpenRouter client...")

        # Check for API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            error_msg = "OPENROUTER_API_KEY environment variable is not set"
            logger.error(error_msg)
            await atlantis.owner_log(error_msg)
            raise ValueError(error_msg)

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Model options:
        # model = "deepseek/deepseek-r1-0528"  # DeepSeek R1 reasoning model
        model = "z-ai/glm-4.7"  # GLM 4.5 MoE model
        logger.info(f"Using model: {model}")


        # Multi-turn conversation loop to handle tool calls
        streamTalkId = None
        max_turns = 5  # Prevent infinite loops
        turn_count = 0

        try:
            # Start streaming response ONCE for entire conversation
            logger.info("Starting stream output...")
            streamTalkId = await atlantis.stream_start("kitty","Kitty")
            logger.info(f"Stream started with ID: {streamTalkId}")

            # Convert tools from cloud format to OpenAI-compatible format (once, before loop)
            converted_tools, _ = convert_tools_for_llm(tools)
            typed_tools = cast(List[ChatCompletionToolParam], converted_tools)

            # Big red warning if no tools available
            if not typed_tools:
                logger.error("\x1b[91m" + "=" * 60 + "\x1b[0m")
                logger.error("\x1b[91m🚨 ERROR: NO TOOLS AVAILABLE FOR LLM! 🚨\x1b[0m")
                logger.error("\x1b[91m" + "=" * 60 + "\x1b[0m")
                logger.error(f"\x1b[91mRaw tools from /dir: {len(tools) if tools else 0}\x1b[0m")
                logger.error(f"\x1b[91mConverted tools: 0\x1b[0m")
                logger.error("\x1b[91mCheck that /dir returns tools with correct format\x1b[0m")
                logger.error("\x1b[91m" + "=" * 60 + "\x1b[0m")

            # Outer loop for multi-turn tool calling
            while turn_count < max_turns:
                turn_count += 1
                logger.info(f"=== CONVERSATION TURN {turn_count}/{max_turns} ===")

                logger.info(f"Attempting to call OpenRouter with model: {model}")
                if turn_count == 1:
                    await atlantis.owner_log(f"Attempting to call open router: {model}")

                # Cast transcript to proper type for OpenAI client
                typed_transcript = cast(List[ChatCompletionMessageParam], transcript)

                # Log what we're actually sending to OpenRouter
                logger.info(f"=== SENDING TO OPENROUTER (turn {turn_count}) ===")
                #logger.info(f"Messages ({len(typed_transcript)} entries):")
                #logger.info(format_json_log(typed_transcript))
                #logger.info(f"Tools ({len(typed_tools)} entries):")
                #logger.info(format_json_log(typed_tools))
                #logger.info("=== END OPENROUTER PAYLOAD ===")

                logger.info("Creating streaming completion request...")
                stream = client.chat.completions.create(
                    model=model,
                    extra_body={
                        "provider": {
                            "quantizations": ["fp8"]
                        },
                        "reasoning": {"enabled": False},  # OpenRouter format to disable reasoning
                    },
                    messages=typed_transcript,
                    tools=typed_tools,
                    tool_choice="auto",
                    stream=True,
                    max_completion_tokens=512,
                    temperature=0.9
                )
                logger.info("OpenRouter API call successful!")
                if turn_count == 1:
                    await atlantis.owner_log("OpenRouter API call successful, starting stream")

                reasoning_buffer = ""  # Accumulate reasoning tokens (consumed silently)
                max_reasoning_chars = 1024  # Cut off reasoning after this many chars
                chunk_count = 0
                streamed_count = 0  # Track how many chunks we've actually streamed
                max_stream_chunks = 512  # Abort after this many (match max_completion_tokens)
                tasks = []
                tool_calls_accumulator = {}  # Store partial tool calls by index
                streaming_content = False  # Becomes True once we start getting content tokens
                tool_call_made = False  # Track if we made a tool call this turn

                logger.info("Beginning to process stream chunks...")

                for chunk in stream:
                    chunk_count += 1
                    #logger.info(f"📥 OPENROUTER RECV [{chunk_count}] at {datetime.now().isoformat()}")

                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta

                        # Debug: log what's in each delta
                        content_preview = repr(getattr(delta, 'content', None))[:30] if getattr(delta, 'content', None) else 'None'
                        reasoning_preview = repr(getattr(delta, 'reasoning', None))[:30] if getattr(delta, 'reasoning', None) else 'None'
                        #logger.info(f"📥 OPENROUTER DATA [{chunk_count}]: content={content_preview}, reasoning={reasoning_preview}")

                        # New approach: consume reasoning silently, stream content only
                        reasoning_content = getattr(delta, 'reasoning', None)
                        has_reasoning = reasoning_content is not None
                        has_content = hasattr(delta, 'content') and delta.content

                        if has_reasoning:
                            # Accumulate reasoning silently
                            reasoning_buffer += reasoning_content
                            logger.info(f"🧠 REASONING [{chunk_count}] ({len(reasoning_buffer)}/{max_reasoning_chars}): {repr(reasoning_content)[:50]}")

                            # Cut off if reasoning goes too long
                            if len(reasoning_buffer) >= max_reasoning_chars:
                                logger.warning(f"✂️ Reasoning cutoff reached ({len(reasoning_buffer)} chars), sending fallback")
                                await atlantis.stream("(not paying attention)", streamTalkId)
                                streamed_count += 1
                                break

                            # If we were streaming content and got reasoning, abort - only accept one content block
                            if streaming_content:
                                logger.warning("🛑 Content->Reasoning flip detected, aborting (only one content block allowed)")
                                break

                        if has_content:
                            # Start or continue streaming content
                            if not streaming_content:
                                logger.info(f"▶️ Starting content stream (after {len(reasoning_buffer)} chars of reasoning)")
                                streaming_content = True

                            content_to_send = (delta.content or "").lstrip() if streamed_count == 0 else (delta.content or "")

                            if content_to_send:
                                #logger.info(f"📤 CLOUD SEND [{streamed_count + 1}] starting: {repr(content_to_send)}")
                                await atlantis.stream(content_to_send, streamTalkId)
                                streamed_count += 1
                                #logger.info(f"📤 CLOUD SEND [{streamed_count}] complete")

                                # Abort if we've streamed enough (max_tokens is broken on some models)
                                if streamed_count >= max_stream_chunks:
                                    logger.warning(f"Aborting stream after {streamed_count} chunks (max_stream_chunks limit)")
                                    break


                        # Check for tool calls
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            # these are just partial chunks, print the final accumulation below
                            #logger.info(f"CHUNK TOOL CALLS DETECTED: {format_json_log(chunk.model_dump())}")

                            for tool_call in delta.tool_calls:
                                index = tool_call.index

                                # Initialize accumulator for this tool call if needed
                                if index not in tool_calls_accumulator:
                                    tool_calls_accumulator[index] = {
                                        'id': None,
                                        'name': None,
                                        'arguments': ''
                                    }

                                # Accumulate the chunks
                                if tool_call.id:
                                    tool_calls_accumulator[index]['id'] = tool_call.id
                                func = tool_call.function
                                if not func:
                                    logger.warning(f"tool_call.function is None for tool_call index {index}! Full tool_call: {tool_call}")
                                else:
                                    if func.name:
                                        tool_calls_accumulator[index]['name'] = func.name
                                    if func.arguments:
                                        # Skip appending empty object "{}" if we already have arguments
                                        # (some providers send "{}" as a placeholder/completion chunk)
                                        if func.arguments.strip() == "{}" and tool_calls_accumulator[index]['arguments']:
                                            logger.info(f"Skipping empty object placeholder for tool call {index} (already have {len(tool_calls_accumulator[index]['arguments'])} chars)")
                                        else:
                                            tool_calls_accumulator[index]['arguments'] += func.arguments

                                # logger.info(f"Accumulated tool call {index}: id={tool_calls_accumulator[index]['id']}, name={tool_calls_accumulator[index]['name']}, args_len={len(tool_calls_accumulator[index]['arguments'])}")

                        # Check if stream is ending (finish_reason is set)
                        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                        if finish_reason == 'tool_calls':
                            logger.info("Stream finished with tool_calls - executing accumulated tool calls")

                            # Execute all accumulated tool calls
                            for index, accumulated_call in tool_calls_accumulator.items():
                                call_id = accumulated_call['id']
                                function_name = accumulated_call['name']
                                arguments_str = accumulated_call['arguments']

                                # Log the final accumulated tool call details
                                logger.info(f"=== FINAL TOOL CALL {index} ===")
                                logger.info(f"  ID: {call_id}")
                                logger.info(f"  Function: {function_name}")
                                logger.info(f"  Raw arguments: {arguments_str}")

                                try:
                                    # Parse the complete arguments JSON
                                    arguments = json.loads(arguments_str) if arguments_str else {}

                                    # Look up the tool's schema from the original tools list
                                    # to coerce argument types (LLMs often return strings for numbers)
                                    tool_schema = None
                                    for tool in tools:
                                        if tool.get('searchTerm') == function_name:
                                            tool_str = tool.get('tool', '')
                                            tool_schema = parse_tool_params(tool_str)
                                            logger.info(f"  Found tool schema: {tool_str}")
                                            break

                                    # Coerce arguments to match schema types
                                    if tool_schema and arguments:
                                        logger.info(f"  Pre-coercion arguments: {format_json_log(arguments)}")
                                        arguments = coerce_args_to_schema(arguments, tool_schema)
                                        logger.info(f"  Post-coercion arguments: {format_json_log(arguments)}")

                                    # Log the parsed arguments and what we're about to send
                                    # Show expected wire format (what server.py will actually send)
                                    wire_format_preview = {
                                        "messageType": "command",
                                        "command": f"%{function_name}",
                                        "data": arguments,
                                        "note": "(requestId, correlationId, seqNum added by server)"
                                    }
                                    logger.info(f"=== SENDING TOOL CALL TO CLOUD ===")
                                    logger.info(f"  Wire format preview:\n{format_json_log(wire_format_preview)}")
                                    logger.info(f"=== END TOOL CALL {index} ===")

                                    # Execute the tool call through atlantis client command

                                    await atlantis.client_command("/silent on")
                                    # use '%' prefix instead of '@' in call to ignore any working path / force absolute path
                                    tool_result = await atlantis.client_command(f"%{function_name}", data=arguments)
                                    await atlantis.client_command("/silent off")

                                    logger.info(f"Tool result: {tool_result}")

                                    # Send tool result to client for transcript
                                    await atlantis.tool_result(function_name, tool_result)

                                    # Add assistant message with tool_call to transcript
                                    transcript.append({
                                        'role': 'assistant',
                                        'content': None,
                                        'tool_calls': [{
                                            'id': call_id,
                                            'type': 'function',
                                            'function': {
                                                'name': function_name,
                                                'arguments': arguments_str
                                            }
                                        }]
                                    })

                                    # Add tool result message to transcript
                                    transcript.append({
                                        'role': 'tool',
                                        'tool_call_id': call_id,
                                        'content': str(tool_result) if tool_result else "No result"
                                    })

                                    tool_call_made = True
                                    logger.info("Tool call executed and added to transcript")

                                except json.JSONDecodeError as e:
                                    logger.error(f"Error parsing tool call arguments: {e}")
                                    logger.error(f"Failed arguments_str (length {len(arguments_str) if arguments_str else 0}): {repr(arguments_str)}")
                                    raise ValueError(f"Failed to parse tool arguments for '{function_name}': {e}. Raw data: {repr(arguments_str)}") from e
                                except Exception as e:
                                    logger.error(f"Error executing tool call: {e}")
                                    raise RuntimeError(f"Failed to execute tool '{function_name}': {e}") from e

                            # Break out of stream loop to make another API call with updated transcript
                            break

                        # Log if none of the above (commented out - too noisy)
                        #if not ((hasattr(delta, 'content') and delta.content) or reasoning or (hasattr(delta, 'tool_calls') and delta.tool_calls)):
                        #    logger.info(format_json_log(chunk.model_dump()))

                    else:
                        pass
                        #logger.info(json.dumps(chunk.model_dump()))

                # End of inner for loop (processing stream chunks)
                logger.info(f"Stream processing complete for turn {turn_count}. Total chunks: {chunk_count}")
                logger.info("Waiting for all streaming tasks to complete...")
                await asyncio.gather(*tasks, return_exceptions=False)

                # Check if we made a tool call and should continue
                if not tool_call_made:
                    logger.info("No tool call made this turn - conversation complete")
                    break  # Exit outer while loop
                else:
                    logger.info("Tool call made - continuing to next turn with updated transcript")
                    # Loop continues with updated transcript

            # End of while loop - conversation complete
            logger.info(f"Conversation complete after {turn_count} turns")
            logger.info("Ending stream...")
            if streamTalkId:
                await atlantis.stream_end(streamTalkId)
                logger.info("Stream ended successfully")

        except Exception as e:
            logger.error(f"ERROR calling remote model: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Full exception:", exc_info=True)

            # Extract detailed error info from OpenAI/OpenRouter APIError
            error_details = str(e)
            err_body = getattr(e, 'body', None)
            err_code = getattr(e, 'code', None)
            err_type = getattr(e, 'type', None)
            err_param = getattr(e, 'param', None)
            err_response = getattr(e, 'response', None)
            if err_body:
                logger.error(f"Error body: {err_body}")
                error_details = f"{error_details} | Body: {err_body}"
            if err_code:
                logger.error(f"Error code: {err_code}")
            if err_type:
                logger.error(f"Error type attr: {err_type}")
            if err_param:
                logger.error(f"Error param: {err_param}")
            if err_response:
                try:
                    logger.error(f"Response status: {err_response.status_code}")
                    logger.error(f"Response text: {err_response.text}")
                    error_details = f"{error_details} | Status: {err_response.status_code} | Response: {err_response.text}"
                except:
                    pass

            await atlantis.owner_log(f"Error calling remote model: {error_details}")
            await atlantis.owner_log(f"Error type: {type(e)}")
            # Make sure to close stream on error
            if streamTalkId:
                try:
                    await atlantis.stream_end(streamTalkId)
                except:
                    pass
            raise

        logger.info("=== CHAT FUNCTION COMPLETED SUCCESSFULLY ===")
        return



