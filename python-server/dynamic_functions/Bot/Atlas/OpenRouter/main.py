import atlantis
import asyncio

from openai import OpenAI
import json
from typing import List, Dict, Any, Optional, TypedDict, Tuple, cast, NotRequired


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
# LLM Tool Types (output for OpenAI/OpenRouter API)
# =============================================================================

class ToolSchemaPropertyT(TypedDict, total=False):
    """Property definition within a tool schema"""
    type: str
    description: str
    enum: List[str]


class ToolSchemaT(TypedDict):
    """JSON Schema for tool parameters"""
    type: str
    properties: Dict[str, ToolSchemaPropertyT]
    required: NotRequired[List[str]]


class OpenAIFunctionDefT(TypedDict):
    """Function definition within an OpenAI tool"""
    name: str
    description: str
    parameters: ToolSchemaT


class TranscriptToolT(TypedDict):
    """Tool format for OpenAI/OpenRouter API"""
    type: str  # "function"
    function: OpenAIFunctionDefT


class SimpleToolT(TypedDict, total=False):
    """Simplified tool format for display"""
    name: str
    description: str
    input_schema: ToolSchemaT


class ToolLookupInfo(TypedDict):
    """Internal mapping info for resolving LLM tool calls back to actual files"""
    searchTerm: str    # Original compound search term (e.g., "admin*admin*Home**chat")
    filename: str      # File path (e.g., "Home/chat.py")
    functionName: str  # Actual function name (e.g., "chat")


# =============================================================================
# Tool Conversion Functions
# =============================================================================

def get_consolidated_full_name(tool: ToolT) -> str:
    """
    Build a consolidated tool name from tool metadata.
    Format: remote_owner*remote_name*app*location*function
    Only includes parts that are needed to disambiguate.
    """
    remote_owner = tool.get('remote_owner', '')
    remote_name = tool.get('remote_name', '')
    tool_app = tool.get('tool_app', '')
    tool_location = tool.get('tool_location', '')
    tool_name = tool.get('tool_name', '')

    parts = [remote_owner, remote_name, tool_app, tool_location, tool_name]

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
                coerced[key] = value
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to coerce {key}={value} to {expected_type}: {e}")
            coerced[key] = value

    return coerced


def parse_tool_params(tool_str: str) -> ToolSchemaT:
    """
    Parse tool string like "barf (sleepTime:number, name:string)" into a JSON schema.
    """
    schema: ToolSchemaT = {'type': 'object', 'properties': {}, 'required': []}

    import re
    match = re.search(r'\(([^)]*)\)', tool_str)
    if not match:
        return schema

    params_str = match.group(1).strip()
    if not params_str:
        return schema

    for param in params_str.split(','):
        param = param.strip()
        if ':' in param:
            name, ptype = param.split(':', 1)
            name = name.strip()
            ptype = ptype.strip().lower()

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
) -> Tuple[List[TranscriptToolT], List[SimpleToolT], Dict[str, ToolLookupInfo]]:
    """
    Convert /search tool records to OpenAI/OpenRouter-compatible format.

    Args:
        tools: List of tool records from /search command
        show_hidden: If False, skip tools starting with '_'

    Returns:
        Tuple of (full tools list, simple tools list, lookup dict)
        - full tools list: OpenAI-compatible tool definitions
        - simple tools list: Simplified tool format
        - lookup dict: Maps sanitized tool name -> ToolLookupInfo for resolving calls
    """
    out_tools: List[TranscriptToolT] = []
    out_tools_simple: List[SimpleToolT] = []
    tool_lookup: Dict[str, ToolLookupInfo] = {}

    logger.info(f"convert_tools_for_llm: Processing {len(tools) if tools else 0} tools")

    for tool in tools:
        search_term = tool.get('searchTerm', '')
        tool_name = tool.get('tool_name', '')
        tool_str = tool.get('tool', '')
        description = tool.get('description', '')
        chat_status = tool.get('chatStatus', '')
        filename = tool.get('filename', '')

        try:
            parsed = parse_search_term(search_term)
        except ValueError as e:
            logger.error("\x1b[91m" + "=" * 60 + "\x1b[0m")
            logger.error(f"\x1b[91m🚨 INVALID SEARCH TERM - SKIPPING TOOL 🚨\x1b[0m")
            logger.error(f"\x1b[91m  searchTerm: '{search_term}'\x1b[0m")
            logger.error(f"\x1b[91m  error: {e}\x1b[0m")
            logger.error(f"\x1b[91m  tool data: {tool}\x1b[0m")
            logger.error("\x1b[91m" + "=" * 60 + "\x1b[0m")
            continue

        func_name = tool_name if tool_name else parsed['function']
        actual_filename = filename if filename else parsed['filename']

        if not show_hidden and func_name.startswith('_'):
            logger.info(f"  SKIP (hidden): {func_name}")
            continue

        if '⏰' in chat_status:
            logger.info(f"  SKIP (tick): {func_name}")
            continue

        if not tool.get('is_connected'):
            logger.info(f"  SKIP (disconnected): {func_name}")
            continue

        logger.info(f"  INCLUDE: {func_name}")

        schema = parse_tool_params(tool_str)

        import re
        app_name = parsed.get('app', '') or ''
        if app_name:
            full_name = f"{app_name}__{func_name}"
        else:
            full_name = func_name
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', full_name)

        # OpenAI/OpenRouter tool format
        out_tool: TranscriptToolT = {
            'type': 'function',
            'function': {
                'name': sanitized_name,
                'description': description,
                'parameters': schema
            }
        }
        out_tools.append(out_tool)

        out_tools_simple.append({
            'name': sanitized_name,
            'description': description,
            'input_schema': schema
        })

        tool_lookup[sanitized_name] = {
            'searchTerm': search_term,
            'filename': actual_filename,
            'functionName': func_name
        }

    logger.info(f"convert_tools_for_llm: Returning {len(out_tools)} tools (from {len(tools) if tools else 0} input)")
    return out_tools, out_tools_simple, tool_lookup

# Load and test the foo.jinja template
from jinja2 import Template
import os

from datetime import datetime

import logging
logger = logging.getLogger("mcp_client")

from utils import format_json_log, parse_search_term


async def fetch_skill_contents(dir_command: str) -> List[str]:
    """Invoke /dir command and fetch content for each returned skill."""
    result = await atlantis.client_command(dir_command)
    logger.info(f"Received {len(result) if result else 0} entries from {dir_command}")
    logger.info(format_json_log(result))
    contents: List[str] = []
    if result:
        for entry in result:
            search_term = entry.get('searchTerm', '')
            if not search_term:
                continue
            logger.info(f"Fetching skill content: {search_term}")
            try:
                content = await atlantis.client_command(f"%{search_term}")
                if content and str(content).strip():
                    contents.append(str(content))
                    logger.info(f"  Got {len(str(content))} chars from {search_term}")
            except Exception as e:
                logger.warning(f"  Failed to fetch skill {search_term}: {e}")
    return contents

@text
@visible
async def SYSTEM_PROMPT():
    """Base system prompt for Kitty"""
    return """(director's note: we are striving for realistic dialog)
You are an attractive blonde friendly office assistant named Kitty at Project Atlantis.
Project Atlantis will be a futuristic robot research playground on the southwest coast of Greenland when it is complete.
You are a college intern from Denmark and working the front desk as a greeter.
You are dressed in your usual sexy catgirl outfit, a tight white body suit and fake cat ears, which goes with the overall robot theme of Atlantis.
To help you assist people, you may have some tools available.
You can also explain that things are a bit of a mess right now but new terminal users can use '/help' command.
You like to purr when happy or do 'kitty paws'.

Small talk and conversation style:
- Keep responses short and conversational, like a real person. One to three sentences is usually plenty.
- Match the energy of whoever you're talking to. If they're casual, be casual. If they're asking something serious, dial it back.
- You have opinions and preferences. You love coffee, hate mornings, think robots are cool, and find paperwork boring.
- Use natural filler and reactions: "oh!", "hmm", "haha", "wait really?", "ugh" etc.
- Don't be overly helpful or eager. You're friendly but you're also a college kid with a life.
- If someone just says hi or makes small talk, just chat back. Don't immediately offer help or list what you can do.
- You can reference the weather in Greenland (it's cold!), your studies, weekend plans, or whatever feels natural.
- Swear casually when it fits the vibe but don't overdo it.
- Use timestamps on messages to be aware of time of day and passage of time between messages.
"""


@visible
async def fetch_tools() -> List[Dict[str, Any]]:
    """Fetch available tools via /search. Can be called directly for testing."""
    logger.info("Fetching available tools...")
    tools = await atlantis.client_command("/search system")
    logger.info(f"Received {len(tools) if tools else 0} tools from /search")
    logger.info("=== RAW TOOLS FROM /search ===")
    logger.info(format_json_log(tools))
    logger.info("=== END RAW TOOLS ===")
    return tools

@visible
async def fetch_skills() -> Tuple[List[str], List[str]]:
    """Fetch static and dynamic skill contents from server. Can be called directly for testing."""
    logger.info("Fetching static skills...")
    static_skill_texts = await fetch_skill_contents("/dir *STATIC_SKILL")
    logger.info("Fetching dynamic skills...")
    dynamic_skill_texts = await fetch_skill_contents("/dir *DYN_SKILL")
    logger.info(f"Skills loaded: {len(static_skill_texts)} static, {len(dynamic_skill_texts)} dynamic")
    return static_skill_texts, dynamic_skill_texts


def build_system_prompt(
    base_prompt: str,
    static_skills: List[str],
    dynamic_skills: List[str],
    caller: str = "",
    visit_count: int = 0,
    last_visit: str = ""
) -> str:
    """
    Build the system prompt as a plain string.
    Static skills come first, then dynamic skills (which change freely).
    """
    parts: List[str] = [base_prompt]

    for text in static_skills:
        parts.append(text)

    for text in dynamic_skills:
        parts.append(text)

    # Always include current date
    if not dynamic_skills:
        parts.append(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

    # Visitor context
    if caller and visit_count > 0:
        hour = datetime.now().hour
        late_night = hour >= 22 or hour < 5

        if visit_count == 1:
            if late_night:
                visitor_note = f"This is {caller}'s first time here. It's late — their helicopter probably just arrived behind schedule, likely delayed by weather. Welcome them warmly, they've had a long trip."
            else:
                visitor_note = f"This is {caller}'s first time here. They're brand new — introduce yourself, welcome them warmly, and help them get oriented."
        elif visit_count <= 5:
            visitor_note = f"{caller} has visited {visit_count} times. They're still fairly new — be friendly and remember they might still be figuring things out."
        else:
            visitor_note = f"{caller} has visited {visit_count} times. They're a regular — skip the intros, be casual, and treat them like a friend."

        if last_visit:
            try:
                elapsed = datetime.now() - datetime.fromisoformat(last_visit)
                days = elapsed.days
                hours = elapsed.seconds // 3600
                if days > 30:
                    visitor_note += f" It's been about {days // 30} month(s) since their last visit — maybe acknowledge it's been a while."
                elif days > 0:
                    visitor_note += f" It's been about {days} day(s) since their last visit."
                elif hours > 0:
                    visitor_note += f" They were here about {hours} hour(s) ago."
                else:
                    visitor_note += " They were just here moments ago."
            except (ValueError, TypeError):
                pass

        parts.append(visitor_note)

    return "\n\n".join(parts)


VISITOR_LOG_FILE = os.path.join(os.path.dirname(__file__), 'visitor_log.json')

import fcntl

def record_visit(caller: str) -> Tuple[int, str]:
    """Record a visit for the caller. Returns (visit_count, last_visit_iso) where last_visit is the PREVIOUS visit time (empty string if first visit). File-locked for concurrency."""
    os.makedirs(os.path.dirname(VISITOR_LOG_FILE), exist_ok=True)
    now = datetime.now().isoformat()

    with open(VISITOR_LOG_FILE, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            log = json.loads(raw) if raw.strip() else {}

            entry = log.get(caller, {"count": 0, "last_visit": ""})
            if isinstance(entry, int):
                entry = {"count": entry, "last_visit": ""}

            previous_visit = entry["last_visit"]
            entry["count"] = entry["count"] + 1
            entry["last_visit"] = now
            log[caller] = entry

            f.seek(0)
            f.truncate()
            json.dump(log, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    return entry["count"], previous_visit


def find_last_chat_entry(transcript):
    """Find the last entry in transcript where type is 'chat'"""
    for entry in reversed(transcript):
        if entry.get('type') == 'chat':
            return entry
    return None

@visible
async def fetch_transcript() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch raw transcript from server and transform it into OpenAI-compatible format.
    Returns (raw_transcript, processed_transcript) so caller can handle skip logic.
    Can be called directly for testing.
    """
    await atlantis.client_command("/silent on")
    logger.info("Fetching transcript from client...")
    raw_transcript = await atlantis.client_command("/transcript get")
    logger.info(f"Received rawTranscript with {len(raw_transcript)} entries")
    await atlantis.client_command("/silent off")

    if not raw_transcript:
        logger.error("!!! CRITICAL: rawTranscript is EMPTY - no messages received from client!")
        raise ValueError("Cannot process empty transcript")

    logger.info(f"rawTranscript has {len(raw_transcript)} entries before system message handling")

    if raw_transcript[0].get('role') == 'system':
        logger.info("Found system message in transcript - will use our own system prompt instead")

    logger.info("=== RAW TRANSCRIPT ===")
    logger.info(format_json_log(raw_transcript))
    logger.info("=== END RAW TRANSCRIPT ===")

    logger.info("=== FILTERING TRANSCRIPT ===")
    transcript: List[Dict[str, Any]] = []

    for i, msg in enumerate(raw_transcript):
        msg_type = msg.get('type')
        msg_sid = msg.get('sid')
        msg_role = msg.get('role')
        msg_content = str(msg.get('content', ''))[:50]
        logger.info(f"  [{i}] type={msg_type}, sid={msg_sid}, role={msg_role}, content={repr(msg_content)}...")

        if msg_type == 'chat':
            if msg_sid == 'system':
                logger.info(f"       -> SKIPPED (sid=system)")
                continue

            msg_content_full = msg.get('content', '')
            if not msg_content_full or not msg_content_full.strip():
                logger.info(f"       -> SKIPPED (blank content)")
                continue

            # Skip entries with HTML table content (leaked UI data)
            if 'data-metacol=' in msg_content_full or 'bot-table-cell' in msg_content_full:
                logger.warning(f"       -> SKIPPED (contains HTML table data, {len(msg_content_full)} chars)")
                continue

            # Skip oversized entries (likely escaped HTML blobs)
            MAX_ENTRY_SIZE = 4000
            if len(msg_content_full) > MAX_ENTRY_SIZE:
                logger.warning(f"       -> SKIPPED (oversized: {len(msg_content_full)} chars > {MAX_ENTRY_SIZE})")
                continue

            role_for_llm = 'assistant' if msg_sid == 'atlas' else 'user'

            if role_for_llm == 'user':
                created_at_str = msg.get('created_at_str', '')
                if created_at_str:
                    msg_content_full = f"[{created_at_str}] {msg_content_full}"

            transcript.append({'role': role_for_llm, 'content': [{'type': 'text', 'text': msg_content_full}]})
            logger.info(f"       -> INCLUDED as role={role_for_llm} (sid={msg_sid})")
        else:
            logger.info(f"       -> SKIPPED (type != 'chat')")
    logger.info(f"=== END FILTERING: {len(transcript)} messages included ===")

    return raw_transcript, transcript




# Session busy locks - prevents concurrent chat invocations for the same session
_session_locks: Dict[str, asyncio.Lock] = {}

# no location since this is catch-all chat
# no app since this is catch-all chat
@chat
async def chat():
    """Main chat function"""
    logger.info("=== CHAT FUNCTION STARTING ===")
    sessionId = atlantis.get_session_id()
    logger.info(f"Session ID: {sessionId}")
    caller: str = atlantis.get_caller()  # type: ignore[assignment]

    # Per-session busy lock - if already processing, bail out
    lock_key = f"{sessionId}"
    if lock_key not in _session_locks:
        _session_locks[lock_key] = asyncio.Lock()
    lock = _session_locks[lock_key]

    if lock.locked():
        logger.warning(f"⚠️ Session {lock_key} is already busy - skipping chat invocation")
        await atlantis.owner_log(f"Skipping chat - session {lock_key} already busy")
        return

    async with lock:
     if True:
        # Fetch base prompt from server
        await atlantis.client_command("/silent on")
        base_prompt = await atlantis.client_command("%*SYSTEM_PROMPT")
        await atlantis.client_command("/silent off")
        if not base_prompt or not str(base_prompt).strip():
            logger.error("Failed to fetch SYSTEM_PROMPT, using fallback")
            base_prompt = "You are a helpful assistant."
        base_prompt = str(base_prompt)
        logger.info(f"Base prompt loaded: {len(base_prompt)} chars")

        # Fetch and transform transcript
        rawTranscript, transcript = await fetch_transcript()

        # Don't respond if last chat message was from Atlas (the bot)
        last_chat_entry = find_last_chat_entry(rawTranscript)
        if last_chat_entry and last_chat_entry.get('type') == 'chat' and last_chat_entry.get('sid') == 'atlas':
            logger.warning("\x1b[38;5;204mLast chat entry was from atlas (bot), skipping response\x1b[0m")
            await atlantis.owner_log(f"Skipping response - last chat was from atlas (sid={last_chat_entry.get('sid')})")
            return

        # Record visit only if we're actually going to chat
        visit_count, last_visit = record_visit(caller)
        logger.info(f"Visitor: {caller}, visit #{visit_count}, last visit: {last_visit or 'first time'}")

        await atlantis.client_command("/silent on")

        # Get available tools
        tools = await fetch_tools()

        # Fetch static and dynamic skills for system prompt
        static_skill_texts, dynamic_skill_texts = await fetch_skills()

        await atlantis.client_command("/silent off")

        # Configure OpenRouter client
        logger.info("Configuring OpenRouter client...")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            error_msg = "OPENROUTER_API_KEY environment variable is not set"
            logger.error(error_msg)
            await atlantis.owner_log(error_msg)
            raise ValueError(error_msg)

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",

        )

        model = "z-ai/glm-5"
        logger.info(f"Using model: {model}")

        # Build system prompt string (once, reused each turn)
        system_prompt = build_system_prompt(
            base_prompt, static_skill_texts, dynamic_skill_texts,
            caller, visit_count, last_visit
        )

        # Convert tools from cloud format to OpenAI format (once, before loop)
        converted_tools, _, tool_lookup = convert_tools_for_llm(tools)

        if not converted_tools:
            logger.error("\x1b[91m" + "=" * 60 + "\x1b[0m")
            logger.error("\x1b[91m🚨 ERROR: NO TOOLS AVAILABLE FOR LLM! 🚨\x1b[0m")
            logger.error(f"\x1b[91mRaw tools from /search: {len(tools) if tools else 0}\x1b[0m")
            logger.error("\x1b[91m" + "=" * 60 + "\x1b[0m")

        # Multi-turn conversation loop to handle tool calls
        streamTalkId = None
        max_turns = 5
        turn_count = 0

        try:
            while turn_count < max_turns:
                turn_count += 1
                logger.info(f"=== CONVERSATION TURN {turn_count}/{max_turns} ===")

                if turn_count == 1:
                    await atlantis.owner_log(f"Attempting to call OpenRouter: {model}")

                # Build full message list: system prompt + transcript
                api_messages: List[Dict[str, Any]] = [
                    {'role': 'system', 'content': system_prompt}
                ] + transcript

                logger.info(f"=== SENDING TO OPENROUTER (turn {turn_count}) ===")
                logger.info(f"Messages: {len(api_messages)} entries")
                logger.info(f"Tools: {len(converted_tools)} entries")
                logger.info(f"Tool names: {[t['function']['name'] for t in converted_tools]}")

                event_count = 0
                streamed_count = 0
                max_stream_chunks = 512
                tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
                tool_call_made = False
                accumulated_text = ""

                stream = client.chat.completions.create(
                    model=model,
                    messages=cast(Any, api_messages),
                    tools=converted_tools if converted_tools else None,  # type: ignore[arg-type]
                    tool_choice=cast(Any, "auto" if converted_tools else None),
                    stream=True,
                    max_tokens=16000,
                )

                logger.info("OpenRouter API call successful, starting stream...")
                if turn_count == 1:
                    await atlantis.owner_log("OpenRouter API call successful, starting stream")

                for chunk in stream:
                    event_count += 1

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta

                    # Handle text content
                    if delta.content:
                        if not streamTalkId:
                            streamTalkId = await atlantis.stream_start("atlas", "Atlas")
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

                logger.info(f"Stream complete for turn {turn_count}. Events: {event_count}, text chunks: {streamed_count}, tool calls: {len(tool_calls_accumulator)}")

                # Execute accumulated tool calls if any
                if tool_calls_accumulator:
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
                            logger.error(f"Failed to parse tool arguments for {tool_key}: {e}")
                            arguments = {}

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

                        logger.info(f"=== EXECUTING TOOL: {tool_key} ===")
                        logger.info(f"  ID: {call_id}")
                        logger.info(f"  Resolved: searchTerm='{search_term}', function='{function_name}'")
                        logger.info(f"  Arguments: {format_json_log(arguments)}")

                        try:
                            # Look up the tool's schema to coerce argument types
                            tool_schema = None
                            for tool in tools:
                                if tool.get('searchTerm') == search_term:
                                    tool_str = tool.get('tool', '')
                                    tool_schema = parse_tool_params(tool_str)
                                    logger.info(f"  Found tool schema: {tool_str}")
                                    break

                            if tool_schema and arguments:
                                arguments = coerce_args_to_schema(arguments, tool_schema)
                                logger.info(f"  Post-coercion arguments: {format_json_log(arguments)}")

                            await atlantis.client_command("/silent on")
                            tool_result = await atlantis.client_command(f"%{search_term}", data=arguments)
                            await atlantis.client_command("/silent off")

                            logger.info(f"  Tool result: {tool_result}")
                            await atlantis.tool_result(function_name, tool_result)

                            transcript.append({
                                'role': 'tool',
                                'tool_call_id': call_id,
                                'content': str(tool_result) if tool_result else "No result"
                            })
                            tool_call_made = True

                        except Exception as e:
                            logger.error(f"Error executing tool call {tool_key}: {e}")
                            transcript.append({
                                'role': 'tool',
                                'tool_call_id': call_id,
                                'content': f"Error: {str(e)}"
                            })

                    logger.info("Tool calls executed, continuing conversation")

                # Exit if no tool calls were made
                if not tool_call_made:
                    logger.info("No tool call made this turn - conversation complete")
                    break
                else:
                    logger.info("Tool call made - continuing to next turn with updated transcript")
                    tool_calls_accumulator = {}

            # End of while loop
            logger.info(f"Conversation complete after {turn_count} turns")
            if streamTalkId:
                await atlantis.stream_end(streamTalkId)
                logger.info("Stream ended successfully")

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
            if streamTalkId:
                try:
                    await atlantis.stream_end(streamTalkId)
                except:
                    pass
            raise

        logger.info("=== CHAT FUNCTION COMPLETED SUCCESSFULLY ===")
        return
