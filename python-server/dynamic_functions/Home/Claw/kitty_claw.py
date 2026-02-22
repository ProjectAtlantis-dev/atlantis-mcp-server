import atlantis
import asyncio

from anthropic import Anthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolUseBlock,
)
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
# LLM Tool Types (output for Anthropic API)
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


class TranscriptToolT(TypedDict, total=False):
    """Tool format for LLM transcripts (Anthropic-compatible) - NO extra fields allowed!"""
    name: str
    description: str
    input_schema: ToolSchemaT


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
) -> Tuple[List[TranscriptToolT], List[SimpleToolT], Dict[str, ToolLookupInfo]]:
    """
    Convert /search tool records to LLM-compatible format.

    Args:
        tools: List of tool records from /search command
        show_hidden: If False, skip tools starting with '_'

    Returns:
        Tuple of (full tools list, simple tools list, lookup dict)
        - full tools list: Clean Anthropic-compatible tool definitions
        - simple tools list: Simplified tool format
        - lookup dict: Maps sanitized tool name -> ToolLookupInfo for resolving calls
    """
    out_tools: List[TranscriptToolT] = []
    out_tools_simple: List[SimpleToolT] = []
    tool_lookup: Dict[str, ToolLookupInfo] = {}

    logger.info(f"convert_tools_for_llm: Processing {len(tools) if tools else 0} tools")

    for tool in tools:
        search_term = tool.get('searchTerm', '')
        tool_name = tool.get('tool_name', '')  # The actual clean tool name
        tool_str = tool.get('tool', '')
        description = tool.get('description', '')
        chat_status = tool.get('chatStatus', '')
        filename = tool.get('filename', '')  # e.g., "Home/chat.py"

        # Parse the search term to extract components
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

        # Use tool_name if available, otherwise use parsed function name
        func_name = tool_name if tool_name else parsed['function']

        # Use filename from tool if available, otherwise use derived filename from parser
        actual_filename = filename if filename else parsed['filename']

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

        # Include app name in tool key to avoid collisions between apps
        # Format: AppName__function_name (e.g., "Home__chat")
        import re
        app_name = parsed.get('app', '') or ''
        if app_name:
            full_name = f"{app_name}__{func_name}"
        else:
            full_name = func_name
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', full_name)

        # Build the output tool (Anthropic format) - CLEAN, no extra fields!
        out_tool: TranscriptToolT = {
            'name': sanitized_name,
            'description': description,
            'input_schema': schema
        }
        out_tools.append(out_tool)

        # Build the simple version
        out_tools_simple.append({
            'name': sanitized_name,
            'description': description,
            'input_schema': schema
        })

        # Build lookup entry for resolving tool calls back to files
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
) -> List[TextBlockParam]:
    """
    Build the system prompt as an array of TextBlockParam blocks.
    cache_control goes on the last static block so everything before it gets cached.
    Dynamic skills go after and change freely without busting the cache.
    """
    blocks: List[TextBlockParam] = []

    # Base prompt is always first static block
    blocks.append({"type": "text", "text": base_prompt})

    # Each static skill gets its own block
    for text in static_skills:
        blocks.append({"type": "text", "text": text})

    # Mark the last static block with cache_control
    blocks[-1]["cache_control"] = {"type": "ephemeral"}

    # Dynamic skills go after the cache breakpoint
    for text in dynamic_skills:
        blocks.append({"type": "text", "text": text})

    # Fallback: if no dynamic skills, add date so there's always something after cache
    if not dynamic_skills:
        blocks.append({"type": "text", "text": f"Current date: {datetime.now().strftime('%Y-%m-%d')}"})

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

        # Add elapsed time since last visit if available
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

        blocks.append({"type": "text", "text": visitor_note})

    return blocks


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
            # Handle old format (plain int)
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
    Fetch raw transcript from server and transform it into Anthropic-compatible format.
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

    # Filter transcript to only include chat messages, removing type and sid fields
    # Note: Anthropic doesn't allow system role in messages - we pass it separately
    logger.info("=== FILTERING TRANSCRIPT ===")
    transcript: List[Dict[str, Any]] = []

    for i, msg in enumerate(raw_transcript):
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

            # Prefix user messages with timestamp
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




# no location since this is catch-all chat
# no app since this is catch-all chat
@chat
async def kitty_claw():
    """Main chat function"""
    logger.info("=== CHAT FUNCTION STARTING ===")
    sessionId = atlantis.get_session_id()
    logger.info(f"Session ID: {sessionId}")
    caller: str = atlantis.get_caller()  # type: ignore[assignment]
    visit_count, last_visit = record_visit(caller)
    logger.info(f"Visitor: {caller}, visit #{visit_count}, last visit: {last_visit or 'first time'}")

    # The rest of the function body is indented due to the removed try/finally block
    # Keeping the indentation to avoid reformatting the entire file
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

        # Don't respond if last chat message was from Kitty (the bot)
        last_chat_entry = find_last_chat_entry(rawTranscript)
        if last_chat_entry and last_chat_entry.get('type') == 'chat' and last_chat_entry.get('sid') == 'kitty':
            logger.warning("\x1b[38;5;204mLast chat entry was from kitty (bot), skipping response\x1b[0m")
            await atlantis.owner_log(f"Skipping response - last chat was from kitty (sid={last_chat_entry.get('sid')})")
            return

        await atlantis.client_command("/silent on")

        # Get available tools
        tools = await fetch_tools()





        # Fetch static and dynamic skills for system prompt
        static_skill_texts, dynamic_skill_texts = await fetch_skills()

        await atlantis.client_command("/silent off")


        # uses env var
        # Configure Anthropic client
        logger.info("Configuring Anthropic client...")

        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            error_msg = "ANTHROPIC_API_KEY environment variable is not set"
            logger.error(error_msg)
            await atlantis.owner_log(error_msg)
            raise ValueError(error_msg)

        client = Anthropic(api_key=api_key)

        # Model options:
        model = "claude-opus-4-6"  # Most capable, supports adaptive thinking
        # model = "claude-opus-4-5-20251101"  # Opus 4.5
        # model = "claude-sonnet-4-20250514"  # Good balance
        logger.info(f"Using model: {model}")


        # Multi-turn conversation loop to handle tool calls
        streamTalkId = None
        max_turns = 5  # Prevent infinite loops
        turn_count = 0

        try:
            # Start streaming response ONCE for entire conversation
            logger.info("Starting stream output...")
            streamTalkId = await atlantis.stream_start("kitty","Kitty")
            streamThinkId = None
            logger.info(f"Stream started with ID: {streamTalkId}")

            # Convert tools from cloud format to Anthropic-compatible format (once, before loop)
            # tool_lookup maps Anthropic's tool name -> {searchTerm, filename, functionName}
            converted_tools, _, tool_lookup = convert_tools_for_llm(tools)
            typed_tools = cast(List[ToolParam], converted_tools)

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

                logger.info(f"Attempting to call Anthropic with model: {model}")
                if turn_count == 1:
                    await atlantis.owner_log(f"Attempting to call Anthropic: {model}")

                # Cast transcript to proper type for Anthropic client
                typed_transcript = cast(List[MessageParam], transcript)

                # Log what we're actually sending to Anthropic
                logger.info(f"=== SENDING TO ANTHROPIC (turn {turn_count}) ===")
                logger.info(f"Messages: {len(typed_transcript)} entries")
                logger.info(f"Tools: {len(typed_tools)} entries")
                logger.info(f"Tool names sent to Anthropic: {[t['name'] for t in typed_tools]}")

                logger.info("Creating streaming message request...")

                event_count = 0
                streamed_count = 0  # Track how many text chunks we've streamed
                max_stream_chunks = 512  # Abort after this many
                tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}  # Store tool calls by index
                current_block_index = 0  # Track which content block we're in
                tool_call_made = False  # Track if we made a tool call this turn
                accumulated_tool_uses: List[ToolUseBlock] = []  # Store complete tool use blocks for transcript
                thinking_blocks_accumulator: Dict[int, Dict[str, Any]] = {}  # Accumulate thinking blocks by index
                accumulated_thinking_blocks: List[Dict[str, Any]] = []  # Store complete thinking blocks for transcript
                streamThinkId = None  # Separate stream for thinking

                # Use Anthropic's streaming context manager
                # Anthropic caching starts at 1024 tokens min
                with client.messages.stream(
                    model=model,
                    system=build_system_prompt(base_prompt, static_skill_texts, dynamic_skill_texts, caller, visit_count, last_visit),
                    tools=cast(List[ToolParam], typed_tools),
                    messages=typed_transcript,
                    max_tokens=16000,
                    thinking=cast(Any, {"type": "adaptive"}),
                    output_config=cast(Any, {"effort": "low"}),
                ) as stream:
                    logger.info("Anthropic API call successful!")
                    if turn_count == 1:
                        await atlantis.owner_log("Anthropic API call successful, starting stream")

                    logger.info("Beginning to process stream events...")

                    for event in stream:
                        event_count += 1

                        # Handle different event types
                        if event.type == "content_block_start":
                            current_block_index = event.index
                            if hasattr(event, 'content_block'):
                                block = event.content_block
                                if block.type == "thinking":
                                    logger.info(f"💭 Thinking block starting (index {current_block_index})")
                                    thinking_blocks_accumulator[current_block_index] = {
                                        'thinking': '',
                                        'signature': ''
                                    }
                                elif block.type == "tool_use":
                                    # Tool call starting - capture id and name
                                    logger.info(f"🔧 Tool use block starting: {block.name} (id: {block.id})")
                                    tool_calls_accumulator[current_block_index] = {
                                        'id': block.id,
                                        'name': block.name,
                                        'arguments': ''
                                    }

                        elif event.type == "content_block_delta":
                            if hasattr(event, 'delta'):
                                delta = event.delta
                                if delta.type == "text_delta":
                                    # Close thinking stream before streaming text
                                    if streamThinkId:
                                        await atlantis.stream_end(streamThinkId)
                                        streamThinkId = None
                                    # Stream text content
                                    text = delta.text
                                    content_to_send = text.lstrip() if streamed_count == 0 else text

                                    if content_to_send:
                                        await atlantis.stream(content_to_send, streamTalkId)
                                        streamed_count += 1

                                        if streamed_count >= max_stream_chunks:
                                            logger.warning(f"Aborting stream after {streamed_count} chunks")
                                            break

                                elif delta.type == "thinking_delta":
                                    # Stream thinking to separate thinking stream
                                    if current_block_index in thinking_blocks_accumulator:
                                        thinking_blocks_accumulator[current_block_index]['thinking'] += delta.thinking
                                        if not streamThinkId:
                                            streamThinkId = await atlantis.stream_start("kitty", "Kitty (thinking)")
                                        await atlantis.stream(delta.thinking, streamThinkId)

                                elif delta.type == "signature_delta":
                                    # Accumulate thinking signature
                                    if current_block_index in thinking_blocks_accumulator:
                                        thinking_blocks_accumulator[current_block_index]['signature'] += delta.signature

                                elif delta.type == "input_json_delta":
                                    # Accumulate tool arguments
                                    if current_block_index in tool_calls_accumulator:
                                        tool_calls_accumulator[current_block_index]['arguments'] += delta.partial_json

                        elif event.type == "content_block_stop":
                            # Block complete - if it was a thinking block, store it
                            if current_block_index in thinking_blocks_accumulator:
                                acc = thinking_blocks_accumulator[current_block_index]
                                logger.info(f"💭 Thinking block complete ({len(acc['thinking'])} chars)")
                                accumulated_thinking_blocks.append({
                                    'type': 'thinking',
                                    'thinking': acc['thinking'],
                                    'signature': acc['signature']
                                })

                            # Block complete - if it was a tool use, store it
                            if current_block_index in tool_calls_accumulator:
                                acc = tool_calls_accumulator[current_block_index]
                                logger.info(f"🔧 Tool use block complete: {acc['name']}")
                                # Parse and store the complete tool use block
                                try:
                                    parsed_input = json.loads(acc['arguments']) if acc['arguments'] else {}
                                    accumulated_tool_uses.append(ToolUseBlock(
                                        type="tool_use",
                                        id=acc['id'],
                                        name=acc['name'],
                                        input=parsed_input
                                    ))
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse tool arguments: {e}")

                        elif event.type == "message_delta":
                            # Check stop reason - just log it, we handle tools after stream ends
                            if hasattr(event, 'delta') and hasattr(event.delta, 'stop_reason'):
                                stop_reason = event.delta.stop_reason
                                logger.info(f"Message stop_reason: {stop_reason}")

                # End of stream context - now handle any tool calls that were accumulated
                logger.info(f"Stream processing complete for turn {turn_count}. Total events: {event_count}")

                # Execute accumulated tool calls if any
                if accumulated_tool_uses:
                    logger.info(f"Executing {len(accumulated_tool_uses)} accumulated tool calls")

                    # First, add assistant message with thinking + tool_use blocks to transcript
                    # Thinking blocks MUST be preserved for tool use continuity
                    assistant_content: List[Dict[str, Any]] = []
                    for thinking_block in accumulated_thinking_blocks:
                        assistant_content.append(thinking_block)
                    for tool_use in accumulated_tool_uses:
                        assistant_content.append({
                            'type': 'tool_use',
                            'id': tool_use.id,
                            'name': tool_use.name,
                            'input': tool_use.input
                        })
                    transcript.append({
                        'role': 'assistant',
                        'content': assistant_content
                    })

                    # Now execute each tool and collect results
                    tool_results: List[Dict[str, Any]] = []
                    for tool_use in accumulated_tool_uses:
                        call_id = tool_use.id
                        tool_key = tool_use.name  # This is the sanitized key Anthropic knows
                        arguments = tool_use.input

                        # Look up the real function info from our lookup table
                        if tool_key not in tool_lookup:
                            logger.error(f"\x1b[91m🚨 UNKNOWN TOOL KEY: '{tool_key}' not in tool_lookup!\x1b[0m")
                            logger.error(f"\x1b[91m  Available keys: {list(tool_lookup.keys())}\x1b[0m")
                            raise ValueError(f"Unknown tool: {tool_key}")

                        lookup_info = tool_lookup[tool_key]
                        search_term = lookup_info['searchTerm']
                        function_name = lookup_info['functionName']
                        filename = lookup_info['filename']

                        logger.info(f"=== EXECUTING TOOL: {tool_key} ===")
                        logger.info(f"  ID: {call_id}")
                        logger.info(f"  Resolved: searchTerm='{search_term}', function='{function_name}', file='{filename}'")
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

                            # Coerce arguments to match schema types
                            if tool_schema and arguments:
                                arguments = coerce_args_to_schema(arguments, tool_schema)
                                logger.info(f"  Post-coercion arguments: {format_json_log(arguments)}")

                            # Execute the tool call through atlantis client command using search term
                            await atlantis.client_command("/silent on")
                            tool_result = await atlantis.client_command(f"%{search_term}", data=arguments)
                            await atlantis.client_command("/silent off")

                            logger.info(f"  Tool result: {tool_result}")

                            # Send tool result to client for transcript display
                            await atlantis.tool_result(function_name, tool_result)

                            # Collect result for transcript
                            tool_results.append({
                                'type': 'tool_result',
                                'tool_use_id': call_id,
                                'content': str(tool_result) if tool_result else "No result"
                            })

                            tool_call_made = True

                        except Exception as e:
                            logger.error(f"Error executing tool call: {e}")
                            # Add error result
                            tool_results.append({
                                'type': 'tool_result',
                                'tool_use_id': call_id,
                                'is_error': True,
                                'content': f"Error: {str(e)}"
                            })

                    # Add all tool results as a single user message (Anthropic format)
                    transcript.append({
                        'role': 'user',
                        'content': tool_results
                    })

                    logger.info("Tool calls executed and added to transcript")

                # Check if we made a tool call and should continue
                if not tool_call_made:
                    logger.info("No tool call made this turn - conversation complete")
                    break  # Exit outer while loop
                else:
                    logger.info("Tool call made - continuing to next turn with updated transcript")
                    # Reset for next turn
                    accumulated_tool_uses = []
                    tool_calls_accumulator = {}
                    accumulated_thinking_blocks = []
                    thinking_blocks_accumulator = {}
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

            # Extract detailed error info from Anthropic APIError
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
            # Make sure to close streams on error
            for sid in [streamThinkId, streamTalkId]:
                if sid:
                    try:
                        await atlantis.stream_end(sid)
                    except:
                        pass
            raise

        logger.info("=== CHAT FUNCTION COMPLETED SUCCESSFULLY ===")
        return



