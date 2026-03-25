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
# LLM Tool Types (output for Anthropic API)
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


class TranscriptToolT(TypedDict, total=False):
    """Tool format for Anthropic API"""
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
    Convert /search tool records to Anthropic-compatible format.

    Args:
        tools: List of tool records from /search command
        show_hidden: If False, skip tools starting with '_'

    Returns:
        Tuple of (full tools list, simple tools list, lookup dict)
        - full tools list: Anthropic-compatible tool definitions
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

        # Anthropic tool format
        out_tool: TranscriptToolT = {
            'name': sanitized_name,
            'description': description,
            'input_schema': schema
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
    try:
        result = await atlantis.client_command(dir_command)
    except Exception as e:
        logger.warning(f"Failed to fetch skill list from {dir_command}: {e}")
        return []
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
async def fetch_skills() -> List[str]:
    """Fetch skill contents from server. Can be called directly for testing."""
    logger.info("Fetching skills...")
    skill_texts = await fetch_skill_contents("/dir *SKILL")
    logger.info(f"Skills loaded: {len(skill_texts)}")
    return skill_texts


def build_system_prompt(
    base_prompt: str,
    skills: List[str],
    caller: str = "",
    visit_count: int = 0,
    last_visit: str = ""
) -> str:
    """
    Build the system prompt as a plain string.
    """
    parts: List[str] = [base_prompt]

    for text in skills:
        parts.append(text)

    # Always include current date and time
    parts.append(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

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


VISITOR_LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'visitor_data.json')

import fcntl

def get_visit_info(caller: str) -> Tuple[int, str]:
    """Get visit info for the caller. Returns (visit_count, last_visit_iso). Does not modify the log."""
    os.makedirs(os.path.dirname(VISITOR_LOG_FILE), exist_ok=True)

    try:
        with open(VISITOR_LOG_FILE, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                raw = f.read()
                log = json.loads(raw) if raw.strip() else {}
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except FileNotFoundError:
        return 0, ""

    entry = log.get(caller, {"count": 0, "last_visit": ""})
    if isinstance(entry, int):
        entry = {"count": entry, "last_visit": ""}

    return entry["count"], entry["last_visit"]


def record_new_conversation(caller: str):
    """Increment visit count and update timestamp. Called on first visit or when >1hr gap detected."""
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
            entry["count"] = entry["count"] + 1
            entry["last_visit"] = now
            log[caller] = entry
            f.seek(0)
            f.truncate()
            json.dump(log, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    logger.info(f"New conversation for {caller}: visit #{entry['count']}, timestamp {now}")


SEARCH_PSEUDO_TOOL: TranscriptToolT = {
    'name': 'search',
    'description': 'Search for available tools and commands. Results will be added to your tool list so you can call them. Use this when a user asks you to do something and you don\'t have an appropriate tool yet.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Search query to find tools (e.g. "weather", "admin", "game")'
            }
        },
        'required': ['query']
    }
}


DIR_PSEUDO_TOOL: TranscriptToolT = {
    'name': 'dir',
    'description': 'Look up tools by name. Use this when you know the specific name of a tool you want to load. Results will be added to your tool list.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'name': {
                'type': 'string',
                'description': 'Tool name to look up (e.g. "foo", "admin*Home*chat")'
            }
        },
        'required': ['name']
    }
}


async def handle_dir_tool(
    name: str,
    converted_tools: List[TranscriptToolT],
    tool_lookup: Dict[str, ToolLookupInfo]
) -> Tuple[str, List[TranscriptToolT], Dict[str, ToolLookupInfo]]:
    """
    Look up tools by name via /dir and merge into the existing tool list.
    Returns (summary_text, updated_tools, updated_lookup).
    """
    import time as _t
    logger.info(f">>> DIR PSEUDO-TOOL: name='{name}' — sending /dir command...")
    t0 = _t.monotonic()

    try:
        results = await atlantis.client_command(f"/dir {name}")
    except Exception as e:
        elapsed = _t.monotonic() - t0
        logger.error(f"<<< DIR FAILED after {elapsed:.2f}s: {e}")
        return f"Dir lookup failed: {e}", converted_tools, tool_lookup

    elapsed = _t.monotonic() - t0
    logger.info(f"<<< DIR RETURNED in {elapsed:.2f}s — {len(results) if results else 0} results")

    if not results:
        return f"No tools found for '{name}'.", converted_tools, tool_lookup

    new_tools, new_simple, new_lookup = convert_tools_for_llm(results)

    added_names = []
    for tool in new_tools:
        tool_name = tool['name']
        if tool_name not in tool_lookup:
            converted_tools.append(tool)
            added_names.append(f"{tool_name}: {tool['description']}")

    for tool_name, info in new_lookup.items():
        if tool_name not in tool_lookup:
            tool_lookup[tool_name] = info

    if added_names:
        summary = f"Found {len(added_names)} new tool(s) added to your toolkit:\n" + "\n".join(f"- {n}" for n in added_names)
    else:
        summary = f"Dir lookup for '{name}' returned {len(new_tools)} tool(s), but all were already in your toolkit."

    logger.info(f"Dir result: {summary}")
    return summary, converted_tools, tool_lookup


async def handle_search_tool(
    query: str,
    converted_tools: List[TranscriptToolT],
    tool_lookup: Dict[str, ToolLookupInfo]
) -> Tuple[str, List[TranscriptToolT], Dict[str, ToolLookupInfo]]:
    """
    Execute a search query and merge new tools into the existing tool list.
    Returns (summary_text, updated_tools, updated_lookup).
    """
    import time as _t
    logger.info(f">>> SEARCH PSEUDO-TOOL: query='{query}' — sending /search command...")
    t0 = _t.monotonic()

    try:
        results = await atlantis.client_command(f"/search {query}")
    except Exception as e:
        elapsed = _t.monotonic() - t0
        logger.error(f"<<< SEARCH FAILED after {elapsed:.2f}s: {e}")
        return f"Search failed: {e}", converted_tools, tool_lookup

    elapsed = _t.monotonic() - t0
    logger.info(f"<<< SEARCH RETURNED in {elapsed:.2f}s — {len(results) if results else 0} results")

    if not results:
        return f"No tools found for '{query}'.", converted_tools, tool_lookup

    new_tools, new_simple, new_lookup = convert_tools_for_llm(results)

    added_names = []
    for tool in new_tools:
        name = tool['name']
        if name not in tool_lookup:
            converted_tools.append(tool)
            added_names.append(f"{name}: {tool['description']}")

    for name, info in new_lookup.items():
        if name not in tool_lookup:
            tool_lookup[name] = info

    if added_names:
        summary = f"Found {len(added_names)} new tool(s) added to your toolkit:\n" + "\n".join(f"- {n}" for n in added_names)
    else:
        summary = f"Search for '{query}' returned {len(new_tools)} tool(s), but all were already in your toolkit."

    logger.info(f"Search result: {summary}")
    return summary, converted_tools, tool_lookup


def find_last_chat_entry(transcript):
    """Find the last entry in transcript where type is 'chat', skipping thinking entries"""
    for entry in reversed(transcript):
        if entry.get('type') == 'chat':
            who = entry.get('who', '')
            if 'thinking' in str(who).lower():
                continue
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

    # Dump raw transcript to file for debugging (overwrites each time)
    transcript_dump_file = os.path.join(os.path.dirname(__file__), 'raw_transcript.json')
    try:
        with open(transcript_dump_file, 'w') as f:
            json.dump(raw_transcript, f, indent=2, default=str)
        logger.info(f"Raw transcript written to {transcript_dump_file}")
    except Exception as e:
        logger.warning(f"Failed to write raw transcript to file: {e}")

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

            msg_who = str(msg.get('who', ''))
            if 'thinking' in msg_who.lower():
                logger.info(f"       -> SKIPPED (thinking entry, who={msg_who})")
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

            role_for_llm = 'assistant' if msg_sid == 'kitty' else 'user'

            transcript.append({'role': role_for_llm, 'content': [{'type': 'text', 'text': msg_content_full}]})
            logger.info(f"       -> INCLUDED as role={role_for_llm} (sid={msg_sid})")
        else:
            logger.info(f"       -> SKIPPED (type != 'chat')")
    logger.info(f"=== END FILTERING: {len(transcript)} messages included ===")

    return raw_transcript, transcript




# Session-scoped keys — stored in session_shared (auto-scoped per user session)
_BUSY_KEY = "kitty_busy"


# no location since this is catch-all chat
# no app since this is catch-all chat
@chat
async def chat():
    """Anthropic Opus direct"""
    sessionId = atlantis.get_session_id() or "unknown"
    requestId = atlantis.get_request_id() or "unknown"
    caller: str = atlantis.get_caller()  # type: ignore[assignment]

    logger.info("=" * 60)
    logger.info(f"=== CHAT TRIGGERED === session={sessionId} request={requestId} caller={caller}")

    # Check if this session is already being handled
    owner_req = atlantis.session_shared.get(_BUSY_KEY)
    if owner_req:
        logger.warning(f"🔒 BUSY: session={sessionId} already owned by request={owner_req}, this request={requestId} — skipping")
        await atlantis.owner_log(f"Skipping chat — session {sessionId} busy (owned by request {owner_req})")
        return

    atlantis.session_shared.set(_BUSY_KEY, requestId)
    logger.info(f"🔒 ACQUIRED: session={sessionId} by request={requestId}")

    try:
        import time as _t

        # Fetch base prompt from server
        logger.info(f">>> Fetching SYSTEM_PROMPT via client_command...")
        t0 = _t.monotonic()
        await atlantis.client_command("/silent on")
        base_prompt = await atlantis.client_command("@../SYSTEM_PROMPT")
        await atlantis.client_command("/silent off")
        if not base_prompt or not str(base_prompt).strip():
            logger.error("Failed to fetch SYSTEM_PROMPT, using fallback")
            base_prompt = "You are a helpful assistant."
        base_prompt = str(base_prompt)
        logger.info(f"<<< SYSTEM_PROMPT loaded in {_t.monotonic() - t0:.2f}s ({len(base_prompt)} chars)")

        # Fetch and transform transcript
        logger.info(f">>> Fetching transcript...")
        t0 = _t.monotonic()
        rawTranscript, transcript = await fetch_transcript()
        logger.info(f"<<< Transcript fetched in {_t.monotonic() - t0:.2f}s ({len(rawTranscript)} raw, {len(transcript)} filtered)")

        # Don't respond if last chat message was from the bot
        last_chat_entry = find_last_chat_entry(rawTranscript)
        if last_chat_entry:
            logger.info(f"  Last chat entry: sid={last_chat_entry.get('sid')} type={last_chat_entry.get('type')} content={str(last_chat_entry.get('content',''))[:80]}")
        if last_chat_entry and last_chat_entry.get('type') == 'chat' and last_chat_entry.get('sid') == 'kitty':
            logger.warning(f"\x1b[38;5;204mLast chat was from kitty — skipping (session={sessionId} request={requestId})\x1b[0m")
            await atlantis.owner_log(f"Skipping response - last chat was from kitty")
            return

        # Check visit info before recording, so we can detect time gaps
        prev_count, prev_last_visit = get_visit_info(caller)
        logger.info(f"Visitor: {caller}, visit #{prev_count}, last visit: {prev_last_visit or 'first time'}")

        # Fetch skills for system prompt
        logger.info(f">>> Fetching skills...")
        t0 = _t.monotonic()
        await atlantis.client_command("/silent on")
        skill_texts = await fetch_skills()
        await atlantis.client_command("/silent off")
        logger.info(f"<<< Skills fetched in {_t.monotonic() - t0:.2f}s ({len(skill_texts)} skills)")

        # Configure Anthropic client (direct API)
        logger.info("Configuring Anthropic client...")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            error_msg = "ANTHROPIC_API_KEY environment variable is not set"
            logger.error(error_msg)
            await atlantis.owner_log(error_msg)
            raise ValueError(error_msg)

        client = Anthropic(
            api_key=api_key,
        )

        model = "claude-opus-4-6"
        logger.info(f"Using model: {model}")

        # If more than an hour since last visit (or first visit), stamp the convo start time
        if not prev_last_visit:
            record_new_conversation(caller)
        else:
            try:
                elapsed = datetime.now() - datetime.fromisoformat(prev_last_visit)
                if elapsed.total_seconds() > 3600:
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                    gap_msg = f"[Some time has passed since your last interaction. The current date and time is now {now_str}.]"
                    transcript.append({'role': 'user', 'content': [{'type': 'text', 'text': gap_msg}]})
                    record_new_conversation(caller)
                    logger.info(f"Injected time-gap message: {gap_msg}")
            except (ValueError, TypeError):
                pass

        # Re-read visit info after recording so the count is up-to-date for the system prompt
        visit_count, last_visit = get_visit_info(caller)

        # Build system prompt string (once, reused each turn)
        system_prompt = build_system_prompt(
            base_prompt, skill_texts,
            caller, visit_count, last_visit
        )

        # Start with only pseudo-tools — LLM discovers others dynamically
        converted_tools: List[TranscriptToolT] = [SEARCH_PSEUDO_TOOL, DIR_PSEUDO_TOOL]
        tool_lookup: Dict[str, ToolLookupInfo] = {}
        logger.info("Starting with search + dir pseudo-tools only (no pre-loaded tools)")

        # Multi-turn conversation loop to handle tool calls
        streamTalkId = None
        streamThinkId = None
        max_turns = 5
        turn_count = 0

        try:
            while turn_count < max_turns:
                turn_count += 1
                logger.info(f"=== TURN {turn_count}/{max_turns} === session={sessionId} request={requestId}")

                if turn_count == 1:
                    await atlantis.owner_log(f"Attempting to call Anthropic: {model}")

                typed_transcript = cast(List[MessageParam], transcript)
                typed_tools = cast(List[ToolParam], converted_tools)

                logger.info(f"=== SENDING TO ANTHROPIC (turn {turn_count}) ===")
                logger.info(f"Messages: {len(typed_transcript)} entries")
                logger.info(f"Tools: {len(typed_tools)} entries")
                logger.info(f"Tool names: {[t['name'] for t in converted_tools]}")

                # Dump full API payload for debugging (clobbers each turn)
                api_dump_file = os.path.join(os.path.dirname(__file__), 'api_payload.json')
                try:
                    with open(api_dump_file, 'w') as f:
                        json.dump({
                            'model': model,
                            'system': system_prompt,
                            'messages': transcript,
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
                current_block_index = 0
                tool_call_made = False
                accumulated_tool_uses: List[ToolUseBlock] = []
                thinking_blocks_accumulator: Dict[int, Dict[str, Any]] = {}
                accumulated_thinking_blocks: List[Dict[str, Any]] = []

                logger.info(f">>> Calling Anthropic ({model})...")
                t_api = _t.monotonic()

                with client.messages.stream(
                    model=model,
                    system=system_prompt,
                    tools=typed_tools,
                    messages=typed_transcript,
                    max_tokens=16000,
                    thinking=cast(Any, {"type": "adaptive"}),
                    output_config=cast(Any, {"effort": "low"}),
                ) as stream:
                    logger.info(f"<<< Anthropic stream opened in {_t.monotonic() - t_api:.2f}s, reading events...")
                    if turn_count == 1:
                        await atlantis.owner_log("Anthropic API call successful, starting stream")

                    for event in stream:
                        event_count += 1

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

                                    if not streamTalkId:
                                        streamTalkId = await atlantis.stream_start("kitty", "Kitty")
                                        logger.info(f"Talk stream started with ID: {streamTalkId}")

                                    text = delta.text
                                    content_to_send = text.lstrip() if streamed_count == 0 else text

                                    if content_to_send:
                                        await atlantis.stream(content_to_send, streamTalkId)
                                        streamed_count += 1

                                        if streamed_count >= max_stream_chunks:
                                            logger.warning(f"Aborting stream after {streamed_count} chunks")
                                            break

                                elif delta.type == "thinking_delta":
                                    if current_block_index in thinking_blocks_accumulator:
                                        thinking_blocks_accumulator[current_block_index]['thinking'] += delta.thinking
                                        if not streamThinkId:
                                            streamThinkId = await atlantis.stream_start("kitty", "Kitty (thinking)")
                                            logger.info(f"Think stream started with ID: {streamThinkId}")
                                        await atlantis.stream(delta.thinking, streamThinkId)

                                elif delta.type == "signature_delta":
                                    if current_block_index in thinking_blocks_accumulator:
                                        thinking_blocks_accumulator[current_block_index]['signature'] += delta.signature

                                elif delta.type == "input_json_delta":
                                    if current_block_index in tool_calls_accumulator:
                                        tool_calls_accumulator[current_block_index]['arguments'] += delta.partial_json

                        elif event.type == "content_block_stop":
                            if current_block_index in thinking_blocks_accumulator:
                                acc = thinking_blocks_accumulator[current_block_index]
                                logger.info(f"💭 Thinking block complete ({len(acc['thinking'])} chars)")
                                accumulated_thinking_blocks.append({
                                    'type': 'thinking',
                                    'thinking': acc['thinking'],
                                    'signature': acc['signature']
                                })

                            if current_block_index in tool_calls_accumulator:
                                acc = tool_calls_accumulator[current_block_index]
                                logger.info(f"🔧 Tool use block complete: {acc['name']}")
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
                            if hasattr(event, 'delta') and hasattr(event.delta, 'stop_reason'):
                                stop_reason = event.delta.stop_reason
                                logger.info(f"Message stop_reason: {stop_reason}")

                logger.info(f"Stream complete: turn={turn_count} events={event_count} text_chunks={streamed_count} tool_calls={len(accumulated_tool_uses)} session={sessionId}")

                # Execute accumulated tool calls if any
                if accumulated_tool_uses:
                    logger.info(f"Executing {len(accumulated_tool_uses)} tool calls")

                    # Add assistant message with thinking + tool_use blocks to transcript
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

                    # Execute each tool and collect results
                    tool_results: List[Dict[str, Any]] = []
                    for tool_use in accumulated_tool_uses:
                        call_id = tool_use.id
                        tool_key = tool_use.name
                        arguments = tool_use.input

                        # Handle dir pseudo-tool
                        if tool_key == 'dir':
                            dir_name = arguments.get('name', '')
                            logger.info(f"=== DIR TOOL CALLED: name='{dir_name}' ===")
                            summary, converted_tools, tool_lookup = await handle_dir_tool(
                                dir_name, converted_tools, tool_lookup
                            )
                            tool_results.append({
                                'type': 'tool_result',
                                'tool_use_id': call_id,
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
                            tool_results.append({
                                'type': 'tool_result',
                                'tool_use_id': call_id,
                                'content': summary
                            })
                            tool_call_made = True
                            continue

                        if tool_key not in tool_lookup:
                            logger.error(f"\x1b[91m🚨 UNKNOWN TOOL KEY: '{tool_key}' not in tool_lookup!\x1b[0m")
                            logger.error(f"\x1b[91m  Available keys: {list(tool_lookup.keys())}\x1b[0m")
                            tool_results.append({
                                'type': 'tool_result',
                                'tool_use_id': call_id,
                                'is_error': True,
                                'content': f"Error: Unknown tool: {tool_key}"
                            })
                            continue

                        lookup_info = tool_lookup[tool_key]
                        search_term = lookup_info['searchTerm']
                        function_name = lookup_info['functionName']

                        import time as _t
                        logger.info(f">>> EXECUTING TOOL: {tool_key} (call_id={call_id})")
                        logger.info(f"    searchTerm='{search_term}' function='{function_name}'")
                        logger.info(f"    args={format_json_log(arguments)}")

                        try:
                            # Look up the tool's schema for argument coercion
                            tool_schema = None
                            for ct in converted_tools:
                                if ct['name'] == tool_key:
                                    tool_schema = ct['input_schema']
                                    break

                            if tool_schema and arguments:
                                arguments = coerce_args_to_schema(arguments, tool_schema)
                                logger.info(f"    post-coercion args={format_json_log(arguments)}")

                            t0 = _t.monotonic()
                            logger.info(f"    sending /silent on...")
                            await atlantis.client_command("/silent on")
                            logger.info(f"    sending %{search_term}...")
                            tool_result = await atlantis.client_command(f"%{search_term}", data=arguments)
                            logger.info(f"    sending /silent off...")
                            await atlantis.client_command("/silent off")
                            elapsed = _t.monotonic() - t0

                            logger.info(f"<<< TOOL {tool_key} RETURNED in {elapsed:.2f}s — result: {str(tool_result)[:200]}")
                            await atlantis.tool_result(function_name, tool_result)

                            tool_results.append({
                                'type': 'tool_result',
                                'tool_use_id': call_id,
                                'content': str(tool_result) if tool_result else "No result"
                            })
                            tool_call_made = True

                        except Exception as e:
                            logger.error(f"<<< TOOL {tool_key} FAILED: {e}")
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

                    logger.info("Tool calls executed, continuing conversation")
                    logger.info(f"tool_lookup keys ({len(tool_lookup)}): {list(tool_lookup.keys())}")

                # Exit if no tool calls were made
                if not tool_call_made:
                    logger.info(f"No tool calls — conversation complete (session={sessionId})")
                    break
                else:
                    logger.info(f"Tool calls executed — continuing to turn {turn_count + 1} (session={sessionId})")
                    accumulated_tool_uses = []
                    tool_calls_accumulator = {}
                    accumulated_thinking_blocks = []
                    thinking_blocks_accumulator = {}

            # End of while loop
            logger.info(f"Conversation complete after {turn_count} turns")
            if streamThinkId:
                await atlantis.stream_end(streamThinkId)
                logger.info("Think stream ended successfully")
            if streamTalkId:
                await atlantis.stream_end(streamTalkId)
                logger.info("Talk stream ended successfully")

        except Exception as e:
            logger.error(f"ERROR calling Anthropic: {str(e)}")
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

            await atlantis.owner_log(f"Error calling Anthropic: {error_details}")
            for sid in [streamThinkId, streamTalkId]:
                if sid:
                    try:
                        await atlantis.stream_end(sid)
                    except:
                        pass
            raise

        logger.info(f"=== CHAT COMPLETED === session={sessionId} request={requestId} turns={turn_count}")

    finally:
        atlantis.session_shared.remove(_BUSY_KEY)
        logger.info(f"🔓 RELEASED: session={sessionId} request={requestId}")

    return
