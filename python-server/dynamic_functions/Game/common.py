import atlantis
import re
import itertools
import json
import os
import logging

from typing import List, Dict, Any, Optional, TypedDict, Tuple, NotRequired
from datetime import datetime
from jinja2 import Template

from utils import format_json_log, parse_search_term
from dynamic_functions.Tools.todo import TODO_PSEUDO_TOOL, handle_todo_tool, list_tasks as _list_tasks, _read_store
from dynamic_functions.Tools.visitor import get_visit_info, record_new_conversation, is_checkin_complete

logger = logging.getLogger("mcp_client")


# =============================================================================
# Bot Identity — change these to re-skin for a different bot
# =============================================================================
BOT_SID = "kitty"              # sid used in transcript events and stream calls
BOT_DISPLAY_NAME = "Kitty"     # display name shown in chat bubbles
BOT_SESSION_PREFIX = "kitty"   # prefix for session-scoped keys (busy, tools, lookup)

# =============================================================================
# Bot Pool — round-robin through these on each chat invocation
# =============================================================================
BOTS = [
    {
        "model": "z-ai/glm-5",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
]

_bot_cycle = itertools.cycle(range(len(BOTS)))

def next_bot() -> Tuple[int, Dict[str, Any]]:
    """Pick the next bot from the pool. Returns (index, config)."""
    idx = next(_bot_cycle)
    return idx, BOTS[idx]


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


def _repair_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try to fix common LLM JSON mistakes before giving up.

    Handles: single quotes, trailing commas, unquoted keys,
    Python booleans (True/False/None), and single-quoted strings
    with embedded double quotes.
    """
    s = raw.strip()

    # 1. Python literals (True -> true, False -> false, None -> null)
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)

    # 2. Replace single-quoted strings with double-quoted strings
    #    Match: 'some text' but be careful about apostrophes
    s = re.sub(r"(?<=[\[{,:\s])\s*'([^']*?)'\s*(?=[,\]}:])", r'"\1"', s)

    # 3. Trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)

    # 4. Unquoted keys:  { foo: ... } -> { "foo": ... }
    s = re.sub(r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', s)

    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 5. Last resort: ast.literal_eval for Python dict literals
    try:
        import ast
        result = ast.literal_eval(raw.strip())
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    return None


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


# =============================================================================
# Pseudo-tools (search, dir) + handlers
# =============================================================================

SEARCH_PSEUDO_TOOL: TranscriptToolT = {
    'type': 'function',
    'function': {
        'name': 'search',
        'description': 'Search for available tools and commands. Results will be added to your tool list so you can call them. Use this when a user asks you to do something and you don\'t have an appropriate tool yet.',
        'parameters': {
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
}


DIR_PSEUDO_TOOL: TranscriptToolT = {
    'type': 'function',
    'function': {
        'name': 'dir',
        'description': 'Look up tools by name. Use this when you know the specific name of a tool you want to load. Results will be added to your tool list.',
        'parameters': {
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
        tool_name = tool['function']['name']
        if tool_name not in tool_lookup:
            converted_tools.append(tool)
            added_names.append(f"{tool_name}: {tool['function']['description']}")

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
        name = tool['function']['name']
        if name not in tool_lookup:
            converted_tools.append(tool)
            added_names.append(f"{name}: {tool['function']['description']}")

    for name, info in new_lookup.items():
        if name not in tool_lookup:
            tool_lookup[name] = info

    if added_names:
        summary = f"Found {len(added_names)} new tool(s) added to your toolkit:\n" + "\n".join(f"- {n}" for n in added_names)
    else:
        summary = f"Search for '{query}' returned {len(new_tools)} tool(s), but all were already in your toolkit."

    logger.info(f"Search result: {summary}")
    return summary, converted_tools, tool_lookup


# =============================================================================
# Transcript helpers
# =============================================================================

def find_last_chat_entry(transcript):
    """Find the last entry in transcript where type is 'chat' (including thinking entries)."""
    for entry in reversed(transcript):
        if entry.get('type') == 'chat':
            return entry
    return None


async def fetch_transcript() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch raw transcript from server and transform it into OpenAI-compatible format.
    Returns (raw_transcript, processed_transcript) so caller can handle skip logic.
    """
    logger.info("fetch_transcript: /silent on")
    await atlantis.client_command("/silent on")
    logger.info("fetch_transcript: /transcript get")
    raw_transcript = await atlantis.client_command("/transcript get")
    logger.info(f"fetch_transcript: received {len(raw_transcript)} entries")
    await atlantis.client_command("/silent off")
    logger.info("fetch_transcript: /silent off")

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

            role_for_llm = 'assistant' if msg_sid == BOT_SID else 'user'

            transcript.append({'role': role_for_llm, 'content': [{'type': 'text', 'text': msg_content_full}]})
            logger.info(f"       -> INCLUDED as role={role_for_llm} (sid={msg_sid})")
        else:
            logger.info(f"       -> SKIPPED (type != 'chat')")
    logger.info(f"=== END FILTERING: {len(transcript)} messages included ===")

    return raw_transcript, transcript


# =============================================================================
# System prompt building
# =============================================================================

def build_visitor_context(caller: str, visit_count: int, last_visit: str) -> str:
    """Build a visitor context note based on caller info. Returns empty string if no context applies."""
    if not caller or visit_count <= 0:
        return ""

    hour = datetime.now().hour
    late_night = hour >= 22 or hour < 5

    if visit_count == 1:
        if late_night:
            visitor_note = "A new visitor just arrived. It's late — their helicopter probably just arrived behind schedule, likely delayed by weather. Welcome them warmly, they've had a long trip. You don't know their name yet — ask for it naturally."
        else:
            visitor_note = "A new visitor just arrived. They're brand new — introduce yourself, welcome them warmly, and help them get oriented. You don't know their name yet — ask for it naturally."
    elif visit_count <= 5:
        visitor_note = f"{caller} has visited {visit_count} times. They're still fairly new — be friendly and remember they might still be figuring things out."
    else:
        visitor_note = f"{caller} has visited {visit_count} times. They're a regular — skip the intros, be casual, and treat them like a friend."

    if last_visit and visit_count > 1:
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

    return visitor_note


def build_system_prompt(
    base_prompt: str,
    caller: str = "",
    visit_count: int = 0,
    last_visit: str = ""
) -> str:
    """
    Build the system prompt as a plain string.
    """
    parts: List[str] = [base_prompt]

    # Always include current date and time
    parts.append(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Visitor context
    visitor_note = build_visitor_context(caller, visit_count, last_visit)
    if visitor_note:
        parts.append(visitor_note)

    return "\n\n".join(parts)


# =============================================================================
# Session-scoped tool storage
# =============================================================================

def _busy_key() -> str:
    """BUSY lock key scoped to session only — no shell, so concurrent chats on different shells still block."""
    return f"{BOT_SESSION_PREFIX}_busy"

def _tools_key(bot_index: int) -> str:
    return f"{BOT_SESSION_PREFIX}_tools_{bot_index}"

def _lookup_key(bot_index: int) -> str:
    return f"{BOT_SESSION_PREFIX}_lookup_{bot_index}"


def get_session_tools(bot_index: int) -> Tuple[List[TranscriptToolT], Dict[str, ToolLookupInfo]]:
    """Get or initialize tool inventory for the current session + bot."""
    tk = _tools_key(bot_index)
    lk = _lookup_key(bot_index)
    tools = atlantis.session_shared.get(tk)
    lookup = atlantis.session_shared.get(lk)
    if tools is None:
        tools = [SEARCH_PSEUDO_TOOL, DIR_PSEUDO_TOOL, TODO_PSEUDO_TOOL]
        atlantis.session_shared.set(tk, tools)
    if lookup is None:
        lookup = {}
        atlantis.session_shared.set(lk, lookup)
    return tools, lookup
