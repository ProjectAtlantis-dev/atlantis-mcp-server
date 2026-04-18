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
from dynamic_functions.Data.todo import TODO_PSEUDO_TOOL

logger = logging.getLogger("mcp_client")


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

    started_at: str
    remote_updated_at: str


class ToolSchemaPropertyT(TypedDict, total=False):
    type: str
    description: str
    enum: List[str]


class ToolSchemaT(TypedDict):
    type: str
    properties: Dict[str, ToolSchemaPropertyT]
    required: NotRequired[List[str]]


class OpenAIFunctionDefT(TypedDict):
    name: str
    description: str
    parameters: ToolSchemaT


class TranscriptToolT(TypedDict):
    type: str
    function: OpenAIFunctionDefT


class SimpleToolT(TypedDict, total=False):
    name: str
    description: str
    input_schema: ToolSchemaT


class ToolLookupInfo(TypedDict):
    searchTerm: str
    filename: str
    functionName: str


def get_consolidated_full_name(tool: ToolT) -> str:
    remote_owner = tool.get('remote_owner', '')
    remote_name = tool.get('remote_name', '')
    tool_app = tool.get('tool_app', '')
    tool_location = tool.get('tool_location', '')
    tool_name = tool.get('tool_name', '')

    parts = [remote_owner, remote_name, tool_app, tool_location, tool_name]

    if all(p == '' for p in parts[:-1]):
        return parts[-1]

    return '*'.join(parts)


def analyze_participants(raw_transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a participant summary from the raw transcript.

    Returns {
        'participants': {sid: {'who': str, 'last_spoke': str, 'message_count': int}},
        'last_speaker': str or None,  # sid of whoever spoke last
    }
    """
    participants: Dict[str, Any] = {}
    last_speaker: Optional[str] = None

    for msg in raw_transcript:
        if msg.get('type') != 'chat':
            continue
        sid = msg.get('sid')
        if not sid or sid == 'system':
            continue

        timestamp = msg.get('created_at') or msg.get('created_at_str', '')

        if sid not in participants:
            participants[sid] = {
                'who': msg.get('who', sid),
                'last_spoke': timestamp,
                'message_count': 0,
            }

        participants[sid]['last_spoke'] = timestamp
        participants[sid]['message_count'] += 1
        last_speaker = sid

    return {
        'participants': participants,
        'last_speaker': last_speaker,
    }


def _repair_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try to fix common LLM JSON mistakes before giving up."""
    s = raw.strip()
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)
    s = re.sub(r"(?<=[\[{,:\s])\s*'([^']*?)'\s*(?=[,\]}:])", r'"\1"', s)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    s = re.sub(r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', s)

    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    try:
        import ast
        result = ast.literal_eval(raw.strip())
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    return None


def coerce_args_to_schema(args: Dict[str, Any], schema: ToolSchemaT) -> Dict[str, Any]:
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


def _filter_by_allowed_apps(results: List[Dict[str, Any]], allowed_apps: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter search results to only include tools from allowed app prefixes."""
    if not allowed_apps:
        return results
    filtered = []
    for r in results:
        app = (r.get('app', '') or r.get('tool_app', '')).lower()
        if any(app.startswith(prefix.lower()) for prefix in allowed_apps):
            filtered.append(r)
    return filtered


async def handle_search_tool(
    query: str,
    converted_tools: List[TranscriptToolT],
    tool_lookup: Dict[str, ToolLookupInfo],
    allowed_apps: Optional[List[str]] = None,
) -> Tuple[str, List[TranscriptToolT], Dict[str, ToolLookupInfo]]:
    import time as _t
    logger.info(f">>> SEARCH PSEUDO-TOOL: query='{query}' allowed_apps={allowed_apps} — sending /search command...")
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

    results = _filter_by_allowed_apps(results, allowed_apps)
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


async def fetch_transcript(caller: str = "") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch raw transcript from server and transform it into OpenAI-compatible format.
    Pass caller sid to correctly assign 'user' vs 'assistant' roles.
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

            if 'data-metacol=' in msg_content_full or 'bot-table-cell' in msg_content_full:
                logger.warning(f"       -> SKIPPED (contains HTML table data, {len(msg_content_full)} chars)")
                continue

            MAX_ENTRY_SIZE = 4000
            if len(msg_content_full) > MAX_ENTRY_SIZE:
                logger.warning(f"       -> SKIPPED (oversized: {len(msg_content_full)} chars > {MAX_ENTRY_SIZE})")
                continue

            # Caller sid is the user, everything else is a bot
            role_for_llm = 'user' if (caller and msg_sid == caller) else 'assistant' if caller else 'user'

            transcript.append({'role': role_for_llm, 'content': [{'type': 'text', 'text': msg_content_full}]})
            logger.info(f"       -> INCLUDED as role={role_for_llm} (sid={msg_sid})")
        else:
            logger.info(f"       -> SKIPPED (type != 'chat')")
    logger.info(f"=== END FILTERING: {len(transcript)} messages included ===")

    return raw_transcript, transcript


def get_base_tools() -> Tuple[List[TranscriptToolT], Dict[str, ToolLookupInfo]]:
    """Return a fresh set of base pseudo-tools and an empty lookup.

    Tools are discovered via /search at runtime — never cached in session_shared.
    """
    tools = [SEARCH_PSEUDO_TOOL, DIR_PSEUDO_TOOL, TODO_PSEUDO_TOOL]
    lookup: Dict[str, ToolLookupInfo] = {}
    return tools, lookup


