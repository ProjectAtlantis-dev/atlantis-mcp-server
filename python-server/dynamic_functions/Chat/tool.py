"""Tool format conversion — Atlantis search results to OpenAI function-calling format."""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, NotRequired, cast

from utils import format_json_log, parse_search_term

logger = logging.getLogger("mcp_client")


class ToolT(TypedDict, total=False):
    """Cloud tool record (legacy)"""
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


class AtlantisSearchToolT(TypedDict, total=False):
    """Subset of Atlantis search/dir result fields needed for LLM tool conversion."""
    tool_name: str
    tool_description: str
    tool_app: str
    tool_location: str
    input_schema: str
    remote_owner: str
    remote_name: str


class ToolSchemaPropertyT(TypedDict, total=False):
    type: str
    description: str
    enum: List[str]


class ToolSchemaT(TypedDict):
    type: str
    properties: Dict[str, ToolSchemaPropertyT]
    required: NotRequired[List[str]]


class OpenAIFunction(TypedDict):
    name: str
    description: str
    parameters: ToolSchemaT


class OpenAITool(TypedDict):
    type: str
    function: OpenAIFunction


class SimpleToolT(TypedDict, total=False):
    name: str
    description: str
    input_schema: ToolSchemaT


class ToolLookupInfo(TypedDict):
    searchTerm: str
    filename: str
    functionName: str


def convert_search_tools(
    tools: List[AtlantisSearchToolT],
) -> Tuple[List[OpenAITool], Dict[str, ToolLookupInfo]]:
    """Convert Atlantis search results to OpenAI function-calling format.

    Returns (openai_tools, tool_lookup) where tool_lookup maps the
    sanitised OpenAI name back to the Atlantis search term.
    """
    openai_tools: List[OpenAITool] = []
    tool_lookup: Dict[str, ToolLookupInfo] = {}

    for tool in tools:
        name = tool.get('tool_name', '')
        if not name:
            logger.warning(f"convert_search_tools: skipping tool with no tool_name: {tool}")
            continue

        app = tool.get('tool_app', '')
        full_name = f"{app}__{name}" if app else name
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', full_name)

        raw_schema = tool.get('input_schema', '')
        if raw_schema:
            try:
                schema: ToolSchemaT = cast(ToolSchemaT, json.loads(raw_schema))
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"convert_search_tools: bad input_schema for {name}, defaulting to empty")
                schema = ToolSchemaT(type='object', properties={})
        else:
            schema = ToolSchemaT(type='object', properties={})

        fn: OpenAIFunction = {
            'name': sanitized,
            'description': tool.get('tool_description', ''),
            'parameters': schema,
        }
        ot: OpenAITool = {'type': 'function', 'function': fn}
        openai_tools.append(ot)

        parts = [
            tool.get('remote_owner', ''),
            tool.get('remote_name', ''),
            tool.get('tool_app', ''),
            tool.get('tool_location', ''),
            name,
        ]
        if all(p == '' for p in parts[:-1]):
            search_term = name
        else:
            search_term = '*'.join(parts)

        tool_lookup[sanitized] = {
            'searchTerm': search_term,
            'filename': '',
            'functionName': name,
        }

    logger.info(f"convert_search_tools: {len(tools)} in -> {len(openai_tools)} out")
    return openai_tools, tool_lookup


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


def _repair_json(raw: str) -> Optional[Dict[str, Any]]:
    """Repair common LLM JSON mistakes"""
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
) -> Tuple[List[OpenAITool], List[SimpleToolT], Dict[str, ToolLookupInfo]]:
    out_tools: List[OpenAITool] = []
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
            logger.error(f"\x1b[91m\U0001f6a8 INVALID SEARCH TERM - SKIPPING TOOL \U0001f6a8\x1b[0m")
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

        out_tool: OpenAITool = {
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
