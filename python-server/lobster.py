import json
import logging
import traceback
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple

import atlantis
import utils
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, ErrorData, Tool
from state import logger
from starlette.websockets import WebSocket, WebSocketDisconnect
from utils import format_json_log, write_tools_debug_file

if TYPE_CHECKING:
    from server import DynamicAdditionServer


lobster = logging.getLogger("lobster")


def get_default_lobster_tools() -> List[Tool]:
    return [
        Tool(
            name="readme",
            description="Get information about how to use Atlantis commands",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="command",
            description="Send a command to Atlantis. The text is passed through as-is (e.g. '/help', '@myFunction()', 'hello').",
            inputSchema={
                "type": "object",
                "properties": {
                    "commandText": {
                        "type": "string",
                        "description": "The exact command text to send, including any prefixes like / or @",
                    }
                },
                "required": ["commandText"],
            },
        ),
        Tool(
            name="chat",
            description="Send a chat message to Atlantis. Use this for conversational messages, not commands.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The chat message to send",
                    }
                },
                "required": ["message"],
            },
        ),
    ]


def get_lobster_tools_for_response(server: "DynamicAdditionServer") -> List[Dict[str, Any]]:
    """Returns lobster tools for local WebSocket connections as serialized dicts."""
    if server.lobster_tools:
        logger.info(f"Returning {len(server.lobster_tools)} lobster tools from cloud welcome event")
        return [t.model_dump(mode="json", exclude_none=True) for t in server.lobster_tools]

    logger.error("No lobster tools available - cloud has not sent lobsterTools in welcome event")
    return []


def get_local_tools_if_lobster_mode(
    server: "DynamicAdditionServer",
    has_cloud_connection: bool,
) -> Optional[List[Tool]]:
    """Return lobster-only tool list for local clients when cloud mode is active."""
    if not has_cloud_connection:
        return None

    if server.lobster_tools:
        logger.info(
            f"Local client request with cloud connection - returning {len(server.lobster_tools)} lobster tools from welcome event"
        )
        return list(server.lobster_tools)

    logger.warning("Local client request with cloud connection but no lobster tools received from welcome event yet")
    return []


def transform_transcript_for_llm(raw_transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mirror Bot Kitty transcript filtering for callers that need chat-only history."""
    transcript: List[Dict[str, Any]] = []

    for msg in raw_transcript:
        if msg.get("type") != "chat":
            continue

        msg_sid = msg.get("sid")
        if msg_sid == "system":
            continue

        msg_content_full = msg.get("content", "")
        if not isinstance(msg_content_full, str) or not msg_content_full.strip():
            continue

        role_for_llm = "assistant" if msg_sid == "kitty" else "user"
        if role_for_llm == "user":
            created_at_str = msg.get("created_at_str", "")
            if created_at_str:
                msg_content_full = f"[{created_at_str}] {msg_content_full}"

        transcript.append({
            "role": role_for_llm,
            "content": [{"type": "text", "text": msg_content_full}],
        })

    return transcript


async def fetch_lobster_transcript(
    server: "DynamicAdditionServer",
    cloud_client_id: str,
    lobster_request_id: Optional[str],
    user: Optional[str],
    seq_start: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch transcript from the cloud client using the same sequence as Bot Kitty."""
    logger.info("🦞 fetch_lobster_transcript: /silent on")
    await server.send_awaitable_client_command(
        client_id_for_routing=cloud_client_id,
        request_id=lobster_request_id,
        command="/silent on",
        command_data={},
        seq_num=seq_start,
        entry_point_name="lobster_transcript_silent_on",
        user=user,
    )

    try:
        logger.info("🦞 fetch_lobster_transcript: /transcript get")
        raw_transcript = await server.send_awaitable_client_command(
            client_id_for_routing=cloud_client_id,
            request_id=lobster_request_id,
            command="/transcript get",
            command_data={},
            seq_num=seq_start + 1,
            entry_point_name="lobster_transcript_get",
            user=user,
        )
        logger.info(f"🦞 fetch_lobster_transcript: got {len(raw_transcript) if isinstance(raw_transcript, list) else '?'} entries")
    finally:
        try:
            logger.info("🦞 fetch_lobster_transcript: /silent off")
            await server.send_awaitable_client_command(
                client_id_for_routing=cloud_client_id,
                request_id=lobster_request_id,
                command="/silent off",
                command_data={},
                seq_num=seq_start + 2,
                entry_point_name="lobster_transcript_silent_off",
                user=user,
            )
        except Exception as silent_off_error:
            logger.warning(f"🦞 Failed to disable silent mode after lobster transcript fetch: {silent_off_error}")

    if not isinstance(raw_transcript, list):
        raise ValueError(f"Unexpected transcript payload type: {type(raw_transcript).__name__}")

    logger.info("=== LOBSTER RAW TRANSCRIPT ===")
    logger.info(format_json_log(raw_transcript))
    logger.info("=== END LOBSTER RAW TRANSCRIPT ===")

    transcript = transform_transcript_for_llm(raw_transcript)
    logger.info("=== LOBSTER FILTERED TRANSCRIPT ===")
    logger.info(format_json_log(transcript))
    logger.info("=== END LOBSTER FILTERED TRANSCRIPT ===")
    return raw_transcript, transcript


async def handle_local_lobster_tool_call(
    server: "DynamicAdditionServer",
    *,
    tool_name: str,
    params: Dict[str, Any],
    request_id: Any,
    cloud_client_id: str,
    lobster_request_id: Optional[str],
    user: Optional[str],
) -> Dict[str, Any]:
    """Execute a local lobster tool call through the cloud client and return an MCP response."""
    logger.info(f"Sending lobster tool '{tool_name}' to cloud client {cloud_client_id}")
    logger.info(
        f"Lobster tool detected - using lobster request_id ({lobster_request_id}) "
        f"instead of MCP request_id ({request_id})"
    )

    tool_args = params.get("arguments", {}) or {}
    command_data: Dict[str, Any] = {}
    message_type = "command"

    if tool_name == "readme":
        # Read MULTIX.md directly from disk instead of routing through ~*Home**README
        # because if multiple remotes are connected, the search term would match multiple
        # README functions and the cloud would return an ambiguity error
        from pathlib import Path
        md_path = Path(__file__).parent / "dynamic_functions" / "Home" / "MULTIX.md"
        readme_text = md_path.read_text()
        combined_response = {"transcript": [], "returnValue": readme_text}
        result = {
            "content": [{"type": "text", "text": format_json_log(combined_response, colored=False)}],
            "structuredContent": combined_response,
        }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
    elif tool_name == "command":
        command_text = tool_args.get("commandText")
        if not command_text:
            raise ValueError("Missing required argument 'commandText' for lobster tool 'command'")
        # Ensure command has a recognized prefix; default to '/' if none present
        if not command_text[0] in ('/', '\\', '%', '@', '~'):
            command_text = '/' + command_text
        command = command_text
    elif tool_name == "chat":
        # Plain text pass-through — no prefix manipulation, sent as message_type="chat"
        message = tool_args.get("message")
        if not message:
            raise ValueError("Missing required argument 'message' for lobster tool 'chat'")
        command = message
        message_type = "chat"
    elif tool_name == "function":
        function_target = tool_args.get("functionName")
        if not function_target:
            raise ValueError("Missing required argument 'functionName' for lobster tool 'function'")
        command = "@" + function_target
    else:
        raise ValueError(f"Unknown lobster tool '{tool_name}'")

    response = await server.send_awaitable_client_command(
        client_id_for_routing=cloud_client_id,
        request_id=lobster_request_id,
        command=command,
        command_data=command_data,
        message_type=message_type,
        seq_num=1,
        entry_point_name=tool_name,
        local_lobster_call=True,
        user=user,
    )

    logger.info("Got response from cloud client")
    logger.info(
        f"Response structure: {format_json_log(response) if isinstance(response, (dict, list)) else repr(response)}"
    )

    raw_transcript, transcript = await fetch_lobster_transcript(
        server=server,
        cloud_client_id=cloud_client_id,
        lobster_request_id=lobster_request_id,
        user=user,
    )
    logger.info(
        f"Pulled lobster transcript with {len(raw_transcript)} raw entries and {len(transcript)} filtered messages"
    )

    combined_response = {
        "transcript": raw_transcript[-5:],
        "returnValue": response,
    }

    result = {
        "content": [{"type": "text", "text": format_json_log(combined_response, colored=False)}],
        "structuredContent": combined_response,
    }
    mcp_response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }
    logger.info(f"Returning MCP response: {format_json_log(mcp_response)}")
    return mcp_response


def apply_cloud_welcome(service_client: Any, data: Any) -> None:
    """Apply cloud welcome payload and update lobster-related state."""
    if isinstance(data, dict):
        owner_usernames = data.get("usernames", [])
        lobster_request_id = data.get("lobsterRequestId")
        # NOTE: lobsterTools from the welcome message is logged but not used.
        # We use hardcoded pseudo tools (readme, command) that pass through to the cloud client.
        lobster_tools_data = data.get("lobsterTools", [])
        lobster_tool_names = [t.get("name", "?") for t in lobster_tools_data] if lobster_tools_data else []

        r = "\033[1;91m"
        x = "\033[0m"
        logger.info("")
        logger.info(f"  {r}{'=' * 59}{x}")
        logger.info(f"  {r}   🦞 WELCOME FROM THE CLOUD! LOBSTER BOAT IS IN PORT! 🦞   {x}")
        logger.info(f"  {r}{'=' * 59}{x}")
        logger.info(f"  {r}   Captain:    {(', '.join(owner_usernames) if owner_usernames else 'unknown'):<44}{x}")
        logger.info(f"  {r}   Trap tag:   {(lobster_request_id or 'MISSING!'):<44}{x}")
        logger.info(f"  {r}   Shell:      {(data.get('shellPath') or 'not provided'):<44}{x}")
        logger.info(f"  {r}   Catch ({len(lobster_tool_names)}):  {str(lobster_tool_names):<44}{x}")
        logger.info(f"  {r}{'=' * 59}{x}")
        logger.info("")

        atlantis._set_owner_usernames(owner_usernames)
        atlantis._set_owner(owner_usernames[0] if owner_usernames else service_client.email)
        shell_path = data.get("shellPath")
        if shell_path:
            service_client.lobster_shell_path = shell_path
            logger.info(f"🦞 Lobster shell path (from welcome): {shell_path}")

        if lobster_request_id:
            service_client.lobster_request_id = lobster_request_id
        else:
            logger.error("FATAL: No lobsterRequestId in welcome message! Cannot operate without it!")
            logger.error(f"Welcome data received: {format_json_log(data)}")
            raise RuntimeError("Cloud welcome message missing required 'lobsterRequestId' - cannot continue")

        logger.info(
            f"Using {len(service_client.mcp_server.lobster_tools)} hardcoded lobster tools: "
            f"{[t.name for t in service_client.mcp_server.lobster_tools]}"
        )
        return

    if isinstance(data, list):
        logger.warning("Welcome message using LEGACY format (array of usernames) - missing lobsterRequestId and lobsterTools!")
        owner_usernames = data
        atlantis._set_owner_usernames(owner_usernames)
        atlantis._set_owner(owner_usernames[0] if owner_usernames else service_client.email)
        return

    logger.warning("Welcome message using LEGACY format (single string) - missing lobsterRequestId and lobsterTools!")
    atlantis._set_owner(data)
    atlantis._set_owner_usernames([data] if data else [])


async def lobster_initialize(
    server: "DynamicAdditionServer",
    params: Optional[Dict[str, Any]] = None,
    *,
    server_version: str,
) -> Dict[str, Any]:
    """Initialize the server when an MCP client connects."""
    params = params or {}
    lobster.info("\033[1;91m🦞 === LOBSTER INITIALIZE ===\033[0m")
    lobster.info(f"\033[1;91m🦞 Server version {server_version}\033[0m")
    utils.set_server_instance(server)
    lobster.info("\033[1;91m🦞 Dynamic functions utility module initialized\033[0m")
    await server._get_tools_list(caller_context="initialize_method")
    lobster.info("\033[1;91m🦞 Lobster initialize complete\033[0m")
    return {
        "protocolVersion": params.get("protocolVersion"),
        "capabilities": {
            "tools": {},
            "prompts": {},
            "resources": {},
            "logging": {},
        },
        "serverInfo": {"name": server.name, "version": server_version},
    }


async def process_mcp_request(
    server: "DynamicAdditionServer",
    request: Dict[str, Any],
    client_id: Optional[str] = None,
    *,
    get_all_tools_for_response: Callable[["DynamicAdditionServer", str], Awaitable[List[Dict[str, Any]]]],
    server_version: str,
) -> Dict[str, Any]:
    """Process an MCP request and return a response."""
    method = request.get("method")
    lobster.debug(f"\033[1;91m🦞 LOBSTER PATH: {method}\033[0m")

    if "id" not in request:
        return {"error": "Missing request ID"}

    req_id = request.get("id")
    params = request.get("params", {})

    try:
        if method == "initialize":
            logger.info(f"Processing 'initialize' request with params:\n{format_json_log(params)}")
            result = await lobster_initialize(server, params, server_version=server_version)
            logger.info("Successfully processed 'initialize' request")
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        if method == "tools/list":
            logger.info("Processing 'tools/list' request via helper for local WebSocket connection")
            lobster_tools_list = get_lobster_tools_for_response(server)
            if not lobster_tools_list:
                logger.error(
                    f"EMPTY TOOL LIST being returned for tools/list request (ID: {req_id})! "
                    f"Cloud has not sent lobsterTools in welcome event. "
                    f"The MCP client will see zero tools available."
                )
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": lobster_tools_list},
            }
            write_tools_debug_file(response)
            lobster_tool_names = [t.get("name", "?") for t in lobster_tools_list]
            logger.info(
                f"Prepared tools/list response (ID: {req_id}) with {len(lobster_tools_list)} lobster tools: {lobster_tool_names}"
            )
            return response
        if method == "tools/list_all":
            logger.info("Processing 'tools/list_all' request via helper")
            all_tools_dict_list = await get_all_tools_for_response(server, "process_mcp_request_websocket")
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": all_tools_dict_list},
            }
            write_tools_debug_file(response)
            return response
        if method == "prompts/list":
            result = await server._get_prompts_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": result}}
        if method == "resources/list":
            result = await server._get_resources_list()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": result}}
        if method == "tools/call":
            if client_id is None:
                return {"jsonrpc": "2.0", "id": req_id, "error": "Missing client_id for tools/call"}
            return await server._handle_tools_call(
                params=params,
                client_id=client_id,
                request_id=req_id,
                for_cloud=False,
            )

        logger.warning(f"Unknown method requested: {method}")
        return {"jsonrpc": "2.0", "id": req_id, "error": f"Unknown method: {method}"}
    except Exception as e:
        logger.error(f"Error processing request '{method}': {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {"jsonrpc": "2.0", "id": req_id, "error": f"Error processing request: {e}"}


async def handle_lobster_socket(
    websocket: WebSocket,
    *,
    mcp_server: "DynamicAdditionServer",
    active_websockets: set,
    client_connections: Dict[str, Dict[str, Any]],
    get_all_tools_for_response: Callable[["DynamicAdditionServer", str], Awaitable[List[Dict[str, Any]]]],
    server_version: str,
) -> None:
    """Handle local MCP websocket connections."""
    await websocket.accept(subprotocol="mcp")

    # Set shell path contextvar for this async task so all log lines show the lobster shell
    lobster_shell = getattr(mcp_server.cloud_client, 'lobster_shell_path', None) if mcp_server.cloud_client else None
    if lobster_shell:
        atlantis.set_shell_path(lobster_shell)

    client_id = f"ws_{websocket.client.host}_{id(websocket)}"
    active_websockets.add(websocket)
    client_connections[client_id] = {"type": "websocket", "connection": websocket}
    connection_count = len(active_websockets)

    has_cloud = any(info.get("type") == "cloud" for info in client_connections.values())
    lobster_count = len(mcp_server.lobster_tools) if mcp_server.lobster_tools else 0
    lobster_names = [t.name for t in mcp_server.lobster_tools] if mcp_server.lobster_tools else []

    r = "\033[1;91m"
    x = "\033[0m"
    lobster.info("")
    if connection_count > 1:
        lobster.info(f"  {r}🦞 ANOTHER LOBSTER IN THE TRAP! ({connection_count} total) 🦞{x}")
    else:
        lobster.info(f"  {r}🦞 FRESH CATCH! NEW MCP CLIENT HAULED ABOARD! 🦞{x}")
    lobster.info(f"  {r}  Host:         {websocket.client.host}{x}")
    lobster.info(f"  {r}  Client ID:    {client_id}{x}")
    lobster.info(f"  {r}  Trap count:   {connection_count}{x}")
    lobster.info(f"  {r}  Cloud:        {'⛵ AYE' if has_cloud else '🌊 NAY'}{x}")
    lobster.info(f"  {r}  Tools in pot: {lobster_count}{x}")
    if lobster_names:
        lobster.info(f"  {r}  The haul:     {lobster_names}{x}")
    lobster.info("")

    try:
        while True:
            message = await websocket.receive_text()

            try:
                request_data = json.loads(message)

                if (
                    isinstance(request_data, dict)
                    and request_data.get("method") == "notifications/commandResult"
                    and "params" in request_data
                ):
                    params = request_data["params"]
                    correlation_id = params.get("correlationId")

                    if hasattr(mcp_server, "awaitable_requests") and correlation_id and correlation_id in mcp_server.awaitable_requests:
                        future = mcp_server.awaitable_requests.pop(correlation_id, None)
                        if future and not future.done():
                            if "result" in params:
                                logger.info(f"Received result for awaitable command (correlationId: {correlation_id})")
                                future.set_result(params["result"])
                            elif "error" in params:
                                client_error_details = params["error"]
                                logger.error(
                                    f"Received error from client for awaitable command (correlationId: {correlation_id}): {client_error_details}"
                                )
                                error_data = ErrorData(
                                    code=INTERNAL_ERROR,
                                    message=f"Client error for command (correlationId: {correlation_id}): {client_error_details}",
                                )
                                future.set_exception(McpError(error_data))
                            else:
                                future.set_result(None)
                            logger.debug(f"Handled notifications/commandResult for {correlation_id}, continuing WebSocket loop.")
                            continue
                        if future and future.done():
                            logger.warning(f"Received commandResult for {correlation_id}, but future was already done. Ignoring.")
                            continue
                        logger.warning(
                            f"Received commandResult for {correlation_id}, but no active future found. Might have timed out. Ignoring."
                        )
                        continue
                    logger.warning(
                        f"Received notifications/commandResult without a valid/pending correlationId: '{correlation_id}'."
                    )

                logger.debug(f"Received (for MCP processing):\n{format_json_log(request_data)}")
                lobster.debug(f"\033[1;91m🦞 TRAP→process_mcp_request: {request_data.get('method')}\033[0m")
                response = await process_mcp_request(
                    mcp_server,
                    request_data,
                    client_id,
                    get_all_tools_for_response=get_all_tools_for_response,
                    server_version=server_version,
                )
                lobster.info(f"\033[1;91m🦞 TRAP→sending response:\n{format_json_log(response)}\033[0m")
                await websocket.send_text(json.dumps(response))

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {message}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect as e:
        lobster.info(f"\033[1;91m🦞 Lobster socket client disconnected: code={e.code}, reason={e.reason}\033[0m")
    except Exception as e:
        lobster.error(f"\033[1;91m🦞 Lobster socket error: {e}\033[0m")
    finally:
        active_websockets.discard(websocket)
        to_remove = []
        for cid, info in client_connections.items():
            if info.get("type") == "websocket" and info.get("connection") is websocket:
                to_remove.append(cid)
        for cid in to_remove:
            client_connections.pop(cid, None)

        connection_count = len(active_websockets)
        lobster.info(f"\033[1;91m🦞 Lobster released back to sea: {websocket.client.host} (Active: {connection_count})\033[0m")
