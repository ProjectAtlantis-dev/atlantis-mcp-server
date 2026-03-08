from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from state import logger
from utils import format_json_log

if TYPE_CHECKING:
    from server import DynamicAdditionServer


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
            "content": [{"type": "text", "text": msg_content_full}]
        })

    return transcript


async def fetch_lobster_transcript(
    server: "DynamicAdditionServer",
    cloud_client_id: str,
    lobster_request_id: Optional[str],
    user: Optional[str],
    seq_start: int = 2
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch transcript from the cloud client using the same sequence as Bot Kitty."""
    await server.send_awaitable_client_command(
        client_id_for_routing=cloud_client_id,
        request_id=lobster_request_id,
        command="/silent on",
        command_data={},
        seq_num=seq_start,
        entry_point_name="lobster_transcript_silent_on",
        user=user
    )

    try:
        raw_transcript = await server.send_awaitable_client_command(
            client_id_for_routing=cloud_client_id,
            request_id=lobster_request_id,
            command="/transcript get",
            command_data={},
            seq_num=seq_start + 1,
            entry_point_name="lobster_transcript_get",
            user=user
        )
    finally:
        try:
            await server.send_awaitable_client_command(
                client_id_for_routing=cloud_client_id,
                request_id=lobster_request_id,
                command="/silent off",
                command_data={},
                seq_num=seq_start + 2,
                entry_point_name="lobster_transcript_silent_off",
                user=user
            )
        except Exception as silent_off_error:
            logger.warning(f"Failed to disable silent mode after lobster transcript fetch: {silent_off_error}")

    if not isinstance(raw_transcript, list):
        raise ValueError(f"Unexpected transcript payload type: {type(raw_transcript).__name__}")

    transcript = transform_transcript_for_llm(raw_transcript)
    return raw_transcript, transcript


async def handle_local_lobster_tool_call(
    server: "DynamicAdditionServer",
    *,
    tool_name: str,
    params: Dict[str, Any],
    request_id: Any,
    cloud_client_id: str,
    lobster_request_id: Optional[str],
    user: Optional[str]
) -> Dict[str, Any]:
    """Execute a local lobster tool call through the cloud client and return an MCP response."""
    logger.info(f"Sending lobster tool '{tool_name}' to cloud client {cloud_client_id}")
    logger.info(
        f"Lobster tool detected - using lobster request_id ({lobster_request_id}) "
        f"instead of MCP request_id ({request_id})"
    )

    tool_args = params.get("arguments", {}) or {}
    command_data: Dict[str, Any] = {}

    if tool_name == "readme":
        command = "@*Claw*README"
    elif tool_name == "command":
        command_text = tool_args.get("commandText")
        if not command_text:
            raise ValueError("Missing required argument 'commandText' for lobster tool 'command'")
        command = "/" + command_text
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
        seq_num=1,
        entry_point_name=tool_name,
        local_lobster_call=True,
        user=user
    )

    logger.info("Got response from cloud client")
    logger.info(
        f"Response structure: {format_json_log(response) if isinstance(response, (dict, list)) else repr(response)}"
    )

    raw_transcript, transcript = await fetch_lobster_transcript(
        server=server,
        cloud_client_id=cloud_client_id,
        lobster_request_id=lobster_request_id,
        user=user
    )
    logger.info(
        f"Pulled lobster transcript with {len(raw_transcript)} raw entries and {len(transcript)} filtered messages"
    )

    combined_response = {
        "response": response,
        "rawTranscript": raw_transcript,
        "transcript": transcript
    }

    result = {
        "content": [{"type": "text", "text": format_json_log(combined_response, colored=False)}],
        "structuredContent": combined_response
    }
    mcp_response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    }
    logger.info(f"Returning MCP response: {format_json_log(mcp_response)}")
    return mcp_response
