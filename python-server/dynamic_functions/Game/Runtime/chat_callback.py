import atlantis
import logging

from dynamic_functions.Bot.Runtime.common import analyze_participants
from dynamic_functions.Game.Data.main import ensure_player_record

logger = logging.getLogger("mcp_server")


_BUSY_KEY = "chat_busy"


@chat
async def chat_callback():
    """Main chat callback for the game room."""
    session_id = atlantis.get_session_id() or "unknown"
    request_id = atlantis.get_request_id() or "unknown"
    game_id = atlantis.get_game_id()
    caller = atlantis.get_caller()
    if not caller:
        raise ValueError("Chat callback fired without a caller identity")

    logger.info(f"Chat: session={session_id} request={request_id} game={game_id} caller={caller}")

    # Busy lock — one chat at a time per session
    owner_req = atlantis.session_shared.get(_BUSY_KEY)
    if owner_req:
        logger.warning(f"Chat busy: session={session_id} owned by {owner_req}, skipping {request_id}")
        return

    atlantis.session_shared.set(_BUSY_KEY, request_id)
    try:
        return await _handle_chat(session_id, request_id, game_id, caller)
    finally:
        atlantis.session_shared.remove(_BUSY_KEY)


async def _handle_chat(session_id, request_id, game_id, caller):
    # Fetch transcript
    await atlantis.client_command("/silent on")
    raw_transcript = await atlantis.client_command("/transcript get")
    await atlantis.client_command("/silent off")

    if not raw_transcript:
        raise RuntimeError(f"Empty transcript for game={game_id}")

    analysis = analyze_participants(raw_transcript)
    logger.info(f"Participants: {analysis}")

    player_record, _ = ensure_player_record(caller)
    location = player_record["where"]

    logger.info(f"Location: {location}")

    # TODO: route to location-specific logic
    return {
        "status": "ok",
        "caller": caller,
        "game_id": game_id,
        "location": location,
        "participants": analysis["participants"],
    }
