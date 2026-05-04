"""Game chat callback — figure out who spoke, where, and which bot heard it."""

import atlantis
import logging

from dynamic_functions.Home.chat_common import analyze_participants, fetch_transcript
from dynamic_functions.Home.location import position_get, position_query
from dynamic_functions.Home.common import _load_bot_config

logger = logging.getLogger("mcp_server")

_BUSY_KEY = "chat_busy"


@chat
async def chat_callback():
    """chatty kathy"""
    session_id = atlantis.get_session_id() or "unknown"
    request_id = atlantis.get_request_id() or "unknown"
    caller = atlantis.get_caller()
    if not caller:
        await atlantis.client_log("Chat callback fired without a caller identity")
        return

    owner_req = atlantis.session_shared.get(_BUSY_KEY)
    if owner_req:
        logger.warning(f"Chat busy: session={session_id} owned by {owner_req}, skipping {request_id}")
        return

    atlantis.session_shared.set(_BUSY_KEY, request_id)
    try:
        await _handle_chat(caller)
    finally:
        atlantis.session_shared.remove(_BUSY_KEY)


async def _handle_chat(caller: str):
    raw_transcript, transcript = await fetch_transcript(caller)
    logger.info(f"Chat: {len(raw_transcript)} raw / {len(transcript)} filtered")

    participants = analyze_participants(raw_transcript)
    speaker_sid = participants.get("last_speaker")
    if not speaker_sid:
        await atlantis.client_log("No chat speaker found in transcript")
        return

    location = position_get(speaker_sid)
    if not location:
        # this should be a client_error
        await atlantis.client_log(f"📍 {speaker_sid} has no position — nowhere to chat")
        return

    occupants = position_query(location)
    if not occupants or len(occupants) <= 1:
        await atlantis.client_log(f"📍 {speaker_sid} is alone in {location}")
        return

    names = [ch.get("displayName", ch["sid"]) for ch in occupants]
    await atlantis.client_log(f"🏠 Room [{location}]: {', '.join(names)}")

    bots_heard = []
    for ch in occupants:
        sid = ch["sid"]
        if sid == speaker_sid:
            continue
        if not ch.get("isBot"):
            continue
        if _load_bot_config(sid):
            bots_heard.append(ch)

    if not bots_heard:
        await atlantis.client_log("🎤 No bots heard it")
        return

    next_up = bots_heard[0]
    await atlantis.client_log(
        f"🎤 Heard by: {next_up.get('displayName', next_up['sid'])}"
    )
