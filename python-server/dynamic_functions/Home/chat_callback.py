"""Game chat callback — main game tick. Fired on every transcript change."""

import atlantis
import logging

from dynamic_functions.Home.chat_common import (
    analyze_participants, fetch_transcript,
)
from dynamic_functions.Home.location import position_get, position_query
from dynamic_functions.Home.casting import casting_for_occupant, is_bot_driven
from dynamic_functions.Home.prompt_common import prompt_assemble
from dynamic_functions.Home.interactions import record_interaction
from dynamic_functions.Home.turn import bot_turn

logger = logging.getLogger("mcp_server")

_BUSY_KEY = "chat_busy"


@chat
async def chat_callback(game_key: str):
    """Game tick: fired on every transcript change. The speaker is read from the transcript itself."""
    if not atlantis.get_session_key():
        logger.warning("chat_callback fired without session context, skipping")
        return

    request_id = atlantis.get_request_id() or "unknown"
    if atlantis.session_shared.get(_BUSY_KEY):
        logger.debug(f"chat_callback busy, skipping {request_id}")
        return

    atlantis.session_shared.set(_BUSY_KEY, request_id)
    try:
        await _handle_chat(game_key)
    finally:
        atlantis.session_shared.remove(_BUSY_KEY)


async def _handle_chat(game_key: str):
    from dynamic_functions.Home.session import session_room

    location = session_room(game_key)

    raw_transcript, transcript = await fetch_transcript(game_key)
    logger.info(f"Chat: {len(raw_transcript)} raw / {len(transcript)} filtered (room={location})")

    occupants = position_query(game_key, location)
    occupant_sids = {ch["occupant"] for ch in occupants}

    speaker_sid = None
    for msg in reversed(raw_transcript):
        if msg.get("type") != "chat":
            continue
        sid = msg.get("sid")
        if not sid or sid == "system":
            continue
        if sid in occupant_sids:
            speaker_sid = sid
            break

    if not speaker_sid:
        await atlantis.client_log(f"📍 No speaker in room [{location}]")
        return

    if len(occupants) <= 1:
        await atlantis.client_log(f"📍 {speaker_sid} is alone in {location}")
        return

    names = [ch.get("displayName", ch["occupant"]) for ch in occupants]
    await atlantis.client_log(f"🏠 Room [{location}]: {', '.join(names)}")

    bots_heard = [
        ch for ch in occupants
        if ch["occupant"] != speaker_sid and is_bot_driven(ch["occupant"])
    ]
    if not bots_heard:
        await atlantis.client_log("🎤 No bots heard it")
        return

    next_up = bots_heard[0]
    await atlantis.client_log(f"🎤 Heard by: {next_up.get('displayName', next_up['occupant'])}")

    await _respond_as_bot(
        game_key=game_key,
        bot_record=next_up,
        speaker_sid=speaker_sid,
        transcript=transcript,
    )



async def greet_entrant(game_key: str, entrant_sid: str, location: str):
    """Fire an in-character greeting from a bot already at `location` toward a newcomer.

    Bot recognition (stranger vs. familiar) falls out of the existing per-bot
    interaction memory injected via build_interaction_context — no special prompt
    needed. Mirrors _handle_chat's "first bot heard responds" pattern.
    """
    if not atlantis.get_session_key():
        logger.warning("greet_entrant called without session context, skipping")
        return
    if atlantis.session_shared.get(_BUSY_KEY):
        logger.debug("greet_entrant: chat busy, skipping")
        return

    occupants = position_query(game_key, location)
    bots_here = [
        ch for ch in occupants
        if ch["occupant"] != entrant_sid and is_bot_driven(ch["occupant"])
    ]
    if not bots_here:
        return

    greeter = bots_here[0]
    request_id = atlantis.get_request_id() or f"greet:{entrant_sid}->{greeter['occupant']}"
    atlantis.session_shared.set(_BUSY_KEY, request_id)
    try:
        _, transcript = await fetch_transcript(game_key)
        await atlantis.client_log(
            f"\U0001f44b {greeter.get('displayName', greeter['occupant'])} greets the arrival"
        )
        await _respond_as_bot(
            game_key=game_key,
            bot_record=greeter,
            speaker_sid=entrant_sid,
            transcript=transcript,
        )
    finally:
        atlantis.session_shared.remove(_BUSY_KEY)


async def _respond_as_bot(*, game_key: str, bot_record: dict, speaker_sid: str, transcript: list):
    bot_sid = bot_record["occupant"]
    system_prompt = prompt_assemble(game_key, bot_sid, speaker_sid)

    await bot_turn(
        bot_sid=bot_sid,
        system_prompt=system_prompt,
        transcript=transcript,
    )

    try:
        speaker_name = casting_for_occupant(game_key, speaker_sid).get("displayName", "")
    except ValueError:
        speaker_name = ""
    record_interaction(game_key, bot_sid, speaker_sid, first_name=speaker_name)
