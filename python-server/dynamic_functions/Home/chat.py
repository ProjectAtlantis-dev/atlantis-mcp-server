"""Chat entry point — determine room occupants and next speaker."""

import atlantis
import logging

from dynamic_functions.Home.location import get_player_position, get_players_at
from dynamic_functions.Home.location import position_query
from dynamic_functions.Home.bot_common import analyze_participants, fetch_transcript

logger = logging.getLogger("mcp_server")



async def chat():
    """Chat"""
    caller = atlantis.get_caller()
    if not caller:
        logger.warning("Chat fired without a caller identity")
        return

    raw_transcript, transcript = await fetch_transcript(caller)
    logger.info(
        "Chat transcript fetched: %s raw entries, %s filtered entries",
        len(raw_transcript),
        len(transcript),
    )

    participants = analyze_participants(raw_transcript)
    speaker_sid = participants.get("last_speaker")
    if not speaker_sid:
        await atlantis.client_log("No chat speaker found in transcript")
        return

    # Where is the most recent speaker?
    location = get_player_position(speaker_sid)
    if not location:
        await atlantis.client_log(f"📍 {speaker_sid} has no position — nowhere to chat")
        return

    # Who else is here?
    occupants = position_query(location)
    if not occupants:
        await atlantis.client_log(f"📍 {speaker_sid} is alone in {location}")
        return

    # Build a list of everyone in the room (with display names)
    names = []
    bots = []
    for ch in occupants:
        display = ch.get("displayName", ch["sid"])
        names.append(display)
        if ch.get("isBot") and ch["sid"] != speaker_sid:
            bots.append(ch)

    await atlantis.client_log(
        f"🏠 Room [{location}]: {', '.join(names)}"
    )

    # Next to speak: first bot in the room that isn't the caller
    if bots:
        next_up = bots[0]
        await atlantis.client_log(
            f"🎤 Next to speak: {next_up.get('displayName', next_up['sid'])}"
        )
    else:
        await atlantis.client_log("🎤 No bots present")
