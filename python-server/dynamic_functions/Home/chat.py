"""Chat tools"""

import atlantis
import logging

from dynamic_functions.Home.location import position_get, get_players_at
from dynamic_functions.Home.location import position_query
from dynamic_functions.Home.chat_common import analyze_participants, fetch_transcript
from dynamic_functions.Home.game import require_game_key

logger = logging.getLogger("mcp_server")



async def chat():
    """Chat"""
    caller = atlantis.get_caller()
    if not caller:
        logger.warning("Chat fired without a caller identity")
        return

    game_key = require_game_key()
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

    # Find the speaker location
    location = position_get(game_key, speaker_sid)
    if not location:
        await atlantis.client_log(f"📍 {speaker_sid} has no position — nowhere to chat")
        return

    # Find room occupants
    occupants = position_query(game_key, location)
    if not occupants:
        await atlantis.client_log(f"📍 {speaker_sid} is alone in {location}")
        return

    # Build room participant lists
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

    # Pick the next bot speaker
    if bots:
        next_up = bots[0]
        await atlantis.client_log(
            f"🎤 Next to speak: {next_up.get('displayName', next_up['sid'])}"
        )
    else:
        await atlantis.client_log("🎤 No bots present")
