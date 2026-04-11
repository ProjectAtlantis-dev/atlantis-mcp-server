import atlantis
import logging

from dynamic_functions.Bot.Runtime.common import (
    logger,
    analyze_participants,
    fetch_transcript,
)
from dynamic_functions.Game.Runtime.game_callback import (
    _spawn_bot,
)
from dynamic_functions.Computer.query import _connect


@session
async def session_callback():
    """Fires on session reconnect — checks room state and spawns a bot if needed."""
    game_id = atlantis.get_game_id()
    caller = atlantis.get_caller() or "unknown"
    logger.info(f"🔄 Session reconnect: game={game_id} caller={caller}")

    # Figure out where the user is
    conn = _connect()
    guest = conn.execute("SELECT * FROM guests WHERE username = ?", (caller,)).fetchone()
    conn.close()
    location = guest["location"] if guest else "AtlasLobby"
    if location == "Lobby":
        location = "AtlasLobby"

    # Fetch transcript and check if a bot has already been spawned
    raw_transcript, _ = await fetch_transcript(caller)
    analysis = analyze_participants(raw_transcript)
    participants = analysis.get('participants', {})
    last_speaker = analysis.get('last_speaker')

    # Is there a bot in the room? (anyone who isn't the caller)
    bot_sids = [sid for sid in participants if sid != caller]

    logger.info(f"🔄 Room state: location={location}, participants={list(participants.keys())}, "
                f"bot_sids={bot_sids}, last_speaker={last_speaker}")

    if not bot_sids:
        logger.info(f"🔄 Room empty, spawning Atlas at {location}")
        await _spawn_bot("atlas")
    else:
        logger.info(f"🔄 Bot(s) already in room: {bot_sids}")
