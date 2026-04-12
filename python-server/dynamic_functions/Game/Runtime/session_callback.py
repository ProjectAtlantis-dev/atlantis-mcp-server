import atlantis
import logging

from dynamic_functions.Bot.Runtime.common import (
    logger,
    analyze_participants,
    fetch_transcript,
)
from dynamic_functions.Game.Runtime.common import spawn_bot, _load_bot_config
from dynamic_functions.Game.Runtime.roles import get_role_for_bot


@session
async def session_callback():
    """Fires on session reconnect — checks room state and re-spawns the bot if needed."""
    game_id = atlantis.get_game_id()
    caller = atlantis.get_caller() or "unknown"
    logger.info(f"🔄 Session reconnect: game={game_id} caller={caller}")

    # Fetch transcript and check if a bot is already present
    raw_transcript, _ = await fetch_transcript(caller)
    analysis = analyze_participants(raw_transcript)
    participants = analysis.get('participants', {})

    # Find known bots in the room (anyone who isn't the caller with a bot config)
    bot_sids = []
    for sid in participants:
        if sid != caller:
            cfg, _ = _load_bot_config(sid)
            if cfg:
                bot_sids.append(sid)

    logger.info(f"🔄 Room state: participants={list(participants.keys())}, bot_sids={bot_sids}")

    if not bot_sids:
        # Room is empty — the game scenario already wrote the roster,
        # but we don't know which role to re-spawn without scanning.
        # This shouldn't normally happen since the game callback spawns the bot.
        raise RuntimeError("No bot in room on reconnect — game scenario should have spawned one")
    else:
        logger.info(f"🔄 Bot(s) already in room: {bot_sids}")
