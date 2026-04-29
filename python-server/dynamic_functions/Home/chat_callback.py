"""Game chat callback — routes messages to whichever bot is in the room.

Flow:
1. Fetch transcript and detect which bot is already in the room (by sid)
2. Look up the bot's current role assignment from the game roster
3. Build a bot chat context with game-owned procedure hooks
4. Dispatch to the bot's configured chat handler

The bot's identity is determined by who's actually present.
Procedure text belongs to the Games.<Location> module; the bot runtime
decides whether to apply it.
"""

import atlantis
import time as _t
from importlib import import_module

from dynamic_functions.Home.chat_common import (
    logger,
    fetch_transcript,
)
from dynamic_functions.Home.chat import BotChatContext, dispatch_chat
from dynamic_functions.Home.common import _load_bot_config


_BUSY_KEY = "chat_busy"
_PURPLE = "\033[1;35m"
_RESET = "\033[0m"




# =========================================================================
# Chat callback
# =========================================================================


async def chat_callback():
    """Main chat callback — routes to the right bot via player data."""
    session_id = atlantis.get_session_id() or "unknown"
    request_id = atlantis.get_request_id() or "unknown"
    game_id = atlantis.get_game_id()
    caller = atlantis.get_caller()
    if not caller:
        raise ValueError("Chat callback fired without a caller identity")

    logger.info(f"Chat: session={session_id} request={request_id} game={game_id} caller={caller}")

    # Busy lock
    owner_req = atlantis.session_shared.get(_BUSY_KEY)
    if owner_req:
        logger.warning(f"Chat busy: session={session_id} owned by {owner_req}, skipping {request_id}")
        return

    atlantis.session_shared.set(_BUSY_KEY, request_id)
    try:
        return await _handle_chat(session_id, request_id, game_id, caller)
    finally:
        atlantis.session_shared.remove(_BUSY_KEY)


def _find_bot_in_room(raw_transcript, caller):
    """Detect which bot is in the room by scanning transcript participants.

    Returns (bot_sid, bot_cfg, folder) or (None, None, None) if no bot found.
    The bot is any chat participant whose sid matches a known bot config,
    excluding the caller and 'system'.
    """
    # Collect all sids that have spoken (excluding caller and system)
    participant_sids = set()
    for msg in raw_transcript:
        if msg.get('type') != 'chat':
            continue
        sid = msg.get('sid')
        if sid and sid != 'system' and sid != caller:
            participant_sids.add(sid)

    # Try to match each participant sid to a known bot config
    for sid in participant_sids:
        loaded = _load_bot_config(sid)
        if loaded:
            cfg, folder = loaded
            return sid, cfg, folder

    return None, None, None


async def _build_procedure_injections(context):
    """Build transcript injections by delegating to the location content module."""
    role = context.role
    caller = context.caller
    location = context.location
    guest = context.guest

    if not role.get("requiresCheckin"):
        return []

    module_name = f"dynamic_functions.Games.{location}.checkin"
    try:
        checkin_mod = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Role {role['id']} requires check-in but {module_name} is missing") from exc

    build_checkin_injections = getattr(checkin_mod, "build_checkin_injections", None)
    if not callable(build_checkin_injections):
        raise RuntimeError(f"Role {role['id']} requires check-in but {module_name}.build_checkin_injections is missing")

    result = build_checkin_injections(caller, guest, context.interaction)
    # Support both sync and async build_checkin_injections
    if hasattr(result, '__await__'):
        result = await result
    return result


async def _handle_chat(session_id, request_id, game_id, caller):

    # Fetch transcript \u2014 we need it to detect the bot in the room
    t0 = _t.monotonic()
    raw_transcript, transcript = await fetch_transcript(caller)
    logger.info(f"Transcript fetched in {_t.monotonic() - t0:.2f}s ({len(raw_transcript)} raw, {len(transcript)} filtered)")

    # Detect which bot is in the room from transcript participants
    bot_sid, bot_cfg, _ = _find_bot_in_room(raw_transcript, caller)
    if not bot_cfg:
        raise RuntimeError("No bot detected in room — game_callback should have spawned one")

    # Look up the role this bot is filling
    role = get_role_for_bot(game_id, bot_sid, caller)
    if not role:
        raise RuntimeError(f"Bot {bot_sid} is in the room but has no assigned role in game {game_id}")

    location = role["location"]
    role_title = role.get("title", "Assistant")
    logger.info(f"Game {game_id}: bot={bot_sid} role={role_title} location={location}")

    context = BotChatContext(
        session_id=session_id,
        request_id=request_id,
        game_id=game_id,
        caller=caller,
        bot_sid=bot_sid,
        bot_cfg=bot_cfg,
        role=role,
        raw_transcript=raw_transcript,
        transcript=transcript,
        procedure_injection_provider=_build_procedure_injections,
    )

    result = await dispatch_chat(context)

    # Re-fetch transcript so debug dump includes the bot's response
    raw_after, _ = await fetch_transcript(caller)
    logger.info(f"Post-turn transcript: {len(raw_after)} entries")

    return result
