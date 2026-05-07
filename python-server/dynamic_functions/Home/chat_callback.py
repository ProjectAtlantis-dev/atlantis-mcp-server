"""Game chat callback"""

import atlantis
import logging
import os

from openai import OpenAI

from dynamic_functions.Home.chat_common import (
    analyze_participants, fetch_transcript, get_base_tools,
)
from dynamic_functions.Home.location import position_get, position_query
from dynamic_functions.Home.common import _load_bot_config
from dynamic_functions.Home.character import _load_characters
from dynamic_functions.Home.game import require_game_key
from dynamic_functions.Home.prompt_common import build_system_prompt, load_role_system_prompt
from dynamic_functions.Home.interactions import read_interaction, record_interaction
from dynamic_functions.Home.turn import run_turn

logger = logging.getLogger("mcp_server")

_BUSY_KEY = "chat_busy"


@chat
async def chat_callback():
    """Handle game chat"""
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
        await _handle_chat(caller, session_id, request_id)
    finally:
        atlantis.session_shared.remove(_BUSY_KEY)


async def _handle_chat(caller: str, session_id: str, request_id: str):
    game_key = require_game_key()

    raw_transcript, transcript = await fetch_transcript(caller)
    logger.info(f"Chat: {len(raw_transcript)} raw / {len(transcript)} filtered")

    participants = analyze_participants(raw_transcript)
    speaker_sid = participants.get("last_speaker")
    if not speaker_sid:
        await atlantis.client_log("No chat speaker found in transcript")
        return

    location = position_get(game_key, speaker_sid)
    if not location:
        await atlantis.client_log(f"📍 {speaker_sid} has no position — nowhere to chat")
        return

    occupants = position_query(game_key, location)
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
    bot_sid = next_up["sid"]
    bot_display = next_up.get("displayName", bot_sid)
    await atlantis.client_log(f"🎤 Heard by: {bot_display}")

    await _respond_as_bot(
        game_key=game_key,
        bot_record=next_up,
        speaker_sid=speaker_sid,
        transcript=transcript,
        session_id=session_id,
        request_id=request_id,
    )


def _speaker_first_name(speaker_sid: str) -> str:
    for ch in _load_characters():
        if ch.get("sid") == speaker_sid and not ch.get("isBot", True):
            return ch.get("humanName", "") or ""
    return ""


def _bot_role(bot_sid: str) -> str:
    for ch in _load_characters():
        if ch.get("sid") == bot_sid and ch.get("isBot", True):
            return ch.get("role", "") or ""
    return ""


async def _respond_as_bot(
    *,
    game_key: str,
    bot_record: dict,
    speaker_sid: str,
    transcript: list,
    session_id: str,
    request_id: str,
):
    bot_sid = bot_record["sid"]
    bot_display = bot_record.get("displayName", bot_sid)

    loaded = _load_bot_config(bot_sid)
    if not loaded:
        await atlantis.client_log(f"⚠️ No config for bot {bot_sid}")
        return
    cfg, _folder = loaded

    role = _bot_role(bot_sid)
    if not role:
        await atlantis.client_log(f"⚠️ Bot {bot_sid} has no assigned role")
        return

    base_prompt = load_role_system_prompt(role)

    history = read_interaction(game_key, role, speaker_sid)
    first_name = _speaker_first_name(speaker_sid) or history.get("first_name", "")

    system_prompt = build_system_prompt(
        base_prompt=base_prompt,
        caller=speaker_sid,
        prior_interaction_count=int(history.get("count") or 0),
        last_interaction_at=history.get("last_interaction_at", ""),
        first_name=first_name,
    )

    api_key_env = cfg.get("apiKeyEnv", "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    base_url = cfg.get("baseUrl", "") or None
    model = cfg.get("model", "")
    if not api_key or not model:
        await atlantis.client_log(f"⚠️ Bot {bot_sid} missing model/api key (env={api_key_env})")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    converted_tools, tool_lookup = get_base_tools()

    await run_turn(
        client=client,
        model=model,
        bot_sid=bot_sid,
        bot_display_name=bot_display,
        system_prompt=system_prompt,
        transcript=transcript,
        converted_tools=converted_tools,
        tool_lookup=tool_lookup,
        sessionId=session_id,
        requestId=request_id,
    )

    record_interaction(game_key, role, speaker_sid, first_name=first_name)
