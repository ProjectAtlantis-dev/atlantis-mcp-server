"""Game chat callback — main game tick. Fired on every transcript change."""

import atlantis
import logging
import os

from openai import OpenAI

from dynamic_functions.Home.chat_common import (
    analyze_participants, fetch_transcript, get_base_tools,
)
from dynamic_functions.Home.location import position_get, position_query
from dynamic_functions.Home.common import _load_bot_config
from dynamic_functions.Home.character import _load_characters, is_bot_driven
from dynamic_functions.Home.character import load_character_prompt
from dynamic_functions.Home.prompt_common import build_system_prompt, load_role_system_prompt, load_persona
from dynamic_functions.Home.interactions import read_interaction, record_interaction
from dynamic_functions.Home.turn import run_turn

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
    raw_transcript, transcript = await fetch_transcript(game_key)
    logger.info(f"Chat: {len(raw_transcript)} raw / {len(transcript)} filtered")

    speaker_sid = analyze_participants(raw_transcript).get("last_speaker")
    if not speaker_sid:
        await atlantis.client_log("No chat speaker found in transcript")
        return

    location = position_get(game_key, speaker_sid)
    if not location:
        await atlantis.client_log(f"📍 {speaker_sid} has no position — nowhere to chat")
        return

    occupants = position_query(game_key, location)
    if len(occupants) <= 1:
        await atlantis.client_log(f"📍 {speaker_sid} is alone in {location}")
        return

    names = [ch.get("displayName", ch["sid"]) for ch in occupants]
    await atlantis.client_log(f"🏠 Room [{location}]: {', '.join(names)}")

    bots_heard = [
        ch for ch in occupants
        if ch["sid"] != speaker_sid and is_bot_driven(ch["sid"])
    ]
    if not bots_heard:
        await atlantis.client_log("🎤 No bots heard it")
        return

    next_up = bots_heard[0]
    await atlantis.client_log(f"🎤 Heard by: {next_up.get('displayName', next_up['sid'])}")

    await _respond_as_bot(
        game_key=game_key,
        bot_record=next_up,
        speaker_sid=speaker_sid,
        transcript=transcript,
    )


def _character_field(sid: str, field: str) -> str:
    return next(
        (ch.get(field, "") or "" for ch in _load_characters() if ch.get("sid") == sid),
        "",
    )


async def _respond_as_bot(*, game_key: str, bot_record: dict, speaker_sid: str, transcript: list):
    bot_sid = bot_record["sid"]
    bot_display = bot_record.get("displayName", bot_sid)

    loaded = _load_bot_config(bot_sid)
    if not loaded:
        await atlantis.client_log(f"⚠️ No config for bot {bot_sid}")
        return
    cfg, _folder = loaded

    role = _character_field(bot_sid, "role")
    if not role:
        await atlantis.client_log(f"⚠️ Bot {bot_sid} has no assigned role")
        return

    base_prompt = load_role_system_prompt(role)
    history = read_interaction(game_key, bot_sid, speaker_sid)
    first_name = _character_field(speaker_sid, "displayName") or history.get("first_name", "")

    system_prompt = build_system_prompt(
        base_prompt=base_prompt,
        persona=load_persona(bot_sid),
        character_prompt=load_character_prompt(bot_sid, role),
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
        game_key=game_key,
        client=client,
        model=model,
        bot_sid=bot_sid,
        bot_display_name=bot_display,
        system_prompt=system_prompt,
        transcript=transcript,
        converted_tools=converted_tools,
        tool_lookup=tool_lookup,
    )

    record_interaction(game_key, bot_sid, speaker_sid, first_name=first_name)
