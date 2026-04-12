"""Game chat callback — routes messages to whichever bot is in the room.

Flow:
1. Fetch transcript and detect which bot is already in the room (by sid)
2. Look up the bot's current Role for location + procedures
3. Load the bot's system prompt; apply role procedures (checkin etc.)
4. Hand off to run_turn

The bot's identity is determined by who's actually present.
Procedures (checkin, etc.) belong to the Role, not the bot or location.
"""

import atlantis
import logging
import os
import time as _t
from datetime import datetime
from importlib import import_module

from openai import OpenAI

from dynamic_functions.Bot.Runtime.common import (
    logger,
    analyze_participants,
    fetch_transcript,
    get_base_tools,
)
from dynamic_functions.Bot.Runtime.turn import run_turn
# prompt module is loaded dynamically per-bot below
from dynamic_functions.Data.main import get_guest
from dynamic_functions.Data.todo import _read_store
from dynamic_functions.Game.Runtime.common import _load_bot_config
from dynamic_functions.Game.Runtime.roles import get_role_for_bot, get_role_for_location


_BUSY_KEY = "chat_busy"




# =========================================================================
# Chat callback
# =========================================================================

@chat
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
        cfg, folder = _load_bot_config(sid)
        if cfg:
            return sid, cfg, folder

    return None, None, None


def _get_checkin_injections(role, caller, location, guest):
    """Build transcript injections for the role's procedures.

    Returns a list of message dicts to append to the transcript.
    All procedure logic flows from the role config.
    """
    injections = []

    if not role.get("requiresCheckin"):
        return injections

    needs_checkin = not guest or not guest.get("cleared")
    last_visit = guest.get("last_visit", "") if guest else ""

    if needs_checkin:
        existing_todos = _read_store(caller)
        if existing_todos:
            injections.append({'role': 'system', 'content': [{'type': 'text', 'text':
                "[PROCEDURE IN PROGRESS] This guest is mid-check-in. "
                "Your checklist is already loaded in the `todo` tool \u2014 "
                "do NOT call `get_guest_checklist` again. Call `todo` "
                "(no arguments) to see your current progress, then "
                "continue working through the remaining pending steps. "
                "Use `todo` with merge=true to update each step's status "
                "as you go. You do NOT know their name or username yet "
                "unless you have already verified their paperwork."
            }]})
        else:
            injections.append({'role': 'system', 'content': [{'type': 'text', 'text':
                "[PROCEDURE REQUIRED] This is an unidentified guest who "
                "has NOT completed check-in. Your FIRST action MUST be "
                f"to call `find_checklist` with location=\"{location}\" "
                "to discover and load the check-in checklist tool. Once "
                "the tool appears in your toolkit, call it to get the "
                "check-in steps. It returns a JSON array \u2014 pass that array "
                "directly to the `todo` tool to load your checklist. Then "
                "use `todo` with merge=true to mark each step in_progress "
                "then completed as you work through them. Do NOT greet or "
                "say anything until your checklist is loaded. You do NOT "
                "know their name or username yet \u2014 that will be revealed "
                "when you verify their paperwork."
            }]})
    else:
        # Returning guest \u2014 check for time gap
        if last_visit:
            try:
                elapsed = datetime.now() - datetime.fromisoformat(last_visit)
                if elapsed.total_seconds() > 3600:
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                    injections.append({'role': 'user', 'content': [{'type': 'text', 'text':
                        f"[Some time has passed since your last interaction. The current date and time is now {now_str}.]"
                    }]})
            except (ValueError, TypeError):
                pass

    return injections


async def _handle_chat(session_id, request_id, game_id, caller):
    guest = get_guest(caller)

    # Fetch transcript \u2014 we need it to detect the bot in the room
    t0 = _t.monotonic()
    raw_transcript, transcript = await fetch_transcript(caller)
    logger.info(f"Transcript fetched in {_t.monotonic() - t0:.2f}s ({len(raw_transcript)} raw, {len(transcript)} filtered)")

    # Detect which bot is in the room from transcript participants
    bot_sid, bot_cfg, _ = _find_bot_in_room(raw_transcript, caller)
    if not bot_cfg:
        raise RuntimeError("No bot detected in room — game_callback should have spawned one")

    # Look up the role this bot is filling
    role = get_role_for_bot(game_id, bot_sid)
    if not role:
        raise RuntimeError(f"Bot {bot_sid} is in the room but has no assigned role in game {game_id}")

    location = role["location"]
    role_title = role.get("title", "Assistant")
    bot_display_name = bot_cfg["displayName"]
    bot_sid_str = bot_cfg["sid"]

    logger.info(f"Game {game_id}: bot={bot_sid} role={role_title} location={location}")

    # Skip if last message was from this bot
    last_chat = None
    for msg in reversed(raw_transcript):
        if msg.get("type") == "chat" and msg.get("sid") != "system":
            last_chat = msg
            break
    if last_chat and last_chat.get("sid") == bot_sid_str:
        logger.warning(f"Last chat was from {bot_display_name} \u2014 skipping")
        return

    # Load system prompt from the bot's content module
    t0 = _t.monotonic()
    prompt_mod = import_module(bot_cfg["systemPromptModule"])
    base_prompt = await prompt_mod.SYSTEM_PROMPT()
    if not base_prompt or not str(base_prompt).strip():
        raise ValueError(f"SYSTEM_PROMPT for {bot_display_name} returned empty")
    base_prompt = str(base_prompt)
    logger.info(f"System prompt loaded in {_t.monotonic() - t0:.2f}s ({len(base_prompt)} chars)")

    prompt_builder_module = bot_cfg["systemPromptModule"].rsplit(".", 1)[0] + ".prompt"
    prompt_builder = import_module(prompt_builder_module)
    build_system_prompt = prompt_builder.build_system_prompt

    # Role procedures \u2014 checkin injections etc.
    injections = _get_checkin_injections(role, caller, location, guest)
    for inj in injections:
        transcript.append(inj)
    if injections:
        logger.info(f"Injected {len(injections)} procedure message(s) from role {role['id']}")

    # Build final system prompt with visitor context
    needs_checkin = role.get("requiresCheckin") and (not guest or not guest.get("cleared"))
    visit_count = guest.get("visit_count", 0) if guest else 0
    last_visit = guest.get("last_visit", "") if guest else ""
    if needs_checkin:
        prompt_caller = ""
        first_name = ""
    else:
        prompt_caller = caller
        first_name = guest.get("first_name", "") if guest else ""
    system_prompt = build_system_prompt(base_prompt, prompt_caller, visit_count, last_visit, first_name=first_name)

    # Tools \u2014 fresh base tools each time; LLM discovers via /search
    converted_tools, tool_lookup = get_base_tools()

    # API key
    api_key = os.getenv(bot_cfg["apiKeyEnv"])
    if not api_key:
        raise ValueError(f"{bot_cfg['apiKeyEnv']} environment variable is not set")

    client = OpenAI(
        api_key=api_key,
        base_url=bot_cfg["baseUrl"],
    )

    logger.info(f"=== HANDING OFF TO {bot_display_name} ({role_title} at {location}) === session={session_id}")
    return await run_turn(
        client=client,
        model=bot_cfg["model"],
        bot_sid=bot_sid_str,
        bot_display_name=bot_display_name,
        system_prompt=system_prompt,
        transcript=transcript,
        converted_tools=converted_tools,
        tool_lookup=tool_lookup,
        sessionId=session_id,
        requestId=request_id,
    )
