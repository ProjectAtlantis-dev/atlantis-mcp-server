"""Game chat callback — routes messages to the active lobby bot.

Flow:
1. Figure out where the user is (player JSON or default AtlasLobby)
2. Use the static bot for that lobby
3. Load the bot's system prompt
4. Gather tools from role_tools + location_tools
5. Build transcript with any checkin injections
6. Hand off to run_turn
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


_BUSY_KEY = "chat_busy"








# =========================================================================
# Bot config — maps bot sid to runtime config

# =========================================================================

def _load_bot_config(bot_sid):
    """Load bot config from its config.json file."""
    import json
    # Look for config.json under Bot/Content/*/
    bots_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Bot", "Content")
    for entry in os.listdir(bots_dir):
        config_path = os.path.join(bots_dir, entry, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("sid") == bot_sid:
                return cfg
    return None





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


async def _handle_chat(session_id, request_id, game_id, caller):
    # Where is the user?
    guest = get_guest(caller)
    location = guest.get("location") if guest else "AtlasLobby"
    if not location:
        location = "AtlasLobby"
    logger.info(f"Location: {location}, guest known: {guest is not None}")

    bot_sid = "atlas" if location == "AtlasLobby" else "kitty"
    logger.info(f"Game {game_id}: bot={bot_sid} location={location}")

    # Get bot config from config.json
    bot_cfg = _load_bot_config(bot_sid)
    if not bot_cfg:
        raise RuntimeError(f"No config.json found for bot: {bot_sid}")

    bot_display_name = bot_cfg["displayName"]
    bot_sid_str = bot_cfg["sid"]

    role_name = "Front Desk Assistant" if bot_sid == "atlas" else "Receptionist"
    logger.info(f"Role: {role_name}")

    # Fetch transcript
    t0 = _t.monotonic()
    raw_transcript, transcript = await fetch_transcript(caller)
    logger.info(f"Transcript fetched in {_t.monotonic() - t0:.2f}s ({len(raw_transcript)} raw, {len(transcript)} filtered)")

    # Skip if last message was from this bot
    last_chat = None
    for msg in reversed(raw_transcript):
        if msg.get("type") == "chat" and msg.get("sid") != "system":
            last_chat = msg
            break
    if last_chat and last_chat.get("sid") == bot_sid_str:
        logger.warning(f"Last chat was from {bot_display_name} — skipping")
        return

    # Load system prompt
    t0 = _t.monotonic()
    prompt_mod = import_module(bot_cfg["systemPromptModule"])
    base_prompt = await prompt_mod.SYSTEM_PROMPT()
    if not base_prompt or not str(base_prompt).strip():
        raise ValueError(f"SYSTEM_PROMPT for {bot_display_name} returned empty")
    base_prompt = str(base_prompt)
    logger.info(f"System prompt loaded in {_t.monotonic() - t0:.2f}s ({len(base_prompt)} chars)")

    # Load the bot's own prompt builder (e.g. Bot.Content.Atlas.prompt)
    prompt_builder_module = bot_cfg["systemPromptModule"].rsplit(".", 1)[0] + ".prompt"
    prompt_builder = import_module(prompt_builder_module)
    build_system_prompt = prompt_builder.build_system_prompt

    # Checkin context — does this guest need the procedure?
    needs_checkin = not guest or not guest.get("cleared")
    visit_count = guest.get("visit_count", 0) if guest else 0
    last_visit = guest.get("last_visit", "") if guest else ""

    if needs_checkin:
        existing_todos = _read_store(caller)
        if existing_todos:
            transcript.append({'role': 'system', 'content': [{'type': 'text', 'text':
                "[PROCEDURE IN PROGRESS] This guest is mid-check-in. "
                "Your checklist is already loaded in the `todo` tool — "
                "do NOT call `get_guest_checklist` again. Call `todo` "
                "(no arguments) to see your current progress, then "
                "continue working through the remaining pending steps. "
                "Use `todo` with merge=true to update each step's status "
                "as you go. You do NOT know their name or username yet "
                "unless you have already verified their paperwork."
            }]})
            logger.info("Injected continue-checkin directive")
        else:
            transcript.append({'role': 'system', 'content': [{'type': 'text', 'text':
                "[PROCEDURE REQUIRED] This is an unidentified guest who "
                "has NOT completed check-in. Your FIRST action MUST be "
                f"to call `Game_Content_{location}__get_guest_checklist` to get the check-in "
                "steps. It returns a JSON array — pass that array directly "
                "to the `todo` tool to load your checklist. Then use `todo` "
                "with merge=true to mark each step in_progress then "
                "completed as you work through them. Do NOT greet or say "
                "anything until your checklist is loaded. You do NOT know "
                "their name or username yet — that will be revealed when "
                "you verify their paperwork."
            }]})
            logger.info("Injected new-checkin directive")
    else:
        # Returning guest — check for time gap
        if last_visit:
            try:
                elapsed = datetime.now() - datetime.fromisoformat(last_visit)
                if elapsed.total_seconds() > 3600:
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                    transcript.append({'role': 'user', 'content': [{'type': 'text', 'text':
                        f"[Some time has passed since your last interaction. The current date and time is now {now_str}.]"
                    }]})
            except (ValueError, TypeError):
                pass

    # Build final system prompt with visitor context
    if needs_checkin:
        prompt_caller = ""
        first_name = ""
    else:
        prompt_caller = caller
        first_name = guest.get("first_name", "") if guest else ""
    system_prompt = build_system_prompt(base_prompt, prompt_caller, visit_count, last_visit, first_name=first_name)

    # Tools — fresh base tools each time; LLM discovers via /search
    converted_tools, tool_lookup = get_base_tools()

    # API key
    api_key = os.getenv(bot_cfg["apiKeyEnv"])
    if not api_key:
        raise ValueError(f"{bot_cfg['apiKeyEnv']} environment variable is not set")

    client = OpenAI(
        api_key=api_key,
        base_url=bot_cfg["baseUrl"],
    )

    logger.info(f"=== HANDING OFF TO {bot_display_name} ({role_name} at {location}) === session={session_id}")
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
