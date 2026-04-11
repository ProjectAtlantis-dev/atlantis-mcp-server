"""Game chat callback — routes messages to bots via the Computer.

Flow:
1. Figure out where the user is (Computer or default Lobby)
2. Check games table for existing bot assignment, or random-pick one
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
    get_session_tools,
    handle_dir_tool,
    convert_tools_for_llm,
)
from dynamic_functions.Bot.Runtime.turn import run_turn
from dynamic_functions.Bot.Content.Kitty.prompt import build_system_prompt, build_visitor_context
from dynamic_functions.Computer.query import _connect
from dynamic_functions.Misc.todo import _read_store


_BUSY_KEY = "chat_busy"


# =========================================================================
# Computer queries
# =========================================================================

def _get_guest(username):
    conn = _connect()
    row = conn.execute("SELECT * FROM guests WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_or_create_game(game_id, username, location):
    """Get existing bot assignment for this game, or random-pick one and store it."""
    conn = _connect()

    # Already assigned?
    row = conn.execute("SELECT * FROM games WHERE game_id = ?", (str(game_id),)).fetchone()
    if row:
        conn.close()
        return dict(row)

    # Random pick a bot at this location
    role = conn.execute("""
        SELECT r.name as role_name, r.bot_sid, b.name as bot_name, b.chat
        FROM roles r JOIN bots b ON b.sid = r.bot_sid
        WHERE r.location = ?
        ORDER BY RANDOM() LIMIT 1
    """, (location,)).fetchone()

    if not role:
        conn.close()
        raise RuntimeError(f"No bots assigned to location: {location}")

    now = datetime.now().isoformat()
    conn.execute("""
        INSERT INTO games (game_id, username, bot_sid, location, started)
        VALUES (?, ?, ?, ?, ?)
    """, (str(game_id), username, role["bot_sid"], location, now))
    conn.commit()

    result = {
        "game_id": str(game_id),
        "username": username,
        "bot_sid": role["bot_sid"],
        "location": location,
        "started": now,
        "role_name": role["role_name"],
        "bot_name": role["bot_name"],
        "chat": role["chat"],
    }
    conn.close()
    return result


def _get_bot_info(bot_sid):
    conn = _connect()
    bot = conn.execute("SELECT * FROM bots WHERE sid = ?", (bot_sid,)).fetchone()
    conn.close()
    return dict(bot) if bot else None


def _get_role_for_bot_at_location(bot_sid, location):
    conn = _connect()
    role = conn.execute(
        "SELECT * FROM roles WHERE bot_sid = ? AND location = ?",
        (bot_sid, location)
    ).fetchone()
    conn.close()
    return dict(role) if role else None





# =========================================================================
# Bot config — maps bot sid to runtime config
# TODO: this should probably live in the Computer too eventually
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
    """Main chat callback — routes to the right bot via the Computer."""
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
    guest = _get_guest(caller)
    location = guest["location"] if guest else "Lobby"
    logger.info(f"Location: {location}, guest known: {guest is not None}")

    # Who's handling this game? (random pick on first message, persisted after)
    game = _get_or_create_game(game_id, caller, location)
    bot_sid = game["bot_sid"]
    logger.info(f"Game {game_id}: bot={bot_sid} location={location}")

    # Get bot config from config.json
    bot_cfg = _load_bot_config(bot_sid)
    if not bot_cfg:
        raise RuntimeError(f"No config.json found for bot: {bot_sid}")

    bot_display_name = bot_cfg["displayName"]
    bot_sid_str = bot_cfg["sid"]

    # Get the role this bot is playing at this location
    role = _get_role_for_bot_at_location(bot_sid, location)
    role_name = role["name"] if role else "Unknown"
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

    # Checkin context — does this guest need the procedure?
    needs_checkin = guest is None
    visit_count = guest["visit_count"] if guest else 0
    last_visit = guest["last_visit"] if guest else ""

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
                "to call `get_guest_checklist` to get the check-in "
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

    # Tools — only pre-load what's needed for active procedures
    # Everything else Kitty discovers via search/dir on her console
    converted_tools, tool_lookup = get_session_tools()

    if needs_checkin:
        await atlantis.client_command("/silent on")
        _, converted_tools, tool_lookup = await handle_dir_tool(
            "get_guest_checklist", converted_tools, tool_lookup
        )
        await atlantis.client_command("/silent off")
        logger.info("Pre-loaded get_guest_checklist for unregistered guest")

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
