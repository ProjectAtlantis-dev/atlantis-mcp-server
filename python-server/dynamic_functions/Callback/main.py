import atlantis
import os

from openai import OpenAI
from typing import List, Dict, Any
from datetime import datetime

from dynamic_functions.Callback.common import (
    logger,
    BOT_SID, BOT_SESSION_PREFIX,
    BOTS, next_bot, add_bot, remove_bot, list_bots,
    get_session_tools, _busy_key,
    fetch_transcript, find_last_chat_entry,
    build_system_prompt, handle_dir_tool,
    get_visit_info, record_new_conversation, is_checkin_complete,
)
from dynamic_functions.Callback.bot import run_bot_turn
from dynamic_functions.Tools.todo import list_tasks as _list_tasks, _read_store


@visible
async def show_tools() -> List[Dict[str, Any]]:
    """Show Kitty's current tool inventory for this session."""
    tools, lookup = get_session_tools(0)
    simple: List[Dict[str, Any]] = []
    for t in tools:
        fn = t['function']
        params = fn.get('parameters', {}).get('properties', {})
        parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get('type', 'any')
            if isinstance(ptype, list):
                ptype = ','.join(ptype)
            parts.append(f"{pname}:{ptype}")
        sig = ', '.join(parts)
        simple.append({
            'name': f"{fn['name']} ({sig})",
            'description': fn.get('description', ''),
        })
    logger.info(f"show_tools: {len(simple)} tools")
    return simple


@visible
async def show_todos():
    """Show Kitty's current todo/task list for this session."""
    return await _list_tasks()


@visible
async def bot_list():
    """List all bots in the chat pool."""
    return list_bots()


@visible
async def bot_add(model: str, base_url: str = "https://openrouter.ai/api/v1",
                  api_key_env: str = "OPENROUTER_API_KEY"):
    """Add a bot to the chat pool. Provide the model name (e.g. 'anthropic/claude-sonnet-4')."""
    bot = add_bot(model, base_url, api_key_env)
    return {"added": bot, "pool_size": len(BOTS)}


@visible
async def bot_remove(index: int):
    """Remove a bot from the chat pool by its index number (see bot_list)."""
    removed = remove_bot(index)
    return {"removed": removed, "pool_size": len(BOTS)}


# no location since this is catch-all chat
# no app since this is catch-all chat
@chat
async def chat():
    """Main chat function"""
    sessionId = atlantis.get_session_id() or "unknown"
    requestId = atlantis.get_request_id() or "unknown"
    caller: str = atlantis.get_caller()  # type: ignore[assignment]
    if not caller:
        raise ValueError("No caller identity available — cannot process chat without a caller")

    logger.info("=" * 60)
    logger.info(f"=== CHAT TRIGGERED === session={sessionId} request={requestId} caller={caller}")

    # Check if this session+shell is already being handled
    busy_key = _busy_key()
    owner_req = atlantis.session_shared.get(busy_key)
    if owner_req:
        logger.warning(f"🔒 BUSY: session={sessionId} shell key={busy_key} already owned by request={owner_req}, this request={requestId} — skipping")
        await atlantis.owner_log(f"Skipping chat — session {sessionId} busy (owned by request {owner_req})")
        return

    atlantis.session_shared.set(busy_key, requestId)
    logger.info(f"🔒 ACQUIRED: session={sessionId} shell key={busy_key} by request={requestId}")

    try:
        import time as _t

        # Load base prompt directly via import
        logger.info(f">>> Loading SYSTEM_PROMPT via direct import...")
        t0 = _t.monotonic()
        from dynamic_functions.Bot.Kitty.system_prompt import SYSTEM_PROMPT
        base_prompt = await SYSTEM_PROMPT()
        if not base_prompt or not str(base_prompt).strip():
            raise ValueError("SYSTEM_PROMPT returned empty — cannot proceed without a system prompt")
        base_prompt = str(base_prompt)
        logger.info(f"<<< SYSTEM_PROMPT loaded in {_t.monotonic() - t0:.2f}s ({len(base_prompt)} chars)")

        # Fetch and transform transcript
        logger.info(f">>> Fetching transcript...")
        t0 = _t.monotonic()
        rawTranscript, transcript = await fetch_transcript()
        logger.info(f"<<< Transcript fetched in {_t.monotonic() - t0:.2f}s ({len(rawTranscript)} raw, {len(transcript)} filtered)")

        # Don't respond if last chat message was from the bot
        last_chat_entry = find_last_chat_entry(rawTranscript)
        if last_chat_entry:
            logger.info(f"  Last chat entry: sid={last_chat_entry.get('sid')} type={last_chat_entry.get('type')} content={str(last_chat_entry.get('content',''))[:80]}")
        if last_chat_entry and last_chat_entry.get('type') == 'chat' and last_chat_entry.get('sid') == BOT_SID:
            logger.warning(f"\x1b[38;5;204mLast chat was from kitty — skipping (session={sessionId} request={requestId})\x1b[0m")
            await atlantis.owner_log(f"Skipping response - last chat was from kitty")
            return

        # Check visit info before recording, so we can detect time gaps
        prev_count, prev_last_visit = get_visit_info(caller)
        logger.info(f"Visitor: {caller}, visit #{prev_count}, last visit: {prev_last_visit or 'first time'}")

        # Pick next bot — round-robin through BOTS pool
        if not BOTS:
            logger.info("No bots registered — nothing to do")
            return
        bot_index, bot_cfg = next_bot()
        model = bot_cfg["model"]
        logger.info(f"Selected bot [{bot_index}/{len(BOTS)}]: model={model}")

        api_key = os.getenv(bot_cfg["api_key_env"])
        if not api_key:
            error_msg = f"{bot_cfg['api_key_env']} environment variable is not set"
            logger.error(error_msg)
            await atlantis.owner_log(error_msg)
            raise ValueError(error_msg)

        client = OpenAI(
            api_key=api_key,
            base_url=bot_cfg["base_url"],
        )

        # Check if guest needs the new guest procedure:
        # Either brand new (count == 0) or never completed check-in (no greeted + first_name)
        needs_checkin = prev_count == 0 or not is_checkin_complete(caller)
        if needs_checkin:
            logger.info(f"Guest needs check-in: prev_count={prev_count}, checkin_complete={is_checkin_complete(caller)}")
            # Check if the guest already has a checklist loaded from a previous turn
            existing_todos = _read_store(caller)
            if existing_todos:
                # Checklist already loaded — tell the LLM to continue where it left off
                transcript.append({'role': 'system', 'content': [{'type': 'text', 'text':
                    "[PROCEDURE IN PROGRESS] This guest is mid-check-in. Your checklist is already loaded in the `todo` tool — do NOT call `get_guest_checklist` again. Call `todo` (no arguments) to see your current progress, then continue working through the remaining pending steps. Use `todo` with merge=true to update each step's status as you go. You do NOT know their name or username yet unless you have already verified their paperwork."
                }]})
                logger.info("Injected continue-checkin directive (existing todos found)")
            else:
                # No checklist yet — tell the LLM to load one
                transcript.append({'role': 'system', 'content': [{'type': 'text', 'text':
                    "[PROCEDURE REQUIRED] This is an unidentified guest who has NOT completed check-in. Your FIRST action MUST be to call `Tools__get_guest_checklist` to get the check-in steps. It returns a JSON array — pass that array directly to the `todo` tool to load your checklist. Then use `todo` with merge=true to mark each step in_progress then completed as you work through them. Do NOT greet or say anything until your checklist is loaded. You do NOT know their name or username yet — that will be revealed when you verify their paperwork."
                }]})
                logger.info("Injected new-checkin directive (no existing todos)")
            logger.info(f"Injected new guest procedure directive for caller={caller}")

        # If more than an hour since last visit (or first visit), stamp the convo start time
        if not prev_last_visit:
            record_new_conversation(caller)
        else:
            try:
                elapsed = datetime.now() - datetime.fromisoformat(prev_last_visit)
                if elapsed.total_seconds() > 3600:
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                    gap_msg = f"[Some time has passed since your last interaction. The current date and time is now {now_str}.]"
                    transcript.append({'role': 'user', 'content': [{'type': 'text', 'text': gap_msg}]})
                    record_new_conversation(caller)
                    logger.info(f"Injected time-gap message: {gap_msg}")
            except (ValueError, TypeError):
                pass

        # Re-read visit count after recording so it's up-to-date for the system prompt,
        # but use the *previous* last_visit so we don't confuse "just recorded now" with
        # "they were just here moments ago"
        visit_count, _ = get_visit_info(caller)

        # Build system prompt string (once, reused each turn)
        # Hide caller identity from system prompt if guest hasn't completed check-in
        prompt_caller = "" if needs_checkin else caller
        system_prompt = build_system_prompt(
            base_prompt,
            prompt_caller, visit_count, prev_last_visit
        )

        # Get or initialize per-session tool inventory (scoped per bot)
        converted_tools, tool_lookup = get_session_tools(bot_index)
        logger.info(f"Session tool inventory: {len(converted_tools)} tools, {len(tool_lookup)} in lookup")

        # Pre-load get_guest_checklist tool for visitors who haven't completed check-in
        if needs_checkin:
            _, converted_tools, tool_lookup = await handle_dir_tool(
                "get_guest_checklist", converted_tools, tool_lookup
            )
            logger.info("Pre-loaded get_guest_checklist tool for anonymous visitor")

        # Hand off to the streaming bot loop
        logger.info(f"=== HANDING OFF TO BOT === session={sessionId} request={requestId}")
        return await run_bot_turn(
            client=client,
            model=model,
            system_prompt=system_prompt,
            transcript=transcript,
            converted_tools=converted_tools,
            tool_lookup=tool_lookup,
            sessionId=sessionId,
            requestId=requestId,
        )

    finally:
        atlantis.session_shared.remove(busy_key)
        logger.info(f"🔓 RELEASED: session={sessionId} shell key={busy_key} request={requestId}")


@tick
async def tick():
    pass
