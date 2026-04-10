import atlantis
import os

from openai import OpenAI

from dynamic_functions.Game.Runtime.common import (
    logger,
    _busy_key,
    fetch_transcript, find_last_chat_entry,
    build_system_prompt, handle_dir_tool,
    get_visit_info, record_new_conversation, is_checkin_complete,
)
from dynamic_functions.Bot.Runtime.turn import run_turn
from dynamic_functions.Misc.todo import _read_store


@chat
async def chat_callback():
    """Main chat function"""
    sessionId = atlantis.get_session_id() or "unknown"
    requestId = atlantis.get_request_id() or "unknown"
    caller: str = atlantis.get_caller()  # type: ignore[assignment]
    if not caller:
        raise ValueError("No caller identity available — cannot process chat without a caller")

    logger.info("=" * 60)
    logger.info(f"=== CHAT TRIGGERED === session={sessionId} request={requestId} caller={caller}")

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
        from importlib import import_module

        logger.info(f">>> Fetching transcript...")
        t0 = _t.monotonic()
        rawTranscript, transcript = await fetch_transcript()
        logger.info(f"<<< Transcript fetched in {_t.monotonic() - t0:.2f}s ({len(rawTranscript)} raw, {len(transcript)} filtered)")

        logger.info("No bots registered for runtime chat yet")
        await atlantis.client_log("No bots in the room right now.")
        return

        last_chat_entry = find_last_chat_entry(rawTranscript)
        if last_chat_entry:
            logger.info(f"  Last chat entry: sid={last_chat_entry.get('sid')} type={last_chat_entry.get('type')} content={str(last_chat_entry.get('content',''))[:80]}")
        if last_chat_entry and last_chat_entry.get('type') == 'chat' and last_chat_entry.get('sid') == bot_sid:
            logger.warning(f"\x1b[38;5;204mLast chat was from {bot_display_name} — skipping (session={sessionId} request={requestId})\x1b[0m")
            await atlantis.owner_log(f"Skipping response - last chat was from {bot_display_name}")
            return

        logger.info(f">>> Loading SYSTEM_PROMPT from {bot_cfg['system_prompt_module']}...")
        t0 = _t.monotonic()
        prompt_mod = import_module(bot_cfg["system_prompt_module"])
        base_prompt = await prompt_mod.SYSTEM_PROMPT()
        if not base_prompt or not str(base_prompt).strip():
            raise ValueError(f"SYSTEM_PROMPT for {bot_display_name} returned empty")
        base_prompt = str(base_prompt)
        logger.info(f"<<< SYSTEM_PROMPT loaded in {_t.monotonic() - t0:.2f}s ({len(base_prompt)} chars)")

        prev_count, prev_last_visit = get_visit_info(caller)
        logger.info(f"Visitor: {caller}, visit #{prev_count}, last visit: {prev_last_visit or 'first time'}")

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

        needs_checkin = prev_count == 0 or not is_checkin_complete(caller)
        if needs_checkin:
            logger.info(f"Guest needs check-in: prev_count={prev_count}, checkin_complete={is_checkin_complete(caller)}")
            existing_todos = _read_store(caller)
            if existing_todos:
                transcript.append({'role': 'system', 'content': [{'type': 'text', 'text':
                    "[PROCEDURE IN PROGRESS] This guest is mid-check-in. Your checklist is already loaded in the `todo` tool — do NOT call `get_guest_checklist` again. Call `todo` (no arguments) to see your current progress, then continue working through the remaining pending steps. Use `todo` with merge=true to update each step's status as you go. You do NOT know their name or username yet unless you have already verified their paperwork."
                }]})
                logger.info("Injected continue-checkin directive (existing todos found)")
            else:
                transcript.append({'role': 'system', 'content': [{'type': 'text', 'text':
                    "[PROCEDURE REQUIRED] This is an unidentified guest who has NOT completed check-in. Your FIRST action MUST be to call `Misc__get_guest_checklist` to get the check-in steps. It returns a JSON array — pass that array directly to the `todo` tool to load your checklist. Then use `todo` with merge=true to mark each step in_progress then completed as you work through them. Do NOT greet or say anything until your checklist is loaded. You do NOT know their name or username yet — that will be revealed when you verify their paperwork."
                }]})
                logger.info("Injected new-checkin directive (no existing todos)")
            logger.info(f"Injected new guest procedure directive for caller={caller}")

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

        visit_count, _ = get_visit_info(caller)

        prompt_caller = "" if needs_checkin else caller
        system_prompt = build_system_prompt(
            base_prompt,
            prompt_caller, visit_count, prev_last_visit
        )

        converted_tools, tool_lookup = get_session_tools(bot_index)
        logger.info(f"Session tool inventory: {len(converted_tools)} tools, {len(tool_lookup)} in lookup")

        if needs_checkin:
            _, converted_tools, tool_lookup = await handle_dir_tool(
                "get_guest_checklist", converted_tools, tool_lookup
            )
            logger.info("Pre-loaded get_guest_checklist tool for anonymous visitor")

        logger.info(f"=== HANDING OFF TO BOT === session={sessionId} request={requestId}")
        return await run_turn(
            client=client,
            model=model,
            bot_sid=bot_sid,
            bot_display_name=bot_display_name,
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
