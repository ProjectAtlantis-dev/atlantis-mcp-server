"""Atlas's guest check-in logic.

Determines whether a visitor needs the security/check-in procedure
and returns transcript injections + tool preloads for the runtime.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from dynamic_functions.Game.Content.Lobby.checkin import (
    get_visit_info,
    is_checkin_complete,
    record_new_conversation,
)
from dynamic_functions.Data.todo import _read_store

logger = logging.getLogger("mcp_server")


def get_checkin_context(caller: str) -> Dict[str, Any]:
    """Figure out what Atlas needs to do for this caller.

    Returns {
        'needs_checkin': bool,
        'visit_count': int,
        'last_visit': str or None,
        'injections': list of {'role': str, 'content': str},
        'preload_tools': list of tool names to dir-lookup before the LLM runs,
        'suppress_caller_name': bool,
    }
    """
    prev_count, prev_last_visit = get_visit_info(caller)
    needs_checkin = prev_count == 0 or not is_checkin_complete(caller)

    logger.info(f"Checkin context: caller={caller} visits={prev_count} last={prev_last_visit} needs_checkin={needs_checkin}")

    injections: List[Dict[str, str]] = []
    preload_tools: List[str] = []

    if needs_checkin:
        existing_todos = _read_store(caller)
        if existing_todos:
            injections.append({
                'role': 'system',
                'content': (
                    "[PROCEDURE IN PROGRESS] This guest is mid-check-in. "
                    "Your checklist is already loaded in the `todo` tool — "
                    "do NOT call `get_guest_checklist` again. Call `todo` "
                    "(no arguments) to see your current progress, then "
                    "continue working through the remaining pending steps. "
                    "Use `todo` with merge=true to update each step's status "
                    "as you go. You do NOT know their name or username yet "
                    "unless you have already verified their paperwork."
                ),
            })
        else:
            injections.append({
                'role': 'system',
                'content': (
                    "[PROCEDURE REQUIRED] This is an unidentified guest who "
                    "has NOT completed check-in. Your FIRST action MUST be "
                    "to call `Game_Content_Lobby__get_guest_checklist` to get the check-in "
                    "steps. It returns a JSON array — pass that array directly "
                    "to the `todo` tool to load your checklist. Then use `todo` "
                    "with merge=true to mark each step in_progress then "
                    "completed as you work through them. Do NOT greet or say "
                    "anything until your checklist is loaded. You do NOT know "
                    "their name or username yet — that will be revealed when "
                    "you verify their paperwork."
                ),
            })
            preload_tools.append("get_guest_checklist")

    # Record the visit
    if not prev_last_visit:
        record_new_conversation(caller)
    elif prev_last_visit:
        try:
            elapsed = datetime.now() - datetime.fromisoformat(prev_last_visit)
            if elapsed.total_seconds() > 3600:
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                injections.append({
                    'role': 'user',
                    'content': f"[Some time has passed since your last interaction. The current date and time is now {now_str}.]",
                })
                record_new_conversation(caller)
        except (ValueError, TypeError):
            pass

    visit_count, _ = get_visit_info(caller)

    return {
        'needs_checkin': needs_checkin,
        'visit_count': visit_count,
        'last_visit': prev_last_visit,
        'injections': injections,
        'preload_tools': preload_tools,
        'suppress_caller_name': needs_checkin,
    }
