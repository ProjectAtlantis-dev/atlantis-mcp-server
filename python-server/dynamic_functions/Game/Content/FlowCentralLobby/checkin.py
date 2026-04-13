"""FlowCentralLobby arrival check-in tools.

MCP-visible tools the receptionist bot uses to walk a new guest
through the FlowCentral front-desk procedure.

Guest data lives in Data/players/{username}/.
"""

import logging
from datetime import datetime

from dynamic_functions.Data.main import (
    get_guest,
    get_visit_info as _get_visit_info,
    is_cleared,
    record_new_conversation as _record_new_conversation,
    list_all_guests,
)
from dynamic_functions.Data.todo import _read_store

logger = logging.getLogger("mcp_server")

LOCATION = "FlowCentralLobby"


# =========================================================================
# Helpers (importable by other modules)
# =========================================================================

def get_visit_info(username: str) -> tuple[int, str]:
    return _get_visit_info(username)


def is_checkin_complete(username: str) -> bool:
    return is_cleared(username)


def record_new_conversation(username: str) -> None:
    _record_new_conversation(username, location=LOCATION)


def build_checkin_injections(caller: str, guest: dict | None) -> list[dict]:
    """Build runtime procedure prompts for FlowCentralLobby check-in."""
    if guest and guest.get("cleared"):
        last_visit = guest.get("last_visit", "")
        if not last_visit:
            return []
        try:
            elapsed = datetime.now() - datetime.fromisoformat(last_visit)
        except (ValueError, TypeError):
            return []
        if elapsed.total_seconds() <= 3600:
            return []
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        return [{'role': 'user', 'content': [{'type': 'text', 'text':
            f"[Some time has passed since your last interaction. The current date and time is now {now_str}.]"
        }]}]

    first_visit_note = (
        " This is their first known FlowCentral visit; when you reach "
        "the checklist greeting step, acknowledge that briefly and help "
        "them get oriented."
        if guest and guest.get("first_seen_at") and int(guest.get("visit_count") or 0) == 0 else ""
    )

    existing_todos = _read_store(caller)
    if existing_todos:
        text = (
            "[PROCEDURE IN PROGRESS] This guest is mid-check-in. "
            "Your checklist is already loaded in the `todo` tool — "
            "do NOT call `get_guest_checklist` again. Call `todo` "
            "(no arguments) to see your current progress, then "
            "continue working through the remaining pending steps. "
            "Use `todo` with merge=true to update each step's status "
            "as you go."
            f"{first_visit_note}"
        )
    else:
        text = (
            "[PROCEDURE REQUIRED] This is a new guest who "
            "has NOT completed FlowCentral check-in. Your FIRST action MUST be "
            f"to call `find_checklist` with location=\"{LOCATION}\" "
            "to discover and load the check-in checklist tool. Once "
            "the tool appears in your toolkit, call it to get the "
            "check-in steps. It returns a JSON array — pass that array "
            "directly to the `todo` tool to load your checklist. Then "
            "use `todo` with merge=true to mark each step in_progress "
            "then completed as you work through them. Do NOT greet or "
            "say anything until your checklist is loaded."
            f"{first_visit_note}"
        )
    return [{'role': 'system', 'content': [{'type': 'text', 'text': text}]}]


# =========================================================================
# Visible tools
# =========================================================================

@visible
async def list_guests():
    """Returns a list of all known guests."""
    logger.info("FlowCentralLobby list_guests called")
    guests = list_all_guests()
    if not guests:
        return "No guests on record yet."
    return guests


@visible
async def guest_info(username: str):
    """
    Look up all stored data for a guest.

    Args:
        username: The guest's name/identifier
    """
    logger.info(f"FlowCentralLobby guest_info called for: {username}")
    guest = get_guest(username)
    if not guest:
        return f"No record found for {username}. They may be a brand new guest."
    return guest


@visible
async def get_guest_checklist():
    """
    Returns the front-desk check-in checklist for a new FlowCentral guest.
    Call this ONCE, then pass the returned array to the todo tool to load it.
    After loading, use todo(merge=true) to update each step's status as you go.
    Do NOT call this again — just use the todo tool to track progress.
    """
    logger.info("FlowCentralLobby get_guest_checklist called")
    return [
        {"id": "greet", "status": "pending", "content": "Greet the guest warmly and introduce yourself — you're the front desk assistant at FlowCentral."},
        {"id": "overview", "status": "pending", "content": "Search for 'get_overview' on your console and call it to get the platform overview, then walk the guest through what FlowCentral has to offer in your own words. Keep it conversational — hit the highlights, don't just dump the whole thing."},
        {"id": "suggest", "status": "pending", "content": "Suggest they try Page Speed — it's the tool that's live right now. Ask if they have a website URL they'd like to test."},
    ]


