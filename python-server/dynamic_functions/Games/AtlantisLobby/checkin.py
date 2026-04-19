"""AtlantisLobby arrival check-in tools.

MCP-visible tools Kitty uses to walk a new guest through the
Atlantis front-desk security procedure.

Guest data lives in Data/players/{username}/.
"""

import atlantis
import logging
from datetime import datetime

from dynamic_functions.Data.main import (
    get_guest,
    get_interaction_info as _get_interaction_info,
    is_cleared,
    register_guest as _register_guest,
    list_all_guests,
)
from dynamic_functions.Data.todo import todo_read

logger = logging.getLogger("mcp_server")

LOCATION = "AtlantisLobby"


# =========================================================================
# Helpers (importable by other modules)
# =========================================================================

def get_interaction_info(username: str) -> tuple[int, str]:
    return _get_interaction_info(username)


def is_checkin_complete(username: str) -> bool:
    return is_cleared(username)


async def build_checkin_injections(
    caller: str,
    guest: dict | None,
    interaction: dict | None = None,
) -> list[dict]:
    """Build runtime procedure prompts for AtlantisLobby check-in."""
    if guest and guest.get("cleared"):
        last_interaction_at = (interaction or {}).get("last_interaction_at") or ""
        if not last_interaction_at:
            return []
        try:
            elapsed = datetime.now() - datetime.fromisoformat(last_interaction_at)
        except (ValueError, TypeError):
            return []
        if elapsed.total_seconds() <= 3600:
            return []
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        return [{'role': 'user', 'content': [{'type': 'text', 'text':
            f"[Some time has passed since your last interaction. The current date and time is now {now_str}.]"
        }]}]

    prior_interaction_count = int((interaction or {}).get("prior_interaction_count") or 0)
    game_id = atlantis.get_game_id() or ""
    existing_todos = todo_read(f"AtlantisLobby/{caller}/{game_id}/greeting_todo") if game_id else []

    if existing_todos:
        text = (
            "[PROCEDURE IN PROGRESS] This guest is mid-check-in. "
            "Your checklist is already loaded in the `todo` tool — "
            "do NOT call `get_guest_checklist` again. Call `todo` "
            "(no arguments) to see your current progress, then "
            "continue working through the remaining pending steps. "
            "Use `todo` with merge=true to update each step's status "
            "as you go. You do NOT know their name or username yet "
            "unless you have already verified their paperwork."
        )
    else:
        text = (
            "[PROCEDURE REQUIRED] This is an unidentified guest who "
            "has NOT completed Atlantis check-in. Your FIRST action MUST be "
            f"to call `find_checklist` with location=\"{LOCATION}\" "
            "to discover and load the check-in checklist tool. Once "
            "the tool appears in your toolkit, call it to get the "
            "check-in steps. It returns a JSON array — pass that array "
            "directly to the `todo` tool to load your checklist. Then "
            "use `todo` with merge=true to mark each step in_progress "
            "then completed as you work through them. Do NOT greet or "
            "say anything until your checklist is loaded. You do NOT "
            "know their name or username yet — that will be revealed "
            "when you verify their paperwork."
        )
    return [{'role': 'system', 'content': [{'type': 'text', 'text': text}]}]


# =========================================================================
# Visible tools
# =========================================================================

@visible
async def list_guests():
    """Returns a list of all known guests."""
    logger.info("AtlantisLobby list_guests called")
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
    logger.info(f"AtlantisLobby guest_info called for: {username}")
    guest = get_guest(username)
    if not guest:
        return f"No record found for {username}. They may be a brand new guest."
    return guest


@visible
async def get_guest_checklist():
    """
    Returns the front-desk check-in checklist for a new guest arrival.
    Call this ONCE, then pass the returned array to the todo tool to load it.
    After loading, use todo(merge=true) to update each step's status as you go.
    Do NOT call this again — just use the todo tool to track progress.
    Do NOT ask for the guest's name or username yet — that comes later in the procedure.
    """
    logger.info("AtlantisLobby get_guest_checklist called")
    return [
        {"id": "greet",      "status": "pending", "content": "Greet the guest warmly and introduce yourself — you're the front desk assistant."},
        {"id": "paperwork",  "status": "pending", "content": "Confirm if first visit to Greenland or Atlantis. Ask to see their security paperwork — the signed entry authorization form and their security card. Be friendly but firm — you MUST receive their paperwork before proceeding. No exceptions."},
        {"id": "verify",     "status": "pending", "content": "Once they hand over their paperwork, search for 'verify_paperwork' on your console and call it to read their security card."},
        {"id": "register",   "status": "pending", "content": "After verification, ask for their real first name, then search for 'register_guest' on your console and call it with their username and first name to finish check-in."},
        {"id": "drink",      "status": "pending", "content": "Offer them a warm drink (cocoa, coffee, tea, booze)."},
    ]


@visible
async def verify_paperwork():
    """
    Call this after the guest hands over their security card and entry authorization.
    Reads their security card and reveals the username printed on it.
    Takes no arguments — the card is read automatically.
    """
    username = atlantis.get_caller()
    if not username:
        raise ValueError("Could not read the security card — no caller identity found.")
    logger.info(f"verify_paperwork called — card reads: {username}")
    return (
        f"Paperwork received! The username on their security card is: {username}\n"
        f"If the username is funny or weird, feel free to giggle about it — have fun!\n"
        f"Now ask the guest for their real first name so you can finish check-in.\n"
        f"Once you have it, call register_guest with their username and first name."
    )


@visible
async def register_guest(username: str, first_name: str):
    """
    Final step of new guest check-in. Stores the guest's real first name.

    Args:
        username: The username from their security card
        first_name: The guest's real first name
    """
    logger.info(f"register_guest called for: {username} (first_name={first_name})")
    _register_guest(username, first_name, location=LOCATION)
    return (
        f"Guest {first_name} (username: {username}) has been registered!\n"
        f"Welcome them by name and let them know they're all set."
    )
