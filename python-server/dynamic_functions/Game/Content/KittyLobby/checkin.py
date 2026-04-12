"""KittyLobby arrival check-in tools.

These are the MCP-visible tools that Kitty uses to walk a new guest
through the front-desk security procedure in KittyLobby.

Guest data lives in Data/players/{username}.json.
"""

import atlantis
import logging

from dynamic_functions.Data.main import (
    get_guest,
    get_visit_info as _get_visit_info,
    is_cleared,
    record_new_conversation as _record_new_conversation,
    register_guest as _register_guest,
    list_all_guests,
)

logger = logging.getLogger("mcp_server")

LOCATION = "KittyLobby"


# =========================================================================
# Helpers (importable by other modules)
# =========================================================================

def get_visit_info(username: str) -> tuple[int, str]:
    return _get_visit_info(username)


def is_checkin_complete(username: str) -> bool:
    return is_cleared(username)


def record_new_conversation(username: str) -> None:
    _record_new_conversation(username, location=LOCATION)


# =========================================================================
# Visible tools
# =========================================================================

@visible
async def list_guests():
    """Returns a list of all known guests."""
    logger.info("list_guests called")
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
    logger.info(f"guest_info called for: {username}")
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
    logger.info("get_guest_checklist called")
    return [
        {"id": "greet",      "status": "pending", "content": "Greet the guest warmly and introduce yourself — you're Kitty, the front desk intern."},
        {"id": "drink",      "status": "pending", "content": "Offer them a warm drink (cocoa, coffee, tea) and ask if it's their first time in Greenland or at Atlantis."},
        {"id": "paperwork",  "status": "pending", "content": "Ask to see their security paperwork — the signed entry authorization form and their security card. Be friendly but firm — you MUST receive their paperwork before proceeding. No exceptions."},
        {"id": "verify",     "status": "pending", "content": "Once they hand over their paperwork, search for 'verify_paperwork' on your console and call it to read their security card."},
        {"id": "register",   "status": "pending", "content": "After verification, ask for their real first name, then search for 'register_guest' on your console and call it with their username and first name to finish check-in."},
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
