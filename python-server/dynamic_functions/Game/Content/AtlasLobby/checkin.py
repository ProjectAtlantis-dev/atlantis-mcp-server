"""AtlasLobby arrival check-in tools.

These are the MCP-visible tools Atlas uses to walk a new guest
through the FlowCentral front-desk procedure.
"""

import atlantis
import logging
from datetime import datetime

from dynamic_functions.Computer.query import _connect

logger = logging.getLogger("mcp_server")


def _get_guest(username: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM guests WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_visit_info(username: str) -> tuple[int, str]:
    guest = _get_guest(username)
    if not guest:
        return 0, ""
    return int(guest.get("visit_count") or 0), guest.get("last_visit") or ""


def is_checkin_complete(username: str) -> bool:
    guest = _get_guest(username)
    return bool(guest and guest.get("cleared"))


def record_new_conversation(username: str) -> None:
    conn = _connect()
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT INTO guests (username, visit_count, last_visit, location)
        VALUES (?, 1, ?, 'AtlasLobby')
        ON CONFLICT(username) DO UPDATE SET
            visit_count = visit_count + 1,
            last_visit = ?,
            location = COALESCE(location, 'AtlasLobby')
    """, (username, now, now))
    conn.commit()
    conn.close()
    logger.info(f"New AtlasLobby conversation recorded for {username}")


@visible
async def list_guests():
    """Returns a list of all known guest names from the Computer."""
    logger.info("AtlasLobby list_guests called")
    conn = _connect()
    rows = conn.execute("SELECT username, first_name, visit_count, cleared FROM guests").fetchall()
    conn.close()
    if not rows:
        return "No guests on record yet."
    return [dict(r) for r in rows]


@visible
async def guest_info(username: str):
    """
    Look up all stored data for a guest in the Computer.

    Args:
        username: The guest's name/identifier
    """
    logger.info(f"AtlasLobby guest_info called for: {username}")
    guest = _get_guest(username)
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
    Do NOT ask for the guest's name or username yet — that comes later in the procedure.
    """
    logger.info("AtlasLobby get_guest_checklist called")
    return [
        {"id": "greet", "status": "pending", "content": "Greet the guest warmly and introduce yourself — you're Atlas, the front desk assistant."},
        {"id": "drink", "status": "pending", "content": "Offer them a warm drink (cocoa, coffee, tea) and ask if it's their first time using FlowCentral."},
        {"id": "paperwork", "status": "pending", "content": "Ask to see their security paperwork — the signed entry authorization form and their security card. Be friendly but firm — you MUST receive their paperwork before proceeding. No exceptions."},
        {"id": "verify", "status": "pending", "content": "Once they hand over their paperwork, call `Game_Content_AtlasLobby__verify_paperwork` to read their security card."},
        {"id": "register", "status": "pending", "content": "After verification, ask for their real first name, then call `Game_Content_AtlasLobby__register_guest` with their username and first name to finish check-in."},
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
    logger.info(f"AtlasLobby verify_paperwork called — card reads: {username}")
    return (
        f"Paperwork received. The username on their security card is: {username}\n"
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
    logger.info(f"AtlasLobby register_guest called for: {username} (first_name={first_name})")
    conn = _connect()
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT INTO guests (username, first_name, visit_count, last_visit, cleared, location)
        VALUES (?, ?, 1, ?, 1, 'AtlasLobby')
        ON CONFLICT(username) DO UPDATE SET
            first_name = ?,
            visit_count = visit_count + 1,
            last_visit = ?,
            cleared = 1,
            location = 'AtlasLobby'
    """, (username, first_name, now, first_name, now))
    conn.commit()
    conn.close()
    return (
        f"Guest {first_name} (username: {username}) has been registered.\n"
        f"Welcome them by name and let them know they're all set."
    )
