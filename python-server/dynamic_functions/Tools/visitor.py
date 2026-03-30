import atlantis
import json
import os
import fcntl
import logging
from datetime import datetime
from typing import Tuple

logger = logging.getLogger("mcp_server")

VISITOR_DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'Data', 'visitor_data.json')


# =========================================================================
# Internal helpers
# =========================================================================

def _read_data() -> dict:
    """Read the visitor data file. Returns empty dict if missing."""
    try:
        with open(VISITOR_DATA_FILE, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                raw = f.read()
                return json.loads(raw) if raw.strip() else {}
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except FileNotFoundError:
        return {}


def _normalize_entry(entry) -> dict:
    if isinstance(entry, int):
        return {"count": entry, "last_visit": ""}
    return entry


def set_visitor_flag(username: str, flag: str, value):
    """Set a flag on a visitor's data record."""
    os.makedirs(os.path.dirname(VISITOR_DATA_FILE), exist_ok=True)
    with open(VISITOR_DATA_FILE, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            data = json.loads(raw) if raw.strip() else {}
            entry = _normalize_entry(data.get(username, {"count": 0, "last_visit": ""}))
            entry[flag] = value
            data[username] = entry
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    logger.info(f"Set {flag}={value} for visitor {username}")


# =========================================================================
# Functions used by Kitty backends
# =========================================================================

def get_visit_info(caller: str) -> Tuple[int, str]:
    """Get visit info for the caller. Returns (visit_count, last_visit_iso). Does not modify the log."""
    os.makedirs(os.path.dirname(VISITOR_DATA_FILE), exist_ok=True)
    data = _read_data()
    entry = _normalize_entry(data.get(caller, {"count": 0, "last_visit": ""}))
    return entry["count"], entry["last_visit"]


def record_new_conversation(caller: str):
    """Increment visit count and update timestamp. Called on first visit or when >1hr gap detected."""
    now = datetime.now().isoformat()
    os.makedirs(os.path.dirname(VISITOR_DATA_FILE), exist_ok=True)
    with open(VISITOR_DATA_FILE, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            log = json.loads(raw) if raw.strip() else {}
            entry = _normalize_entry(log.get(caller, {"count": 0, "last_visit": ""}))
            entry["count"] = entry["count"] + 1
            entry["last_visit"] = now
            log[caller] = entry
            f.seek(0)
            f.truncate()
            json.dump(log, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    logger.info(f"New conversation for {caller}: visit #{entry['count']}, timestamp {now}")


def is_checkin_complete(caller: str) -> bool:
    """Check if caller has completed the new guest check-in process (register step completed in todo list)."""
    data = _read_data()
    entry = data.get(caller)
    if not entry or not isinstance(entry, dict):
        return False
    todos = entry.get("todos", [])
    return any(t.get("id") == "register" and t.get("status") == "completed" for t in todos)


# =========================================================================
# Visible tools — callable via MCP
# =========================================================================

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
        {"id": "greet",      "content": "Greet the guest warmly and introduce yourself — you're Kitty, the front desk intern.", "status": "pending"},
        {"id": "drink",      "content": "Offer them a warm drink (cocoa, coffee, tea) and ask if it's their first time in Greenland or at Atlantis.", "status": "pending"},
        {"id": "paperwork",  "content": "Ask to see their security paperwork — the signed entry authorization form and their security card. Be friendly but firm — you MUST receive their paperwork before proceeding. No exceptions.", "status": "pending"},
        {"id": "verify",     "content": "Once they hand over their paperwork, call verify_paperwork to read their security card.", "status": "pending"},
        {"id": "register",   "content": "After verification, ask for their real first name and call register_guest with their username and first name to finish check-in.", "status": "pending"},
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
    Final step of new guest check-in. Stores the guest's real first name in the visitor database.

    Args:
        username: The username from their security card
        first_name: The guest's real first name
    """
    logger.info(f"register_guest called for: {username} (first_name={first_name})")

    set_visitor_flag(username, "first_name", first_name)

    return (
        f"Guest {first_name} (username: {username}) has been registered!\n"
        f"Welcome them by name and let them know they're all set."
    )


@visible
async def list_guests():
    """
    Returns a list of all known guest names from visitor data.
    """
    logger.info("list_guests called")
    data = _read_data()
    if not data:
        return "No guests on record yet."
    return json.dumps(list(data.keys()))


@visible
async def guest_info(username: str):
    """
    Look up all stored data for a guest.

    Args:
        username: The guest's name/identifier
    """
    logger.info(f"guest_info called for: {username}")
    entry = _read_data().get(username)
    if not entry:
        return f"No record found for {username}. They may be a brand new guest."
    return json.dumps(entry, indent=2)
