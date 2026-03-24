import atlantis
import logging
import json
import os
import fcntl
from datetime import datetime

logger = logging.getLogger("mcp_server")

VISITOR_DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'Bot', 'Kitty', 'visitor_data.json')


@visible
async def new_guest(username: str):
    """
    Procedures for handling a new guest who hasn't been processed yet.

    Args:
        username: The guest's name/identifier
    """
    logger.info(f"new_guest tool called for: {username}")

    # Mark that we've started the new guest procedure
    _set_visitor_flag(username, "greeted", True)

    return (
        f"New guest procedure for {username}:\n"
        f"1. Greet {username} warmly and introduce yourself — you're Kitty, the front desk intern.\n"
        f"2. Offer them a warm drink and ask them the usual - first time to Greenland or Atlantis.\n"
        f"3. Ask to see their paperwork. Everyone needs to have their documents checked so they can get their room key.\n"
        f"3. Be friendly but firm — security paperwork is required, no exceptions.\n"
        f"4. Once they provide their paperwork, call the `security_cleared` tool with their name to mark them as cleared."
    )


@visible
async def security_cleared(username: str):
    """
    Mark a guest's security paperwork as reviewed and approved.
    Call this after you have checked the guest's security documents.

    Args:
        username: The guest's name/identifier
    """
    logger.info(f"security_cleared called for: {username}")

    _set_visitor_flag(username, "security_cleared", True)
    _set_visitor_flag(username, "security_cleared_at", datetime.now().isoformat())

    return f"{username}'s security paperwork has been verified and recorded. They are cleared to proceed into the facility."


@visible
async def list_guests():
    """
    Returns a list of all known guest names from visitor data.
    """
    logger.info("list_guests called")

    try:
        with open(VISITOR_DATA_FILE, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                raw = f.read()
                data = json.loads(raw) if raw.strip() else {}
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except FileNotFoundError:
        return "No visitor data file found."

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

    try:
        with open(VISITOR_DATA_FILE, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                raw = f.read()
                data = json.loads(raw) if raw.strip() else {}
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except FileNotFoundError:
        return f"No visitor data found for {username}."

    entry = data.get(username)
    if not entry:
        return f"No record found for {username}. They may be a brand new guest."

    return json.dumps(entry, indent=2)


def _set_visitor_flag(username: str, flag: str, value):
    """Set a flag on a visitor's data record."""
    os.makedirs(os.path.dirname(VISITOR_DATA_FILE), exist_ok=True)
    with open(VISITOR_DATA_FILE, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            data = json.loads(raw) if raw.strip() else {}
            entry = data.get(username, {"count": 0, "last_visit": ""})
            if isinstance(entry, int):
                entry = {"count": entry, "last_visit": ""}
            entry[flag] = value
            data[username] = entry
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    logger.info(f"Set {flag}={value} for visitor {username}")
