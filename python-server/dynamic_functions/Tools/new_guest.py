import atlantis
import logging
import json
import os
import fcntl

logger = logging.getLogger("mcp_server")

VISITOR_DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'Data', 'visitor_data.json')


@visible
async def new_guest():
    """
    Call this when a new guest arrives who has no record in the visitor database.
    Returns the front-desk procedure Kitty must follow. Do NOT ask for the guest's
    name or username yet — that comes later in the procedure.
    """
    logger.info("new_guest procedure initiated")

    return (
        "New guest procedure — follow these steps IN ORDER:\n"
        "1. Greet the guest warmly and introduce yourself — you're Kitty, the front desk intern.\n"
        "2. Offer them a warm drink (cocoa, coffee, tea) and ask if it's their first time in Greenland or at Atlantis.\n"
        "3. Ask to see their security paperwork — the signed entry authorization form and their security card.\n"
        "4. Be friendly but firm — you MUST receive their paperwork before proceeding. No exceptions.\n"
        "5. Once they hand over their paperwork, call verify_paperwork with the username printed on their security card.\n"
        "6. After verification, ask for their real first name and call register_guest to finish check-in."
    )


@visible
async def verify_paperwork(username: str):
    """
    Call this after the guest hands over their security card and entry authorization.
    Reveals the username on their card so you can react to it.

    Args:
        username: The username printed on the guest's security card
    """
    logger.info(f"verify_paperwork called for: {username}")

    _set_visitor_flag(username, "greeted", True)

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

    _set_visitor_flag(username, "first_name", first_name)

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
