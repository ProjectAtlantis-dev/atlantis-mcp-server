import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger("mcp_server")

PLAYERS_DIR = os.path.join(os.path.dirname(__file__), "players")



# =========================================================================
# Visible tools
# =========================================================================

@visible
async def index():
    """Low-level player data mgmt"""
    return list_player_names()


# =========================================================================
# Low-level per-user file I/O
# =========================================================================

def _player_path(username: str) -> str:
    return os.path.join(PLAYERS_DIR, f"{username}.json")


def read_player(username: str) -> dict[str, Any]:
    """Read a player's JSON file. Returns empty dict if missing."""
    if not username:
        return {}
    try:
        with open(_player_path(username), "r") as f:
            raw = f.read()
            return json.loads(raw) if raw.strip() else {}
    except FileNotFoundError:
        return {}


def write_player(username: str, data: dict[str, Any]) -> None:
    """Write a player's JSON file."""
    if not username:
        raise ValueError("Cannot write player data without a username")
    os.makedirs(PLAYERS_DIR, exist_ok=True)
    with open(_player_path(username), "w") as f:
        json.dump(data, f, indent=2)


def list_player_names() -> list[str]:
    """List all player record names."""
    os.makedirs(PLAYERS_DIR, exist_ok=True)
    return [
        filename.replace(".json", "")
        for filename in os.listdir(PLAYERS_DIR)
        if filename.endswith(".json")
    ]


def get_player_field(username: str, key: str, default=None):
    """Read a single field from a player's record."""
    return read_player(username).get(key, default)


def set_player_field(username: str, key: str, value) -> dict[str, Any]:
    """Set a single field on a player's record and persist."""
    data = read_player(username)
    data[key] = value
    write_player(username, data)
    return data


# =========================================================================
# Player record helpers (used by game/chat callbacks)
# =========================================================================

def get_player_record(username: str) -> dict[str, Any] | None:
    """Return the player record, or None if no file exists."""
    data = read_player(username)
    return data if data else None


def ensure_player_record(username: str) -> tuple[dict[str, Any], bool]:
    """Ensure a player has a record file. Returns (record, created_new)."""
    if not username:
        raise ValueError("Cannot create a player record without a username")

    data = read_player(username)
    created_new = len(data) == 0

    if created_new:
        data["location"] = "AtlasLobby"
        write_player(username, data)
        logger.info(f"Created new player record for {username}")

    return data, created_new


def set_player_location(username: str, location: str) -> dict[str, Any]:
    """Persist a player's current location."""
    if not username:
        raise ValueError("Cannot set location without a username")
    if not location:
        raise ValueError("Cannot set an empty location")
    return set_player_field(username, "location", location)


# =========================================================================
# Guest / check-in helpers
# =========================================================================

def get_guest(username: str) -> dict[str, Any] | None:
    """Return guest-relevant fields from the player record, or None if not found."""
    data = read_player(username)
    if not data:
        return None
    return data


def is_cleared(username: str) -> bool:
    """Has this player completed check-in?"""
    data = read_player(username)
    return bool(data and data.get("cleared"))


def get_visit_info(username: str) -> tuple[int, str]:
    """Return (visit_count, last_visit) for a player."""
    data = read_player(username)
    if not data:
        return 0, ""
    return int(data.get("visit_count") or 0), data.get("last_visit") or ""


def record_new_conversation(username: str, location: str = "AtlasLobby") -> None:
    """Bump visit count and timestamp for a player."""
    if not username:
        return
    data = read_player(username)
    now = datetime.now().isoformat()
    data["visit_count"] = int(data.get("visit_count") or 0) + 1
    data["last_visit"] = now
    if not data.get("location"):
        data["location"] = location
    write_player(username, data)
    logger.info(f"New conversation recorded for {username}")


def register_guest(username: str, first_name: str, location: str = "AtlasLobby") -> dict[str, Any]:
    """Register a guest (final check-in step). Sets cleared=True."""
    data = read_player(username)
    now = datetime.now().isoformat()
    data["first_name"] = first_name
    data["visit_count"] = int(data.get("visit_count") or 0) + 1
    data["last_visit"] = now
    data["cleared"] = True
    data["location"] = location
    write_player(username, data)
    logger.info(f"Registered guest {first_name} ({username}) at {location}")
    return data


def list_all_guests() -> list[dict[str, Any]]:
    """Return summary info for all players who have guest data."""
    result = []
    for name in list_player_names():
        data = read_player(name)
        if data:
            result.append({
                "username": name,
                "first_name": data.get("first_name", ""),
                "visit_count": int(data.get("visit_count") or 0),
                "cleared": bool(data.get("cleared")),
            })
    return result

