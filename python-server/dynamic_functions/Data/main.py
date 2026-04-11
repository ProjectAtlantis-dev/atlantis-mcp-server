import json
import logging
import os
from typing import Any

logger = logging.getLogger("mcp_server")

PLAYERS_DIR = os.path.join(os.path.dirname(__file__), "players")


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
        data["where"] = "Lobby"
        write_player(username, data)
        logger.info(f"Created new player record for {username}")

    return data, created_new


def set_player_location(username: str, where: str) -> dict[str, Any]:
    """Persist a player's current location."""
    if not username:
        raise ValueError("Cannot set location without a username")
    if not where:
        raise ValueError("Cannot set an empty location")
    return set_player_field(username, "where", where)


# =========================================================================
# Visible tools
# =========================================================================

@visible
async def index():
    """List all player files."""
    return list_player_names()
