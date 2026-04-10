import json
import logging
import os
import fcntl
from typing import Any

logger = logging.getLogger("mcp_server")

PLAYER_DATA_FILE = os.path.join(os.path.dirname(__file__), "player_data.json")
DEFAULT_LOCATION = "Lobby"


def _read_data() -> dict[str, Any]:
    """Read the game player-state file. Returns empty dict if missing."""
    try:
        with open(PLAYER_DATA_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                raw = f.read()
                return json.loads(raw) if raw.strip() else {}
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except FileNotFoundError:
        return {}


def _normalize_entry(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        entry = {}

    where = entry.get("where") or DEFAULT_LOCATION
    return {
        **entry,
        "where": where,
    }


def get_player_record(username: str) -> dict[str, Any] | None:
    """Return the persisted game-state record for a player, if any."""
    if not username:
        return None

    data = _read_data()
    entry = data.get(username)
    if entry is None:
        return None
    return _normalize_entry(entry)


def ensure_player_record(username: str) -> tuple[dict[str, Any], bool]:
    """Ensure a player has a game-state record. Returns (record, created_new)."""
    if not username:
        raise ValueError("Cannot create a game-state record without a username")

    os.makedirs(os.path.dirname(PLAYER_DATA_FILE), exist_ok=True)
    with open(PLAYER_DATA_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            data = json.loads(raw) if raw.strip() else {}
            created_new = username not in data
            record = _normalize_entry(data.get(username, {}))
            data[username] = record
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    if created_new:
        logger.info(f"Created new game-state record for player {username}: where={record['where']}")
    return record, created_new


def set_player_location(username: str, where: str) -> dict[str, Any]:
    """Persist a player's current location and return the updated record."""
    if not username:
        raise ValueError("Cannot set game-state location without a username")
    if not where:
        raise ValueError("Cannot set an empty game-state location")

    os.makedirs(os.path.dirname(PLAYER_DATA_FILE), exist_ok=True)
    with open(PLAYER_DATA_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            data = json.loads(raw) if raw.strip() else {}
            record = _normalize_entry(data.get(username, {}))
            record["where"] = where
            data[username] = record
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    logger.info(f"Set player location for {username}: where={where}")
    return record


@visible
async def index():
    """
    Return the raw game player-state database.
    """
    return _read_data()
