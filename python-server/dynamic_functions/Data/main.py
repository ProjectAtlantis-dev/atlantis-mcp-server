import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("mcp_server")

PLAYERS_DIR = os.path.join(os.path.dirname(__file__), "players")
_RESERVED_PLAYER_FILENAMES = {"games"}
_INTERACTIONS_FIELD = "interactions"



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

def _safe_data_id(value: str, label: str = "id") -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "").strip())
    if not safe_id:
        raise ValueError(f"Cannot use an empty {label}")
    return safe_id


def _player_path(username: str) -> str:
    """Legacy flat player path. Read-only compatibility for old data."""
    return os.path.join(PLAYERS_DIR, f"{username}.json")


def player_data_dir(username: str, *, create: bool = False) -> str:
    """Return a player's private data directory."""
    path = os.path.join(PLAYERS_DIR, _safe_data_id(username, "username"))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def player_folder_exists(username: str) -> bool:
    """Return whether this player already has a private data folder."""
    if not username:
        return False
    return os.path.isdir(player_data_dir(username, create=False))


def player_data_exists(username: str) -> bool:
    """Return whether this player has folder data or legacy flat data."""
    if not username:
        return False
    return player_folder_exists(username) or os.path.exists(_player_path(username))


def player_game_dir(username: str, game_id: str, *, create: bool = False) -> str:
    """Return a player's private data directory for a single game."""
    path = os.path.join(
        player_data_dir(username, create=create),
        "games",
        _safe_data_id(game_id, "game_id"),
    )
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _field_path(username: str, key: str, *, create_dir: bool = False) -> str:
    safe_key = _safe_data_id(key, "field")
    if safe_key in _RESERVED_PLAYER_FILENAMES:
        raise ValueError(f"'{safe_key}' is a reserved player data field")
    return os.path.join(player_data_dir(username, create=create_dir), f"{safe_key}.json")


def _read_json_file(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
            return json.loads(raw) if raw.strip() else default
    except FileNotFoundError:
        return default


def _write_json_file(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def read_player(username: str) -> dict[str, Any]:
    """Read all top-level player fields from the player's folder.

    Legacy flat player JSON is merged first for backwards compatibility.
    New writes go to one file per field under players/{username}/.
    """
    if not username:
        return {}
    data = _read_json_file(_player_path(username), {})

    player_dir = player_data_dir(username, create=False)
    if not os.path.isdir(player_dir):
        return data

    for filename in os.listdir(player_dir):
        if not filename.endswith(".json"):
            continue
        key = filename[:-5]
        if key in _RESERVED_PLAYER_FILENAMES:
            continue
        data[key] = _read_json_file(os.path.join(player_dir, filename), None)

    return data


def write_player(username: str, data: dict[str, Any]) -> None:
    """Write player fields as separate files under players/{username}/."""
    if not username:
        raise ValueError("Cannot write player data without a username")
    player_data_dir(username, create=True)
    for key, value in data.items():
        set_player_field(username, key, value)


def list_player_names() -> list[str]:
    """List all player record names from folders and legacy flat JSON."""
    os.makedirs(PLAYERS_DIR, exist_ok=True)
    names = set()
    for filename in os.listdir(PLAYERS_DIR):
        path = os.path.join(PLAYERS_DIR, filename)
        if os.path.isdir(path):
            names.add(filename)
        elif filename.endswith(".json"):
            names.add(filename[:-5])
    return sorted(names)


def get_player_field(username: str, key: str, default=None):
    """Read a single field from a player's record."""
    if not username:
        return default
    field_path = _field_path(username, key)
    if os.path.exists(field_path):
        return _read_json_file(field_path, default)
    return _read_json_file(_player_path(username), {}).get(key, default)


def set_player_field(username: str, key: str, value) -> dict[str, Any]:
    """Set a single field on a player's record and persist."""
    if not username:
        raise ValueError("Cannot write player data without a username")
    _write_json_file(_field_path(username, key, create_dir=True), value)
    return read_player(username)


# =========================================================================
# Neutral bot interaction history
# =========================================================================

def _normalize_interactions(raw) -> dict[str, list[dict[str, Any]]]:
    """Return bot-keyed interaction history, ignoring malformed records."""
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for bot_sid, records in raw.items():
        if not bot_sid or not isinstance(records, list):
            continue
        normalized[str(bot_sid)] = [r for r in records if isinstance(r, dict)]
    return normalized


def _interaction_timestamp(record: dict[str, Any]) -> str:
    return str(record.get("last_seen_at") or record.get("started_at") or "")


def _same_interaction(
    record: dict[str, Any],
    *,
    game_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    if game_id and record.get("game_id") == game_id:
        return True
    if session_id and record.get("session_id") == session_id:
        return True
    return False


def get_bot_interactions(username: str, bot_sid: str) -> list[dict[str, Any]]:
    """Return all recorded interactions between a user and a specific bot."""
    if not username or not bot_sid:
        return []
    interactions = _normalize_interactions(get_player_field(username, _INTERACTIONS_FIELD, {}))
    return list(interactions.get(str(bot_sid), []))


def get_bot_interaction_info(
    username: str,
    bot_sid: str,
    *,
    game_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Return prior interaction context for a specific user/bot pair.

    The current game/session is excluded so one long conversation does not turn
    into "we have met before" on the second message.
    """
    records = get_bot_interactions(username, bot_sid)
    prior_records = [
        record for record in records
        if not _same_interaction(record, game_id=game_id, session_id=session_id)
    ]
    last_prior = prior_records[-1] if prior_records else None

    return {
        "bot_sid": str(bot_sid or ""),
        "has_met_before": bool(prior_records),
        "prior_interaction_count": len(prior_records),
        "last_interaction_at": _interaction_timestamp(last_prior) if last_prior else "",
        "total_recorded_interactions": len(records),
    }


def record_bot_interaction(
    username: str,
    bot_sid: str,
    *,
    location: str = "",
    game_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Record or update the current interaction for a specific user/bot pair."""
    if not username or not bot_sid:
        return {}

    now = datetime.now().isoformat()
    interactions = _normalize_interactions(get_player_field(username, _INTERACTIONS_FIELD, {}))
    bot_key = str(bot_sid)
    records = interactions.setdefault(bot_key, [])

    current = None
    for record in records:
        if _same_interaction(record, game_id=game_id, session_id=session_id):
            current = record
            break

    if current is None:
        current = {
            "bot_sid": bot_key,
            "started_at": now,
            "last_seen_at": now,
            "location": location,
        }
        if game_id:
            current["game_id"] = game_id
        if session_id:
            current["session_id"] = session_id
        records.append(current)
        logger.info(f"Recorded first interaction for user={username} bot={bot_key}")
    else:
        current["last_seen_at"] = now
        if location and not current.get("location"):
            current["location"] = location
        logger.info(f"Updated interaction for user={username} bot={bot_key}")

    current["message_count"] = int(current.get("message_count") or 0) + 1
    set_player_field(username, _INTERACTIONS_FIELD, interactions)
    return current


# =========================================================================
# Player record helpers (used by game/chat callbacks)
# =========================================================================

def get_player_record(username: str) -> dict[str, Any] | None:
    """Return the player record, or None if no file exists."""
    data = read_player(username)
    return data if data else None


def ensure_player_record(username: str, location: str = "FlowCentralLobby") -> tuple[dict[str, Any], bool]:
    """Ensure a player has a data folder. Returns (record, created_new)."""
    if not username:
        raise ValueError("Cannot create a player record without a username")

    created_new = not player_data_exists(username)
    data = read_player(username)

    if created_new:
        now = datetime.now().isoformat()
        set_player_field(username, "location", location)
        set_player_field(username, "visits", [now])
        data = read_player(username)
        logger.info(f"Created new player folder for {username}")

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

def get_user_profile(username: str) -> dict[str, Any] | None:
    """Return stored user profile fields, or None if not found."""
    data = read_player(username)
    if not data:
        return None
    return data


def get_guest(username: str) -> dict[str, Any] | None:
    """Scenario-specific alias for check-in bots."""
    return get_user_profile(username)


def is_cleared(username: str) -> bool:
    """Has this player completed check-in?"""
    data = read_player(username)
    return bool(data and data.get("cleared"))


def _get_visits(data: dict) -> list[str]:
    """Return the visits log from player data."""
    visits = data.get("visits")
    if isinstance(visits, list):
        return visits
    return []


def get_visit_info(username: str) -> tuple[int, str]:
    """Return (visit_count, last_visit) for a player."""
    data = read_player(username)
    if not data:
        return 0, ""
    visits = _get_visits(data)
    return len(visits), (visits[-1] if visits else "")


def record_new_conversation(username: str, location: str = "FlowCentralLobby") -> None:
    """Append a timestamped visit to the player's visit log."""
    if not username:
        return
    data = read_player(username)
    now = datetime.now().isoformat()
    visits = _get_visits(data)
    visits.append(now)
    set_player_field(username, "visits", visits)
    if not data.get("location"):
        set_player_field(username, "location", location)
    logger.info(f"New conversation recorded for {username}")


def register_guest(username: str, first_name: str, location: str = "FlowCentralLobby") -> dict[str, Any]:
    """Register a guest (final check-in step). Sets cleared=True."""
    data = read_player(username)
    now = datetime.now().isoformat()
    visits = _get_visits(data)
    visits.append(now)
    data["first_name"] = first_name
    data["visits"] = visits
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
            visits = _get_visits(data)
            result.append({
                "username": name,
                "first_name": data.get("first_name", ""),
                "visit_count": len(visits),
                "cleared": bool(data.get("cleared")),
            })
    return result
