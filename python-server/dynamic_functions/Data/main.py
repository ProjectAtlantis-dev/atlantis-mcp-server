"""Game-scoped data helpers.

All stateful data lives under Data/{game_id}/.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mcp_server")

DATA_DIR = os.path.dirname(__file__)


# =========================================================================
# Visible tools
# =========================================================================

@visible
async def index():
    """Game data management — all state organized by game_id."""
    return list_games()


# =========================================================================
# Low-level I/O
# =========================================================================

def _safe_id(value: str, label: str = "id") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "").strip())
    if not safe:
        raise ValueError(f"Cannot use an empty {label}")
    return safe


def game_dir(game_id: str, *, create: bool = False) -> str:
    """Return the data directory for a game."""
    path = os.path.join(DATA_DIR, _safe_id(game_id, "game_id"))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
            return json.loads(raw) if raw.strip() else default
    except FileNotFoundError:
        return default


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# =========================================================================
# Positions — {sid: location_name}
# =========================================================================

def _positions_path(game_id: str) -> str:
    return os.path.join(game_dir(game_id, create=True), "positions.json")


def get_positions(game_id: str) -> Dict[str, str]:
    """Return the full sid -> location dict for a game."""
    return _read_json(_positions_path(game_id), {}) or {}


def set_positions(game_id: str, positions: Dict[str, str]) -> None:
    """Persist the full positions dict for a game."""
    _write_json(_positions_path(game_id), positions)


def get_player_position(game_id: str, sid: str) -> Optional[str]:
    """Return a player's current location in a game, or None."""
    return get_positions(game_id).get(sid)


def set_player_position(game_id: str, sid: str, location: str) -> Dict[str, str]:
    """Set a player's location and persist. Returns updated positions."""
    positions = get_positions(game_id)
    positions[sid] = location
    set_positions(game_id, positions)
    return positions



def read_location_data(game_id: str, location: str) -> Optional[Dict[str, Any]]:
    """Read Data/{game_id}/{location}.json, or None if it doesn't exist."""
    path = os.path.join(game_dir(game_id), f"{location}.json")
    return _read_json(path, None)


def write_location_data(game_id: str, location: str, data: Dict[str, Any]) -> None:
    """Write Data/{game_id}/{location}.json."""
    path = os.path.join(game_dir(game_id, create=True), f"{location}.json")
    _write_json(path, data)


def get_players_at(game_id: str, location: str) -> list[str]:
    """Return list of sids at a location in a game."""
    return [s for s, loc in get_positions(game_id).items() if loc == location]


# =========================================================================
# Player profiles — Data/{game_id}/profiles.json  {sid: {...}}
# =========================================================================

def _profiles_path(game_id: str) -> str:
    return os.path.join(game_dir(game_id, create=True), "profiles.json")


def get_profiles(game_id: str) -> Dict[str, Dict[str, Any]]:
    """Return all player profiles for a game."""
    return _read_json(_profiles_path(game_id), {}) or {}


def get_user_profile(game_id: str, sid: str) -> Optional[Dict[str, Any]]:
    """Return a player's profile in a game, or None."""
    return get_profiles(game_id).get(sid)


def set_user_profile(game_id: str, sid: str, profile: Dict[str, Any]) -> None:
    """Set a player's profile and persist."""
    profiles = get_profiles(game_id)
    profiles[sid] = profile
    _write_json(_profiles_path(game_id), profiles)


def update_user_profile(game_id: str, sid: str, **fields) -> Dict[str, Any]:
    """Merge fields into a player's profile. Creates if missing."""
    profile = get_user_profile(game_id, sid) or {}
    profile.update(fields)
    set_user_profile(game_id, sid, profile)
    return profile


# =========================================================================
# Interactions — Data/{game_id}/interactions.json
#   {sid: {bot_sid: [{started_at, last_seen_at, message_count, ...}]}}
# =========================================================================

def _interactions_path(game_id: str) -> str:
    return os.path.join(game_dir(game_id, create=True), "interactions.json")


def _read_interactions(game_id: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    return _read_json(_interactions_path(game_id), {}) or {}


def _write_interactions(game_id: str, data: Dict) -> None:
    _write_json(_interactions_path(game_id), data)


def get_bot_interactions(game_id: str, sid: str, bot_sid: str) -> List[Dict[str, Any]]:
    """Return all recorded interactions between a player and a bot in a game."""
    all_data = _read_interactions(game_id)
    return all_data.get(sid, {}).get(bot_sid, [])


def get_bot_interaction_info(
    game_id: str,
    sid: str,
    bot_sid: str,
    *,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return prior interaction context for a player/bot pair in a game.

    The current session is excluded so one long conversation does not
    register as 'we have met before' on the second message.
    """
    records = get_bot_interactions(game_id, sid, bot_sid)
    prior = [
        r for r in records
        if not (session_id and r.get("session_id") == session_id)
    ]
    last_prior = prior[-1] if prior else None
    last_ts = str(last_prior.get("last_seen_at") or last_prior.get("started_at") or "") if last_prior else ""

    return {
        "bot_sid": bot_sid,
        "has_met_before": bool(prior),
        "prior_interaction_count": len(prior),
        "last_interaction_at": last_ts,
        "total_recorded_interactions": len(records),
    }


def record_bot_interaction(
    game_id: str,
    sid: str,
    bot_sid: str,
    *,
    location: str = "",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record or update an interaction for a player/bot pair in a game."""
    now = datetime.now().isoformat()
    all_data = _read_interactions(game_id)
    player_data = all_data.setdefault(sid, {})
    records = player_data.setdefault(bot_sid, [])

    # Find existing record for this session
    current = None
    if session_id:
        for r in records:
            if r.get("session_id") == session_id:
                current = r
                break

    if current is None:
        current = {
            "bot_sid": bot_sid,
            "started_at": now,
            "last_seen_at": now,
            "location": location,
            "message_count": 0,
        }
        if session_id:
            current["session_id"] = session_id
        records.append(current)
        logger.info(f"Recorded first interaction: game={game_id} sid={sid} bot={bot_sid}")
    else:
        current["last_seen_at"] = now
        if location and not current.get("location"):
            current["location"] = location
        logger.info(f"Updated interaction: game={game_id} sid={sid} bot={bot_sid}")

    current["message_count"] = int(current.get("message_count") or 0) + 1
    _write_interactions(game_id, all_data)
    return current


# =========================================================================
# Game listing
# =========================================================================

def list_games() -> list[str]:
    """List all game_ids that have data."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and not d.startswith(".")
        and d != "__pycache__"
        and d != "players"  # ignore legacy
    )
