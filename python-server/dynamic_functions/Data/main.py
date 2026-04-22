"""Game-scoped data helpers.

All stateful data lives under Data/{game_id}/.
"""

import json
import logging
import os
import re
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





def read_location_data(game_id: str, location: str) -> Optional[Dict[str, Any]]:
    """Read Data/{game_id}/{location}.json, or None if it doesn't exist."""
    path = os.path.join(game_dir(game_id), f"{location}.json")
    return _read_json(path, None)


def write_location_data(game_id: str, location: str, data: Dict[str, Any]) -> None:
    """Write Data/{game_id}/{location}.json."""
    path = os.path.join(game_dir(game_id, create=True), f"{location}.json")
    _write_json(path, data)





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
