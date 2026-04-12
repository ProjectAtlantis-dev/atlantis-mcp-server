"""Roster — persistent role definitions with bot assignments.

roster.json is an array of role records. Each has id, title, location,
bot, requiresCheckin, etc. Game callbacks read these to set up the world.
Editable via exposed tools.
"""

import json
import os
import logging
from typing import Optional, List

logger = logging.getLogger("mcp_server")

_ROSTER_FILE = os.path.join(os.path.dirname(__file__), "roster.json")


def _load_roster() -> List[dict]:
    if not os.path.exists(_ROSTER_FILE):
        return []
    with open(_ROSTER_FILE) as f:
        return json.load(f)


def _save_roster(roster: List[dict]) -> None:
    with open(_ROSTER_FILE, "w") as f:
        json.dump(roster, f, indent=2)


def get_role(role_id: str) -> dict:
    """Get a role record by id. Raises if not found."""
    for role in _load_roster():
        if role["id"] == role_id:
            return role
    raise ValueError(f"Unknown role: '{role_id}'")


def get_bot_for_role(role_id: str) -> str:
    """Look up which bot is assigned to a role. Raises if unassigned."""
    role = get_role(role_id)
    bot = role.get("bot")
    if not bot:
        raise ValueError(f"No bot assigned to role '{role_id}'")
    return bot


@visible
async def list_roles() -> List[dict]:
    """List all roles in the roster."""
    return _load_roster()


@visible
async def assign_bot(role_id: str, bot_sid: str) -> dict:
    """Assign a bot to a role. Takes effect on the next game start.

    Args:
        role_id: Role identifier (e.g. "flowcentral_receptionist")
        bot_sid: Bot sid to assign (e.g. "atlas", "kitty")
    """
    roster = _load_roster()
    for role in roster:
        if role["id"] == role_id:
            role["bot"] = bot_sid
            _save_roster(roster)
            logger.info(f"Roster updated: {role_id} -> {bot_sid}")
            return role
    raise ValueError(f"Unknown role: '{role_id}'")
