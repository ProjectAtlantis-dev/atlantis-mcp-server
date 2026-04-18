"""Static role definitions.

Roles describe what a job is: location, title, procedures, and tool access.
Rosters assign bots to these roles for a specific user's game.
"""

import json
import logging
import os
from typing import List

logger = logging.getLogger("mcp_server")

_ROLES_FILE = os.path.join(os.path.dirname(__file__), "roles.json")


def _load_roles() -> List[dict]:
    if not os.path.exists(_ROLES_FILE):
        return []
    with open(_ROLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Invalid role data in {_ROLES_FILE}: expected a list")
    return data


def get_role(role_id: str) -> dict:
    """Get a static role definition by id. Raises if not found."""
    for role in _load_roles():
        if role["id"] == role_id:
            return role
    raise ValueError(f"Unknown role: '{role_id}'")


@visible
async def roles_list() -> List[dict]:
    """List all static role definitions."""
    return _load_roles()
