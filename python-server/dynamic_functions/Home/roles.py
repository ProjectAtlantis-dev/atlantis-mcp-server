"""Static role definitions.

Roles describe what a job is: location, title, procedures, and tool access.
Scans all Games/*/roles.json files to build a unified role registry.
"""

import json
import logging
import os
from typing import List

logger = logging.getLogger("mcp_server")

_GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Games")


def _load_roles() -> List[dict]:
    roles = []
    if not os.path.isdir(_GAMES_DIR):
        return roles
    for entry in os.listdir(_GAMES_DIR):
        roles_file = os.path.join(_GAMES_DIR, entry, "roles.json")
        if os.path.isfile(roles_file):
            with open(roles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                roles.extend(data)
            else:
                logger.warning(f"Invalid role data in {roles_file}: expected a list")
    return roles


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
