"""Role tools"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from dynamic_functions.Home.common import home_path

logger = logging.getLogger("mcp_server")


def _roles_dir() -> str:
    return home_path("Game", "Roles")


def _load_role_json(role_name: str) -> dict:
    """Read a role config"""
    rjson = os.path.join(_roles_dir(), role_name, "config.json")
    if os.path.isfile(rjson):
        with open(rjson) as f:
            return json.load(f)
    return {}


@visible
async def role_list() -> List[Dict[str, str]]:
    """List available roles"""
    roles_dir = _roles_dir()
    roles: List[Dict[str, str]] = []
    if not os.path.isdir(roles_dir):
        return roles
    for entry in sorted(os.listdir(roles_dir)):
        entry_dir = os.path.join(roles_dir, entry)
        if not os.path.isdir(entry_dir) or entry.startswith(".") or entry == "__pycache__":
            continue
        role_data = _load_role_json(entry)
        mtimes = [
            os.path.getmtime(os.path.join(entry_dir, filename))
            for filename in os.listdir(entry_dir)
            if os.path.isfile(os.path.join(entry_dir, filename))
        ]
        updated = datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M') if mtimes else ''
        roles.append({
            "name": entry,
            "displayName": role_data.get("displayName", entry),
            "greeting": role_data.get("greeting", ""),
            "defaultLocation": role_data.get("defaultLocation", ""),
            "updated": updated,
        })
    return roles


def role_default_location(role: str) -> Optional[str]:
    """Return a role-specific default location from config.json, if set."""
    location = _load_role_json(role).get("defaultLocation", "")
    return str(location).strip() or None


def _validate_role(role: str) -> None:
    """Validate a role folder"""
    if not os.path.isdir(os.path.join(_roles_dir(), role)):
        raise ValueError(f"Role folder not found: {role}")
