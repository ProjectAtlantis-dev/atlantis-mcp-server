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


@visible
def _load_role_json(roles_dir: str, role_name: str) -> dict:
    """Read a role config"""
    rjson = os.path.join(roles_dir, role_name, "role.json")
    if os.path.isfile(rjson):
        with open(rjson) as f:
            return json.load(f)
    return {}


def _collect_roles(roles_dir: str) -> List[Dict[str, str]]:
    """List roles from a directory"""
    roles = []
    if not os.path.isdir(roles_dir):
        return roles
    for entry in sorted(os.listdir(roles_dir)):
        entry_dir = os.path.join(roles_dir, entry)
        if not os.path.isdir(entry_dir):
            continue
        if entry.startswith(".") or entry == "__pycache__":
            continue
        role_data = _load_role_json(roles_dir, entry)
        mtimes = [
            os.path.getmtime(os.path.join(entry_dir, filename))
            for filename in os.listdir(entry_dir)
            if os.path.isfile(os.path.join(entry_dir, filename))
        ]
        updated = datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M') if mtimes else ''
        roles.append({
            "name": entry,
            "title": role_data.get("title", entry),
            "greeting": role_data.get("greeting", ""),
            "defaultLocation": role_data.get("defaultLocation", ""),
            "updated": updated,
        })
    return roles


@visible
async def role_list() -> List[Dict[str, str]]:
    """List available roles"""
    return _collect_roles(_roles_dir())


def role_default_location(role: str) -> Optional[str]:
    """Return a role-specific default location from role.json, if set."""
    role_data = _load_role_json(_roles_dir(), role)
    location = role_data.get("defaultLocation", "")
    return str(location).strip() or None


def _validate_role(role: str) -> None:
    """Validate a role folder"""
    roles_dir = _roles_dir()
    if not os.path.isdir(os.path.join(roles_dir, role)):
        raise ValueError(f"Role folder not found: {role}")
