"""Location tools"""

import atlantis
import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.common import _ensure_thumb, home_path

logger = logging.getLogger("mcp_server")


# =========================================================================
# Directory helpers
# =========================================================================

def _locations_dir() -> str:
    return home_path("Game", "Locations")


# =========================================================================
# Location loading
# =========================================================================

def _location_dir(name: str) -> str:
    return os.path.join(_locations_dir(), name)


def _load_location(name: str) -> Optional[Dict[str, Any]]:
    """Load a location by name (folder name is the identifier)"""
    path = os.path.join(_location_dir(name), "config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _connects_to(location_name: str) -> List[str]:
    loc = _load_location(location_name)
    if not loc:
        return []
    return loc.get("connects_to", [])


def _child_locations(location_name: str) -> List[str]:
    """Names of locations whose parent is `location_name`."""
    loc_dir = _locations_dir()
    if not os.path.isdir(loc_dir):
        return []
    children: List[str] = []
    for entry in os.listdir(loc_dir):
        cfg = os.path.join(loc_dir, entry, "config.json")
        if not os.path.isfile(cfg):
            continue
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (data.get("parent") or "") == location_name:
            children.append(entry)
    return children


def _is_leaf(location_name: str) -> bool:
    """A location is a valid move target only if it has no children (containers aren't standable)."""
    return not _child_locations(location_name)


def _require_leaf(location_name: str) -> None:
    if not _is_leaf(location_name):
        children = _child_locations(location_name)
        raise ValueError(
            f"{location_name} is a container, not a place you can stand. "
            f"It contains: {', '.join(children)}."
        )



# =========================================================================
# Thumbnails (location-specific)
# =========================================================================

def location_thumb(loc_name: str) -> str:
    """Get a location thumbnail path"""
    logger.info(f"[thumb] location_thumb called: {loc_name!r}")
    loc = _load_location(loc_name)
    if not loc:
        logger.warning(f"[thumb] location not found: {loc_name!r}")
        return ""
    image_file = loc.get("image", "")
    if not image_file:
        logger.warning(f"[thumb] no image for location: {loc_name!r}")
        return ""
    image_path = os.path.join(_location_dir(loc_name), image_file)
    logger.info(f"[thumb] location {loc_name!r} -> {image_path}")
    if not os.path.isfile(image_path):
        logger.warning(f"[thumb] image file missing: {image_path}")
        return ""
    return _ensure_thumb(image_path)





# =========================================================================
# Visible tools
# =========================================================================

def _location_rows() -> List[Dict[str, str]]:
    """Pure data: list locations. No client side effects."""
    locations_dir = _locations_dir()
    if not os.path.isdir(locations_dir):
        return []
    locations: List[Dict[str, str]] = []
    for entry in sorted(os.listdir(locations_dir)):
        entry_dir = os.path.join(locations_dir, entry)
        if not os.path.isdir(entry_dir) or entry.startswith('.') or entry == '__pycache__':
            continue
        json_path = os.path.join(entry_dir, 'config.json')
        if not os.path.isfile(json_path):
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        name = entry  # folder name is the identifier
        image_data = ''
        image_file = data.get('image', '')
        mtimes = [os.path.getmtime(json_path)]
        if image_file:
            image_path = os.path.join(entry_dir, image_file)
            if os.path.isfile(image_path):
                mtimes.append(os.path.getmtime(image_path))
                thumb = _ensure_thumb(image_path)
                if thumb:
                    ext = os.path.splitext(thumb)[1].lower().lstrip('.')
                    mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext, 'jpeg')
                    with open(thumb, 'rb') as img:
                        b64 = base64.b64encode(img.read()).decode('ascii')
                    image_data = f'data:image/{mime};base64,{b64}'
        locations.append({
            'name': name,
            'displayName': data.get('displayName', name),
            'parent': data.get('parent') or '',
            'connects_to': '\n'.join(data.get('connects_to', []) or []),
            'description': data.get('description', ''),
            'image': image_data,
            'updated': datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M'),
        })
    return locations

@visible
def location_compose_descriptions(location_name: str) -> str:
    """Walk from the root down to `location_name` and concatenate descriptions.

    Returns one paragraph per level (root first), so the prompt reads
    outer-context → inner-context. Empty string if the location has no
    description and no ancestors with one.
    """
    if not _load_location(location_name):
        raise ValueError(f"Unknown location: {location_name}")
    chain: List[str] = []
    seen: set = set()
    current = location_name
    while current and current not in seen:
        seen.add(current)
        loc = _load_location(current)
        if not loc:
            break
        chain.append(current)
        current = loc.get("parent") or ""
    parts: List[str] = []
    for name in reversed(chain):
        loc = _load_location(name) or {}
        desc = (loc.get("description") or "").strip()
        if desc:
            parts.append(desc)
    return "\n\n".join(parts)


@visible
async def location_list() -> List[Dict[str, str]]:
    """List locations"""
    locations = _location_rows()
    await atlantis.client_data("Locations", locations, column_formatter={
        "description": {"type": "markdown", "maxWidth": "80ch"},
        "connects_to": {"type": "pre"},
    })
    return locations



# camera_look / camera_follow are gone — use terminal.terminal_move(game_key, location)
# instead. "Terminal at a location" is the only viewing concept now.

# Facility map moved to location_map.py

