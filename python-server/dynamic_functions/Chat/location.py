"""Location tools"""

import atlantis
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from .common import _image_data_uri, game_path, _require_str
from dynamic_functions.Home.modal import modal_menu

logger = logging.getLogger("dynamic_function")


class LocationConfigT(TypedDict):
    """A location's config.json, normalized into a typed record.

    `displayName` is required and validated at load. `parent` and `image` are
    nullable — a root location has no parent, a container has no image — and
    surface as None rather than "". The rest carry typed defaults.
    """
    name: str
    displayName: str
    parent: Optional[str]
    connects_to: List[str]
    description: str
    default: bool
    defaultCameraAlign: str
    image: Optional[str]


# =========================================================================
# Directory helpers
# =========================================================================

def _locations_dir(game_name: str) -> str:
    return game_path(game_name, "Game", "Locations")


# =========================================================================
# Location loading
# =========================================================================

def _location_dir(game_name: str, name: str) -> str:
    return os.path.join(_locations_dir(game_name), name)


def _load_location(game_name: str, name: str) -> Optional[Dict[str, Any]]:
    """Load a location by name (folder name is the identifier)"""
    path = os.path.join(_location_dir(game_name, name), "config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_location(game_name: str, name: str) -> LocationConfigT:
    """Load a location's config as a typed record.

    The boundary where a loose config.json becomes a known shape. Raises if the
    name has no config (foreign-key check) or if displayName is missing. Nullable
    fields (parent, image) come back as None when absent rather than "".
    """
    raw = _load_location(game_name, name)
    if raw is None:
        raise ValueError(f"Unknown location: {name}")
    label = f"Location {name!r} config.json"
    return LocationConfigT(
        name=name,
        displayName=_require_str(raw, "displayName", label),
        parent=(str(raw.get("parent") or "").strip() or None),
        connects_to=list(raw.get("connects_to") or []),
        description=str(raw.get("description", "")),
        default=bool(raw.get("default", False)),
        defaultCameraAlign=str(raw.get("defaultCameraAlign", "")),
        image=(str(raw.get("image") or "").strip() or None),
    )


def _connects_to(game_name: str, location_name: str) -> List[str]:
    loc = _load_location(game_name, location_name)
    if not loc:
        return []
    return loc.get("connects_to", [])


def _child_locations(game_name: str, location_name: str) -> List[str]:
    """Names of locations whose parent is `location_name`."""
    loc_dir = _locations_dir(game_name)
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


def _is_leaf(game_name: str, location_name: str) -> bool:
    """A location is a valid move target only if it has no children (containers aren't standable)."""
    return not _child_locations(game_name, location_name)


def _require_leaf(game_name: str, location_name: str) -> None:
    if not _is_leaf(game_name, location_name):
        children = _child_locations(game_name, location_name)
        raise ValueError(
            f"{location_name} is a container, not a place you can stand. "
            f"It contains: {', '.join(children)}."
        )


def _location_keys(game_name: str) -> List[str]:
    """Folder names of all locations."""
    loc_dir = _locations_dir(game_name)
    if not os.path.isdir(loc_dir):
        return []
    return sorted(
        entry for entry in os.listdir(loc_dir)
        if os.path.isdir(os.path.join(loc_dir, entry))
        and not entry.startswith(".") and entry != "__pycache__"
    )


def _leaf_location_keys(game_name: str) -> List[str]:
    """Folder names of standable (leaf) locations — the only valid camera/move targets."""
    return [name for name in _location_keys(game_name) if _is_leaf(game_name, name)]



# =========================================================================
# Images
# =========================================================================

def location_image_file(game_name: str, name: str) -> str:
    """Path to a location's image, or "" when its config names none (containers)."""
    loc = _load_location(game_name, name)
    if not loc:
        raise ValueError(f"Unknown location: {name}")
    image_file = loc.get("image", "")
    return os.path.join(_location_dir(game_name, name), image_file) if image_file else ""


def _location_picker_choices(game_name: str, current_location: str = "") -> List[Dict[str, Any]]:
    choices: List[Dict[str, Any]] = []
    current_location = str(current_location or "").strip()
    for location in _leaf_location_keys(game_name):
        label = load_location(game_name, location).get("displayName") or location
        current_marker = "Current" if location == current_location else ""
        image = _image_data_uri(location_image_file(game_name, location))
        choices.append({
            "id": location,
            "text": f"{label}{' (current)' if current_marker else ''}",
            "location": location,
            "columns": [
                {"type": "image", "src": image, "alt": str(label)},
                str(label),
                current_marker,
            ],
            "column_headers": ["", "Location", ""],
        })
    return choices


async def _location_pick_dialog(game_name: str,
    *,
    title: str = "Location",
    heading: str = "Select location",
    current_location: str = "",
) -> Optional[str]:
    choices = _location_picker_choices(game_name, current_location)
    if not choices:
        raise RuntimeError("No standable locations found")
    choice = await modal_menu(
        choices,
        title=title,
        heading=heading,
    )
    if choice is None:
        return None
    location = str(choice.get("location") or choice.get("id") or "").strip()
    return location or None


def location_image_path(game_name: str, name: str) -> str:
    """Absolute path to a location's full background image.

    Unlike `location_image_file`, this raises rather than returning "" — a location
    used as a camera target must have a real, present image, otherwise the
    "terminal background matches its location" invariant cannot hold.
    """
    image_path = location_image_file(game_name, name)
    if not image_path:
        raise ValueError(f"Location {name!r} has no image to use as a background")
    if not os.path.isfile(image_path):
        raise ValueError(f"Location {name!r} image file is missing: {image_path}")
    return image_path





# =========================================================================
# Visible tools
# =========================================================================

def _location_rows(game_name: str) -> List[Dict[str, Any]]:
    """Pure data: list locations. No client side effects."""
    locations_dir = _locations_dir(game_name)
    if not os.path.isdir(locations_dir):
        return []
    locations: List[Dict[str, Any]] = []
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
        image_file = data.get('image', '')
        image_path = os.path.join(entry_dir, image_file) if image_file else ''
        mtimes = [os.path.getmtime(json_path)]
        if image_path and os.path.isfile(image_path):
            mtimes.append(os.path.getmtime(image_path))
        image_data = _image_data_uri(image_path)
        locations.append({
            'name': name,
            'displayName': data.get('displayName', name),
            'image': image_data,
            'description': data.get('description', ''),
            'parent': data.get('parent') or '',
            'connects_to': '\n'.join(data.get('connects_to', []) or []),
            'updated': datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M'),
        })
    # A location is a leaf (standable) iff nothing else claims it as a parent.
    non_leaf = {loc['parent'] for loc in locations if loc['parent']}
    for loc in locations:
        loc['is_leaf'] = loc['name'] not in non_leaf
    return locations

@visible
def location_compose_descriptions(game_name: str, location_name: str) -> str:
    """Walk from the root down to `location_name` and concatenate descriptions.

    Returns one paragraph per level (root first), so the prompt reads
    outer-context → inner-context. Empty string if the location has no
    description and no ancestors with one.
    """
    if not _load_location(game_name, location_name):
        raise ValueError(f"Unknown location: {location_name}")
    chain: List[str] = []
    seen: set = set()
    current = location_name
    while current and current not in seen:
        seen.add(current)
        loc = _load_location(game_name, current)
        if not loc:
            break
        chain.append(current)
        current = loc.get("parent") or ""
    parts: List[str] = []
    for name in reversed(chain):
        loc = _load_location(game_name, name) or {}
        desc = (loc.get("description") or "").strip()
        if desc:
            parts.append(desc)
    return "\n\n".join(parts)


@public
async def location_list(game_name: str) -> List[Dict[str, str]]:
    """List locations"""
    locations = _location_rows(game_name)
    await atlantis.client_data("Locations", locations, column_formatter={
        "description": {"type": "markdown", "maxWidth": "60ch"},
        "connects_to": {"type": "pre"},
    })
    return locations


@visible
async def location_pick(game_name: str) -> Optional[str]:
    """Pick a standable location using the standard location picker dialog."""
    return await _location_pick_dialog(game_name)



# camera_look / camera_follow are gone — use term.term_move(game_key, location)
# instead. "Terminal at a location" is the only viewing concept now.

# Facility map moved to location_map.py
