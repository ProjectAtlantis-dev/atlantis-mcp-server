"""Scene tools

A scene is a named roster of slots with a short description. Its identity is
the filename: <game_name>/Game/Scenes/<name>.json, the same way a bot's identity is its
folder name. Each slot declares its defaultLocation — where whoever fills the
slot enters the world.

Every AI slot's bot needs a role in <game_name>/Game/Roles/<sid>.md: its job
in this game and the people it knows here. A role may name any bot in the
scene's cast, and nobody else.
"""

import atlantis
import logging
import os
import re
from typing import Any, Dict, List, Optional, cast

from .bot import prompt_bot_refs
from .common import game_path, _read_json, _require_str
from .location import _require_leaf, load_location
from dynamic_functions.Home.modal import modal_menu

logger = logging.getLogger("dynamic_function")


def _scenes_dir(game_name: str) -> str:
    return game_path(game_name, "Game", "Scenes")


def _scene_name(scene: str) -> str:
    """Normalize a scene name or filename to a safe Game/Scenes key."""
    name = str(scene or "").strip()
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"Invalid scene name: {scene!r}")
    return name


def _scene_path(game_name: str, scene: str) -> str:
    return os.path.join(_scenes_dir(game_name), f"{_scene_name(scene)}.json")


def _load_scene_config(game_name: str, scene: str) -> Dict[str, Any]:
    """Load a scene config, accepting old bare-slot arrays for compatibility."""
    raw = _read_json(_scene_path(game_name, scene))
    if raw is None:
        raise ValueError(f"Unknown scene: {scene!r}")
    if isinstance(raw, list):
        return {"description": "", "slots": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"Scene {scene!r} must be a JSON object")
    slots = raw.get("slots")
    if not isinstance(slots, list):
        raise ValueError(f"Scene {scene!r} must define a slots array")
    return {
        "description": str(raw.get("description") or "").strip(),
        "slots": slots,
    }


def _load_scene(game_name: str, scene: str) -> List[Dict[str, str]]:
    """Load a scene's slots."""
    return cast(List[Dict[str, str]], _load_scene_config(game_name, scene)["slots"])


def scene_bot_role(game_name: str, scene: str, bot_sid: str) -> str:
    """Return the role this game gives a bot, checked against the scene's cast."""
    slots = _load_scene(game_name, scene)
    cast_sids = {_require_str(slot, "bot_sid", f"Scene {scene!r} slot") for slot in slots}
    if bot_sid not in cast_sids:
        raise ValueError(f"Bot {bot_sid!r} is not in scene {scene!r}")
    path = game_path(game_name, "Game", "Roles", f"{bot_sid}.md")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Game {game_name!r} has no role for bot {bot_sid!r}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        role = f.read().strip()
    if not role:
        raise ValueError(f"Role for bot {bot_sid!r} in game {game_name!r} is empty: {path}")
    strangers = sorted(prompt_bot_refs(role) - cast_sids)
    if strangers:
        refs = ", ".join(f"{{{{{sid}}}}}" for sid in strangers)
        raise ValueError(
            f"Role for bot {bot_sid!r} in game {game_name!r} references {refs}, "
            f"which scene {scene!r} does not cast"
        )
    return role


def scene_check_roles(game_name: str, scene: str) -> None:
    """Raise unless every AI slot in the scene has a role that names only its cast."""
    for slot in _load_scene(game_name, scene):
        if slot.get("state") == "ai":
            scene_bot_role(game_name, scene, _require_str(slot, "bot_sid", f"Scene {scene!r} slot"))


def scene_slot_default_location(game_name: str, scene: str, slot_key: str) -> str:
    """Return the standable defaultLocation a scene declares for one slot."""
    slot = next((row for row in _load_scene(game_name, scene) if row.get("key") == slot_key), None)
    if slot is None:
        raise ValueError(f"Scene {scene!r} has no slot {slot_key!r}")
    location = _require_str(slot, "defaultLocation", f"Scene {scene!r} slot {slot_key!r}")
    load_location(game_name, location)
    _require_leaf(game_name, location)
    return location


def _scene_names(game_name: str) -> List[str]:
    """List scene names — the filenames under Game/Scenes/, minus .json."""
    scenes_dir = _scenes_dir(game_name)
    names = []
    for entry in os.listdir(scenes_dir):
        if entry.startswith(".") or not entry.endswith(".json"):
            continue
        names.append(entry[: -len(".json")])
    return sorted(names)


def _scene_rows(game_name: str) -> List[Dict[str, Any]]:
    """Pure data for scene_list: one row per scene definition."""
    rows = []
    for name in _scene_names(game_name):
        config = _load_scene_config(game_name, name)
        rows.append({
            "name": name,
            "slots": len(config["slots"]),
            "description": config["description"],
        })
    return rows


def _scene_picker_choices(game_name: str) -> List[Dict[str, Any]]:
    return [
        {
            "id": row["name"],
            "text": row["name"],
            "description": row["description"],
            "scene": row["name"],
        }
        for row in _scene_rows(game_name)
    ]


async def _scene_pick_dialog(game_name: str,
    *,
    title: str = "Scene",
    heading: str = "Select scene",
) -> Optional[str]:
    choices = _scene_picker_choices(game_name)
    if not choices:
        raise RuntimeError("No scenes found")
    # A scene pick is a single act, not a selection to be confirmed, so this
    # is a menu of buttons: one click chooses and closes.
    choice = await modal_menu(
        choices,
        title=title,
        heading=heading,
    )
    scene = str(choice.get("scene") or choice.get("id") or "").strip()
    return scene or None


@public
async def scene_list(game_name: str) -> List[str]:
    """List available scenes by name."""
    rows = _scene_rows(game_name)
    await atlantis.client_data("Scenes", rows)
    return [row["name"] for row in rows]


@visible
async def scene_pick(game_name: str) -> Optional[str]:
    """Pick a scene using the standard scene picker dialog."""
    return await _scene_pick_dialog(game_name)


@public
async def scene_show(game_name: str, scene: str) -> List[Dict[str, Any]]:
    """Show a scene's slots exactly as scene-definition rows."""
    rows = _load_scene(game_name, scene)
    await atlantis.client_data(scene, rows)
    return rows
