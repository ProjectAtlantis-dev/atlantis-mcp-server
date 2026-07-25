"""Scene tools

A scene is a named roster of slots with a short description. Its identity is
the filename: Game/Scenes/<name>.json, the same way a bot's identity is its
folder name.
"""

import atlantis
import logging
import os
import re
from typing import Any, Dict, List, Optional, cast

from .common import home_path, _read_json
from .bot import load_bot
from dynamic_functions.Home.modal import modal_radio

logger = logging.getLogger("dynamic_function")


def _scenes_dir() -> str:
    return home_path("Game", "Scenes")


def _scene_name(scene: str) -> str:
    """Normalize a scene name or filename to a safe Game/Scenes key."""
    name = str(scene or "").strip()
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"Invalid scene name: {scene!r}")
    return name


def _scene_path(scene: str) -> str:
    return os.path.join(_scenes_dir(), f"{_scene_name(scene)}.json")


def _load_scene_config(scene: str) -> Dict[str, Any]:
    """Load a scene config, accepting old bare-slot arrays for compatibility."""
    raw = _read_json(_scene_path(scene))
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


def _load_scene(scene: str) -> List[Dict[str, str]]:
    """Load a scene's slots."""
    return cast(List[Dict[str, str]], _load_scene_config(scene)["slots"])


def _scene_names() -> List[str]:
    """List scene names — the filenames under Game/Scenes/, minus .json."""
    scenes_dir = _scenes_dir()
    names = []
    for entry in os.listdir(scenes_dir):
        if entry.startswith(".") or not entry.endswith(".json"):
            continue
        names.append(entry[: -len(".json")])
    return sorted(names)


def _scene_rows() -> List[Dict[str, Any]]:
    """Pure data for scene_list: one row per scene definition."""
    rows = []
    for name in _scene_names():
        config = _load_scene_config(name)
        rows.append({
            "name": name,
            "slots": len(config["slots"]),
            "description": config["description"],
        })
    return rows


def _scene_picker_choices() -> List[Dict[str, Any]]:
    return [
        {
            "id": row["name"],
            "text": row["name"],
            "description": row["description"],
            "scene": row["name"],
        }
        for row in _scene_rows()
    ]


async def _scene_pick_dialog(
    *,
    title: str = "Scene",
    heading: str = "Select scene",
) -> Optional[str]:
    choices = _scene_picker_choices()
    if not choices:
        raise RuntimeError("No scenes found")
    choice = await modal_radio(
        choices,
        title=title,
        heading=heading,
        cancel_label="",
    )
    scene = str(choice.get("scene") or choice.get("id") or "").strip()
    return scene or None


@public
async def scene_list() -> List[str]:
    """List available scenes by name."""
    rows = _scene_rows()
    await atlantis.client_data("Scenes", rows)
    return [row["name"] for row in rows]


@visible
async def scene_pick() -> Optional[str]:
    """Pick a scene using the standard scene picker dialog."""
    return await _scene_pick_dialog()


@public
async def scene_show(scene: str) -> List[Dict[str, Any]]:
    """Show a scene's slots exactly as scene-definition rows.

    Resolving the sid doubles as foreign-key validation: an unknown bot_sid
    raises rather than rendering a dangling row.
    """
    rows = _load_scene(scene)
    for row in rows:
        load_bot(row["bot_sid"])
    await atlantis.client_data(scene, rows)
    return rows
