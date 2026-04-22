"""Camera — per-game location that determines the background image."""

import json
import os
from typing import Optional

from dynamic_functions.Home.character import game_data_dir


def _camera_path() -> str:
    return os.path.join(game_data_dir(), "camera.json")

@visible
def camera_get() -> str:
    """Return the current camera location for the active game, or empty string."""
    path = _camera_path()
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("location", "")

@visible
def camera_set(location: str) -> None:
    """Persist the camera location for the active game."""
    path = _camera_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"location": location}, f)
    os.replace(tmp, path)


# Alias for convenience
set_camera = camera_set
