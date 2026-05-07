"""Camera tools"""

import json
import os

from dynamic_functions.Home.character import game_data_dir, _require_active_game


def _camera_path() -> str:
    return os.path.join(game_data_dir(), "camera.json")


def _camera_read() -> str:
    """Internal: read camera location from session-pinned game."""
    path = _camera_path()
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("location", "")


def _camera_write(location: str) -> None:
    """Internal: write camera location to session-pinned game."""
    path = _camera_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"location": location}, f)
    os.replace(tmp, path)


@visible
def camera_get(game_key: str) -> str:
    """Get the camera location"""
    _require_active_game(game_key)
    return _camera_read()


@visible
def camera_set(game_key: str, location: str) -> None:
    """Set the camera location"""
    _require_active_game(game_key)
    _camera_write(location)


# Internal alias used by movement code that already has a session-pinned game.
set_camera = _camera_write
