"""Camera tools"""

import json
import os

from dynamic_functions.Home.common import require_game_dir


def _camera_path(game_key: str) -> str:
    return os.path.join(require_game_dir(game_key), "camera.json")


def _camera_read(game_key: str) -> str:
    path = _camera_path(game_key)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("location", "")


def _camera_write(game_key: str, location: str) -> None:
    path = _camera_path(game_key)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"location": location}, f)
    os.replace(tmp, path)


@visible
def camera_get(game_key: str) -> str:
    """Get the camera location"""
    return _camera_read(game_key)


@visible
def camera_set(game_key: str, location: str) -> None:
    """Set the camera location"""
    _camera_write(game_key, location)
