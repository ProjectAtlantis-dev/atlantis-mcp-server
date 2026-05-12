"""Camera tools"""

import atlantis
import json
import os
from typing import Dict

from dynamic_functions.Home.common import require_game_dir


def _camera_path(game_key: str) -> str:
    return os.path.join(require_game_dir(game_key), "camera.json")


def _camera_read_all(game_key: str) -> Dict[str, str]:
    path = _camera_path(game_key)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _camera_write_all(game_key: str, cameras: Dict[str, str]) -> None:
    path = _camera_path(game_key)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)
    os.replace(tmp, path)


def _resolve_sid(sid: str) -> str:
    sid = (sid or "").strip()
    if sid:
        return sid
    caller = atlantis.get_caller()
    if not caller:
        raise ValueError("sid is required (no caller in context)")
    return caller


@visible
def camera_get(game_key: str, sid: str = "") -> str:
    """Get a camera location for an sid (defaults to caller)"""
    sid = _resolve_sid(sid)
    return _camera_read_all(game_key).get(sid, "")


@visible
def camera_set(game_key: str, location: str, sid: str = "") -> None:
    """Set a camera location for an sid (defaults to caller)"""
    sid = _resolve_sid(sid)
    cameras = _camera_read_all(game_key)
    cameras[sid] = location
    _camera_write_all(game_key, cameras)
