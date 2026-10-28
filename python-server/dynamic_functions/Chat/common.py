"""Shared helpers — path resolution, JSON I/O, and image data URIs."""

import base64
import json
import logging
import os
import re
from typing import Any, Dict


logger = logging.getLogger("dynamic_function")


# ---------------------------------------------------------------------------
# Paths & JSON I/O
# ---------------------------------------------------------------------------

# Each game is a folder beside these scripts: Chat/<game_name>/ holds that
# game's static Game/ content and its live Data/.
CHAT_HOME = os.path.abspath(os.path.dirname(__file__))

_GAME_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


def home_path(*parts: str) -> str:
    """Resolve a path under the Chat app folder."""
    return os.path.join(CHAT_HOME, *parts)


def game_path(game_name: str, *parts: str) -> str:
    """Resolve a path under one game's folder, Chat/<game_name>/."""
    if not isinstance(game_name, str) or not _GAME_NAME_RE.fullmatch(game_name):
        raise ValueError(f"Invalid game name: {game_name!r}")
    root = home_path(game_name)
    if not os.path.isdir(os.path.join(root, "Game")):
        raise ValueError(f"Unknown game: {game_name!r}")
    return os.path.join(root, *parts)


def _read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
            return json.loads(raw) if raw.strip() else default
    except FileNotFoundError:
        return default


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _require_str(data: Dict[str, Any], key: str, label: str) -> str:
    """Pull a non-empty string field from a config dict, or raise.

    The strict-config boundary: where a loose config.json field is asserted
    present before it becomes part of a typed record. `label` names the owner
    for the error (e.g. "Bot 'chad' config.json").
    """
    value = str(data.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} is missing required field {key!r}")
    return value


# What every image getter hands out when the picture is missing: a broken-image
# icon, so the gap is visible instead of an empty cell. The viewBox margin keeps
# the icon whole under `object-fit: cover`.
_MISSING_IMAGE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-6 -6 36 36">'
    '<path d="M4 4h16v9l-4-3-3 3-2-2-3 3 2 2H4z M20 15v5H9l3-3 2 2 3-3z" fill="none" '
    'stroke="#fffaf0" stroke-opacity="0.45" stroke-width="1.5" stroke-linejoin="round"/>'
    '<circle cx="8.5" cy="8.5" r="1.5" fill="#fffaf0" fill-opacity="0.45"/>'
    '</svg>'
)
MISSING_IMAGE_URI = "data:image/svg+xml;base64," + base64.b64encode(_MISSING_IMAGE_SVG.encode("ascii")).decode("ascii")


_IMAGE_MIMES = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}


def _image_data_uri(path: str) -> str:
    """Data URI for an image file, or MISSING_IMAGE_URI when there is none."""
    if not path or not os.path.isfile(path):
        return MISSING_IMAGE_URI
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in _IMAGE_MIMES:
        raise ValueError(f"Unsupported image type: {path!r}")
    with open(path, "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/{_IMAGE_MIMES[ext]};base64,{data}"
