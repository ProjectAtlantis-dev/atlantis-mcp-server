"""Shared game helpers"""

import atlantis
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mcp_server")

# ---------------------------------------------------------------------------
# Game data I/O
# ---------------------------------------------------------------------------

def home_path(*parts: str) -> str:
    """Resolve a path under python-server/dynamic_functions/."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", *parts))


def _safe_id(value: str, label: str = "id") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "").strip())
    if not safe:
        raise ValueError(f"Cannot use an empty {label}")
    return safe


def game_dir(game_key: str) -> str:
    """Get a game data directory path"""
    return home_path("Data", "games", _safe_id(game_key, "game_key"))


def require_game_dir(game_key: str) -> str:
    """Get an existing game data directory"""
    path = game_dir(game_key)
    if not os.path.isdir(path):
        raise RuntimeError(f"Invalid game '{game_key}'")
    return path


def create_game_dir(game_key: str) -> str:
    """Create a game data directory"""
    path = game_dir(game_key)
    os.makedirs(path, exist_ok=False)
    return path


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


def _bots_dir() -> str:
    return home_path("Game", "Bots")


def _load_bot_config(bot_sid: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """Load a bot config — folder name is the sid"""
    bot_dir = os.path.join(_bots_dir(), bot_sid)
    config_path = os.path.join(bot_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["_botDir"] = bot_dir
    return cfg, bot_sid


def _available_bot_sids() -> List[str]:
    """List known bot sids (folder names)"""
    bots_dir = _bots_dir()
    if not os.path.isdir(bots_dir):
        return []
    sids = []
    for entry in os.listdir(bots_dir):
        if entry.startswith(".") or entry == "__pycache__":
            continue
        if os.path.isfile(os.path.join(bots_dir, entry, "config.json")):
            sids.append(entry)
    return sorted(sids)


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

THUMB_WIDTH = 360
THUMB_QUALITY = 80
THUMB_SUFFIX = "_thumb.jpg"


def _thumb_path_for(image_path: str) -> str:
    """Get a thumbnail path for an image"""
    base, _ = os.path.splitext(image_path)
    return base + THUMB_SUFFIX


def _ensure_thumb(image_path: str) -> str:
    """Create or reuse a thumbnail"""
    logger.info(f"[thumb] _ensure_thumb called: {image_path}")
    thumb = _thumb_path_for(image_path)
    try:
        # Reuse current thumbnails
        if os.path.isfile(thumb) and os.path.getmtime(thumb) >= os.path.getmtime(image_path):
            logger.info(f"[thumb] cache hit: {thumb}")
            return thumb

        from PIL import Image as _PILImage

        img = _PILImage.open(image_path)
        ratio = THUMB_WIDTH / img.width
        new_h = int(img.height * ratio)
        img = img.resize((THUMB_WIDTH, new_h), _PILImage.Resampling.LANCZOS)
        img = img.convert("RGB")  # JPEG-compatible
        img.save(thumb, "JPEG", quality=THUMB_QUALITY)
        logger.info(f"[thumb] generated: {thumb} ({os.path.getsize(thumb)} bytes)")
        return thumb
    except Exception as exc:
        logger.warning(f"[thumb] FAILED for {image_path}: {exc}")
        return image_path


@visible
def thumbify(image_path: str) -> str:
    """Create a thumbnail for an image"""
    if not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    return _ensure_thumb(image_path)
