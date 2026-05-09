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
        raise RuntimeError(f"Unknown game '{game_key}'. Create a game first with game_new().")
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


def read_location_data(game_key: str, location: str) -> Optional[Dict[str, Any]]:
    """Read location data"""
    path = os.path.join(game_dir(game_key), f"{location}.json")
    return _read_json(path, None)


def write_location_data(game_key: str, location: str, data: Dict[str, Any]) -> None:
    """Write location data"""
    path = os.path.join(require_game_dir(game_key), f"{location}.json")
    _write_json(path, data)


def _bots_dir() -> str:
    return home_path("Game", "Bots")


def _load_bot_config(bot_sid: str, bots_dir: Optional[str] = None) -> Optional[Tuple[Dict[str, Any], str]]:
    if bots_dir is None:
        bots_dir = _bots_dir()
    """Find a bot config by sid"""
    for entry in os.listdir(bots_dir):
        config_path = os.path.join(bots_dir, entry, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("sid") == bot_sid:
                cfg["_botDir"] = os.path.join(bots_dir, entry)
                return cfg, entry
    return None


def _available_bot_sids(bots_dir: Optional[str] = None) -> List[str]:
    """List known bot sids"""
    if bots_dir is None:
        bots_dir = _bots_dir()
    sids = []
    if not os.path.isdir(bots_dir):
        return sids
    for entry in os.listdir(bots_dir):
        config_path = os.path.join(bots_dir, entry, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                if "sid" in cfg:
                    sids.append(cfg["sid"])
            except (json.JSONDecodeError, OSError):
                pass
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
        img = img.resize((THUMB_WIDTH, new_h), _PILImage.LANCZOS)
        img = img.convert("RGB")  # JPEG-compatible
        img.save(thumb, "JPEG", quality=THUMB_QUALITY)
        logger.info(f"[thumb] generated: {thumb} ({os.path.getsize(thumb)} bytes)")
        return thumb
    except Exception as exc:
        logger.warning(f"[thumb] FAILED for {image_path}: {exc}")
        return image_path


def bot_thumb(bot_sid: str) -> str:
    """Get a bot thumbnail path"""
    logger.info(f"[thumb] bot_thumb called: {bot_sid!r}")
    loaded = _load_bot_config(bot_sid)
    if not loaded:
        logger.warning(f"[thumb] bot config not found: {bot_sid!r}")
        return ""
    cfg, folder = loaded
    image_file = cfg.get("image", "")
    if not image_file:
        logger.warning(f"[thumb] no image for bot: {bot_sid!r}")
        return ""
    image_path = os.path.join(_bots_dir(), folder, image_file)
    logger.info(f"[thumb] bot {bot_sid!r} -> {image_path}")
    if not os.path.isfile(image_path):
        logger.warning(f"[thumb] image file missing: {image_path}")
        return ""
    return _ensure_thumb(image_path)


@visible
def thumbify(image_path: str) -> str:
    """Create a thumbnail for an image"""
    if not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    return _ensure_thumb(image_path)
