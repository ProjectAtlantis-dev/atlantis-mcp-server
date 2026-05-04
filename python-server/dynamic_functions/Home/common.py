"""Shared helpers for game callbacks — bot config loading, thumbnails, and spawning."""

import atlantis
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mcp_server")

GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "Games")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")


# ---------------------------------------------------------------------------
# Game-scoped data I/O — all stateful data lives under Data/{game_id}/
# ---------------------------------------------------------------------------

def _safe_id(value: str, label: str = "id") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "").strip())
    if not safe:
        raise ValueError(f"Cannot use an empty {label}")
    return safe


def game_dir(game_id: str) -> str:
    """Return the data directory path for a game_id."""
    return os.path.join(DATA_DIR, _safe_id(game_id, "game_id"))


def require_game_dir(game_id: str) -> str:
    """Return an existing game data directory or fail."""
    path = game_dir(game_id)
    if not os.path.isdir(path):
        raise RuntimeError(f"Unknown game_id '{game_id}'. Create a game first with game_new().")
    return path


def create_game_dir(game_id: str) -> str:
    """Create and return the data directory for a newly minted game_id."""
    path = os.path.join(DATA_DIR, _safe_id(game_id, "game_id"))
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


def read_location_data(game_id: str, location: str) -> Optional[Dict[str, Any]]:
    """Read Data/{game_id}/{location}.json, or None if it doesn't exist."""
    path = os.path.join(game_dir(game_id), f"{location}.json")
    return _read_json(path, None)


def write_location_data(game_id: str, location: str, data: Dict[str, Any]) -> None:
    """Write Data/{game_id}/{location}.json."""
    path = os.path.join(require_game_dir(game_id), f"{location}.json")
    _write_json(path, data)


def list_games() -> list[str]:
    """List all game_ids that have data."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and not d.startswith(".")
        and d != "__pycache__"
        and d != "players"  # ignore legacy
    )


def _bots_dir() -> str:
    from dynamic_functions.Home.main import _get_current_game
    return os.path.join(GAMES_DIR, _get_current_game(), "Bots")


def _load_bot_config(bot_sid: str, bots_dir: Optional[str] = None) -> Optional[Tuple[Dict[str, Any], str]]:
    if bots_dir is None:
        bots_dir = _bots_dir()
    """Find config.json for a bot by sid under bots_dir. Returns (config, folder_name) or None."""
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
    """Return all known bot sids from Bots/ config files."""
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
    """Return the expected thumbnail path for a given source image."""
    base, _ = os.path.splitext(image_path)
    return base + THUMB_SUFFIX


def _ensure_thumb(image_path: str) -> str:
    """Return the path to a thumbnail for *image_path*, creating it if needed.

    The thumb is a JPEG scaled to THUMB_WIDTH px wide (aspect preserved),
    stored alongside the original with a '_thumb.jpg' suffix.
    Regenerated when the source file is newer than the existing thumb.
    Returns the original path if thumbnail generation fails.
    """
    logger.info(f"[thumb] _ensure_thumb called: {image_path}")
    thumb = _thumb_path_for(image_path)
    try:
        # Skip if thumb is already up-to-date
        if os.path.isfile(thumb) and os.path.getmtime(thumb) >= os.path.getmtime(image_path):
            logger.info(f"[thumb] cache hit: {thumb}")
            return thumb

        from PIL import Image as _PILImage

        img = _PILImage.open(image_path)
        ratio = THUMB_WIDTH / img.width
        new_h = int(img.height * ratio)
        img = img.resize((THUMB_WIDTH, new_h), _PILImage.LANCZOS)
        img = img.convert("RGB")  # ensure JPEG-compatible
        img.save(thumb, "JPEG", quality=THUMB_QUALITY)
        logger.info(f"[thumb] generated: {thumb} ({os.path.getsize(thumb)} bytes)")
        return thumb
    except Exception as exc:
        logger.warning(f"[thumb] FAILED for {image_path}: {exc}")
        return image_path


def bot_thumb(bot_sid: str) -> str:
    """Return the filesystem path to a bot's thumbnail image.

    Auto-generates the thumb if it doesn't exist. Returns empty string
    if the bot has no image.
    """
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
    """Generate a thumbnail for any image and return the thumb path.

    The thumbnail is a 360px-wide JPEG stored alongside the original
    with a '_thumb.jpg' suffix. Regenerated only when the source is newer.
    Returns the thumbnail path on success, or the original path on failure.
    """
    if not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    return _ensure_thumb(image_path)
