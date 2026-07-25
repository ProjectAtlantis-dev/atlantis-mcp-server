"""Shared helpers — path resolution, JSON I/O, and thumbnails."""

import base64
import json
import logging
import os
from typing import Any, Dict

import atlantis

logger = logging.getLogger("dynamic_function")

APP_DEFAULT_BG_ALIGN = "75%"

# ---------------------------------------------------------------------------
# Paths & JSON I/O
# ---------------------------------------------------------------------------

# Data/ and Game/ live inside this app, beside these scripts.
GAME_HOME = os.path.abspath(os.path.dirname(__file__))


def home_path(*parts: str) -> str:
    """Resolve a path under the game data root."""
    return os.path.join(GAME_HOME, *parts)


@visible
def app_default_bg_path() -> str:
    return os.path.join(os.path.dirname(__file__), "builder.jpg")


@public
async def app_bg_default() -> None:
    """Set the chat default background image."""
    await atlantis.set_background(
        app_default_bg_path(),
        vertical_align=APP_DEFAULT_BG_ALIGN,
    )


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


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

THUMB_WIDTH = 360
THUMB_QUALITY = 80
THUMB_SUFFIX = "_thumb.jpg"


def _ensure_thumb(image_path: str) -> str:
    """Create or reuse a thumbnail"""
    logger.info(f"[thumb] _ensure_thumb called: {image_path}")
    base, _ = os.path.splitext(image_path)
    thumb = base + THUMB_SUFFIX
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


def _image_data_uri(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
    with open(path, "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/{mime};base64,{data}"
