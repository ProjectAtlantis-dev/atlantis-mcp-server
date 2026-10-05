"""Static bot info served to games.

Each bot lives in <sid>/ next to this file: config.json, prompt.md, and the
image file the config names. bot_get packs them into the one JSON a game takes
when the bot joins (Chat's BotConfigT): config fields plus the raw prompt and
the image thumbnail as a data URI. This app may run on a different machine
than Chat, so it never imports from Chat.
"""

import atlantis
import base64
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger("dynamic_function")

_BOT_SID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_REQUIRED_FIELDS = ("displayName", "provider", "model")

THUMB_WIDTH = 360
THUMB_QUALITY = 80
THUMB_SUFFIX = "_thumb.jpg"


def _bots_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _bot_dir(bot_sid: str) -> str:
    if not _BOT_SID_RE.fullmatch(bot_sid):
        raise ValueError(f"Invalid bot sid: {bot_sid!r}")
    path = os.path.join(_bots_dir(), bot_sid)
    if not os.path.isdir(path):
        raise ValueError(f"Unknown bot: {bot_sid!r}")
    return path


def _bot_sids() -> List[str]:
    return sorted(
        entry for entry in os.listdir(_bots_dir())
        if os.path.isdir(os.path.join(_bots_dir(), entry))
        and not entry.startswith(".") and entry != "__pycache__"
    )


def _load_config(bot_sid: str) -> Dict[str, Any]:
    with open(os.path.join(_bot_dir(bot_sid), "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    for field in _REQUIRED_FIELDS:
        if not str(config.get(field, "")).strip():
            raise ValueError(f"Bot {bot_sid!r} config.json is missing required field {field!r}")
    return config


def _ensure_thumb(image_path: str) -> str:
    """Create or reuse a thumbnail; "" if it can't be made (bot ships with no image)."""
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
    except Exception:
        logger.exception(f"[thumb] FAILED for {image_path}; serving no image")
        return ""


def _image_data_uri(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
    with open(path, "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def bot_image_data(bot_sid: str) -> str:
    """Return a data URI for a bot portrait thumbnail, if one exists."""
    image_file = str(_load_config(bot_sid).get("image", "")).strip()
    if not image_file:
        return ""
    return _image_data_uri(_ensure_thumb(os.path.join(_bot_dir(bot_sid), image_file)))


@public
async def bot_get(bot_sid: str) -> Dict[str, str]:
    """Return a bot's full config, raw prompt, and image thumbnail data URI for joining a game."""
    bot_dir = _bot_dir(bot_sid)
    config = _load_config(bot_sid)
    with open(os.path.join(bot_dir, "prompt.md"), "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError(f"Bot {bot_sid!r} prompt.md is empty")

    return {
        "sid": bot_sid,
        "displayName": str(config["displayName"]),
        "provider": str(config["provider"]),
        "model": str(config["model"]),
        "baseUrl": str(config.get("baseUrl", "")),
        "apiKeyEnv": str(config.get("apiKeyEnv", "")),
        "prompt": prompt,
        "image": bot_image_data(bot_sid),
    }


@public
async def bot_list() -> List[Dict[str, Any]]:
    """List the bots this host offers — config metadata only."""
    bots: List[Dict[str, Any]] = []
    for bot_sid in _bot_sids():
        config = _load_config(bot_sid)
        mtimes = []
        for sub_root, _dirs, sub_files in os.walk(_bot_dir(bot_sid)):
            for filename in sub_files:
                mtimes.append(os.path.getmtime(os.path.join(sub_root, filename)))
        updated = datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M') if mtimes else ''
        with open(os.path.join(_bot_dir(bot_sid), "prompt.md"), "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        bots.append({
            "sid": bot_sid,
            "displayName": str(config["displayName"]),
            "image": bot_image_data(bot_sid),
            "prompt": prompt,
            "model": f"{config['provider']}: {config['model']}",
            "updated": updated,
        })
    await atlantis.client_data("Bots", bots, column_formatter={
        "prompt": {"type": "markdown", "maxWidth": "80ch"},
    })
    return bots
