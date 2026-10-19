"""Static bot info served to games.

Each bot lives in <sid>/ next to this file: config.json, prompt.md, and the
image file the config names. bot_get packs them into the one JSON a game takes
when the bot joins (Chat's BotConfigT): config fields plus the raw prompt and
the full image as a data URI (clients make their own thumbnails). This app may
run on a different machine than Chat, so it never imports from Chat.
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


_IMAGE_MIMES = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}


def _image_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in _IMAGE_MIMES:
        raise ValueError(f"Unsupported bot image type: {path!r}")
    with open(path, "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/{_IMAGE_MIMES[ext]};base64,{data}"


def bot_image_data(bot_sid: str) -> str:
    """Return a data URI for a bot's full portrait, or "" if its config names none."""
    image_file = str(_load_config(bot_sid).get("image", "")).strip()
    if not image_file:
        return ""
    return _image_data_uri(os.path.join(_bot_dir(bot_sid), image_file))


@public
async def bot_get(bot_sid: str) -> Dict[str, str]:
    """Return a bot's full config, raw prompt, and full image data URI for joining a game."""
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
