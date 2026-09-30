"""Bot tools

A bot is the canonical character/persona/engine unit, keyed by `sid`. Chat
knows nothing about a bot until it joins a game: whatever adds the bot hands
over its full BotConfigT (prompt and image data URI included), and the game
caches it at Data/games/<game_key>/bots/<sid>.json. Everything in Chat reads
that cache — never the Bot app's files. Where a bot enters the world belongs
to the scene slot it fills, not the bot.
"""

import atlantis
import base64
import logging
import os
import re
from typing import Any, Dict, List, Mapping, TypedDict

from .common import _read_json, _write_json, _require_str
from .game import require_membership

logger = logging.getLogger("dynamic_function")


class BotConfigT(TypedDict):
    """A bot as handed to a game when it joins, and as the game caches it.

    Core fields (sid, displayName, provider, model, prompt) are required and
    validated at the boundary; the rest carry typed empty-string defaults.
    `prompt` is the raw template ({{<sid>}} placeholders unresolved); `image` is
    an image data URI.
    """
    sid: str
    displayName: str
    provider: str
    model: str
    baseUrl: str
    apiKeyEnv: str
    prompt: str
    image: str


_BOT_SID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_BOTS_DIRNAME = "bots"


def validate_bot_config(raw: Any) -> BotConfigT:
    """The single boundary where a loose bot dict becomes a known shape."""
    if not isinstance(raw, dict):
        raise TypeError(f"Bot config must be an object, got {type(raw).__name__}")
    sid = _require_str(raw, "sid", "Bot config")
    if not _BOT_SID_RE.fullmatch(sid):
        raise ValueError(f"Invalid bot sid: {sid!r}")
    label = f"Bot {sid!r} config"
    return BotConfigT(
        sid=sid,
        displayName=_require_str(raw, "displayName", label),
        provider=_require_str(raw, "provider", label),
        model=_require_str(raw, "model", label),
        baseUrl=str(raw.get("baseUrl", "")),
        apiKeyEnv=str(raw.get("apiKeyEnv", "")),
        prompt=_require_str(raw, "prompt", label),
        image=str(raw.get("image", "")),
    )


def _bots_dir(game_key: str) -> str:
    return os.path.join(require_membership(game_key), _BOTS_DIRNAME)


def _bot_cache_path(game_key: str, bot_sid: str) -> str:
    if not _BOT_SID_RE.fullmatch(bot_sid):
        raise ValueError(f"Invalid bot sid: {bot_sid!r}")
    return os.path.join(_bots_dir(game_key), f"{bot_sid}.json")


def cache_bot(game_key: str, bot: BotConfigT) -> None:
    """Store a joining bot's config in the game's cache."""
    _write_json(_bot_cache_path(game_key, bot["sid"]), bot)


def load_bot(game_key: str, bot_sid: str) -> BotConfigT:
    """Load a joined bot from the game's cache; raises if it never joined."""
    raw = _read_json(_bot_cache_path(game_key, bot_sid))
    if raw is None:
        raise ValueError(f"Bot {bot_sid!r} has not joined game {game_key!r}")
    bot = validate_bot_config(raw)
    if bot["sid"] != bot_sid:
        raise ValueError(f"Cached bot {bot_sid!r} has mismatched sid {bot['sid']!r}")
    return bot


def _joined_bot_sids(game_key: str) -> List[str]:
    bots_dir = _bots_dir(game_key)
    if not os.path.isdir(bots_dir):
        return []
    return sorted(
        entry[: -len(".json")]
        for entry in os.listdir(bots_dir)
        if entry.endswith(".json") and not entry.startswith(".")
    )


_PROMPT_BOT_RE = re.compile(r"\{\{\s*(?P<sid>[A-Za-z0-9_.-]+)\s*\}\}")
_PROMPT_ANY_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+\}\}")


def render_bot_prompt(game_key: str, bot_sid: str, roster_names: Mapping[str, str]) -> str:
    """Render a joined bot's prompt with this game's roster names.

    Supported placeholders:
    - {{<sid>}} for any bot sid named in `roster_names`, including the current bot
    """
    template = load_bot(game_key, bot_sid)["prompt"]

    def replace_bot(match: re.Match[str]) -> str:
        referenced_sid = match.group("sid")
        name = str(roster_names.get(referenced_sid) or "").strip()
        if not name:
            raise ValueError(
                f"Prompt for bot {bot_sid!r} references {{{{{referenced_sid}}}}}, "
                f"which has no name in game {game_key!r}"
            )
        return name

    rendered = _PROMPT_BOT_RE.sub(replace_bot, template)
    unresolved = _PROMPT_ANY_PLACEHOLDER_RE.search(rendered)
    if unresolved:
        raise ValueError(
            f"Unsupported prompt placeholder {unresolved.group(0)!r} in bot {bot_sid!r}"
        )
    return rendered


_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF8", "gif"),
    (b"RIFF", "webp"),
)


def _image_kind(data: bytes) -> str:
    for signature, kind in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return kind
    raise ValueError("Bot image is not a recognized PNG, JPEG, GIF, or WebP")


def bot_image_data(game_key: str, bot_sid: str) -> str:
    """Return a joined bot's portrait as a data URI, or "" if it has none."""
    return load_bot(game_key, bot_sid)["image"]


def bot_image_file(game_key: str, bot_sid: str) -> str:
    """Decode a joined bot's portrait next to its cache entry and return the path.

    Raises if the bot has no image — callers that need a file need a picture.
    """
    image = load_bot(game_key, bot_sid)["image"]
    if not image:
        raise FileNotFoundError(f"Bot {bot_sid!r} has no image in game {game_key!r}")
    data = base64.b64decode(image.split(",", 1)[1])
    path = os.path.join(_bots_dir(game_key), f"{bot_sid}.{_image_kind(data)}")
    if not os.path.isfile(path) or os.path.getmtime(path) < os.path.getmtime(_bot_cache_path(game_key, bot_sid)):
        with open(path, "wb") as f:
            f.write(data)
    return path


def _bot_rows(game_key: str) -> List[Dict[str, Any]]:
    """Pure data: this game's joined bots. No client side effects."""
    rows: List[Dict[str, Any]] = []
    for bot_sid in _joined_bot_sids(game_key):
        bot = load_bot(game_key, bot_sid)
        rows.append({
            "sid": bot["sid"],
            "displayName": bot["displayName"],
            "image": bot_image_data(game_key, bot_sid),
            "prompt": bot["prompt"],
            "model": f"{bot['provider']}: {bot['model']}",
        })
    return rows


@public
async def bot_list(game_key: str) -> List[Dict[str, Any]]:
    """List the bots that have joined this game."""
    bots = _bot_rows(game_key)
    await atlantis.client_data("Bots", bots, column_formatter={
        "prompt": {"type": "markdown", "maxWidth": "80ch"},
    })
    return bots


@public
async def prompt_assemble(game_key: str, bot_sid: str, roster_names: Dict[str, str]) -> str:
    """Render a joined bot's prompt using a roster mapping of bot sid -> desired name."""
    return render_bot_prompt(game_key, bot_sid, roster_names)
