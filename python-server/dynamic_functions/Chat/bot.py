"""Bot tools

A bot is the canonical character/persona/engine unit. Its primary key is the
folder name `sid` under Game/Bots/<sid>/. Prompt, image, default location, and
model settings all live on the bot.
"""

import atlantis
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from .common import _ensure_thumb, _image_data_uri, home_path, _require_str
from .location import _leaf_location_keys
from dynamic_functions.Home.modal import modal_menu

logger = logging.getLogger("dynamic_function")


class BotConfigT(TypedDict):
    """A bot's config.json, normalized into a guaranteed-populated record.

    Core fields (displayName, defaultLocation, provider, model) are required and
    validated at load; the rest carry typed empty-string defaults.
    """
    sid: str
    displayName: str
    defaultLocation: str
    provider: str
    model: str
    baseUrl: str
    apiKeyEnv: str
    image: str


def _bots_dir() -> str:
    return home_path("Game", "Bots")


def _load_bot_json(bot_sid: str) -> dict:
    """Read a bot config."""
    config_path = os.path.join(_bots_dir(), bot_sid, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def _load_bot_prompt(bot_sid: str) -> str:
    """Read a bot prompt."""
    prompt_path = os.path.join(_bots_dir(), bot_sid, "prompt.md")
    if os.path.isfile(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


_PROMPT_BOT_RE = re.compile(r"\{\{\s*(?P<sid>[A-Za-z0-9_.-]+)\s*\}\}")
_PROMPT_ANY_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+\}\}")


def _normalize_roster_names(roster_names: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Normalize a finalized roster mapping of bot sid -> in-game display name."""
    if not roster_names:
        return {}

    names: Dict[str, str] = {}
    for raw_sid, raw_name in roster_names.items():
        sid = str(raw_sid).strip()
        name = str(raw_name).strip()
        if not sid:
            raise ValueError("Roster contains an empty bot sid")
        if not name:
            raise ValueError(f"Roster name for bot {sid!r} is empty")
        _validate_bot(sid)
        names[sid] = name
    return names


def bot_roster_name(bot_sid: str, roster_names: Optional[Mapping[str, str]] = None) -> str:
    """Return the finalized in-game name for a bot sid."""
    _validate_bot(bot_sid)
    names = _normalize_roster_names(roster_names)
    if bot_sid in names:
        return names[bot_sid]
    raw = _load_bot_json(bot_sid)
    default_name = str(raw.get("displayName", bot_sid) or bot_sid).strip()
    return default_name or bot_sid


def render_bot_prompt(bot_sid: str, roster_names: Optional[Mapping[str, str]] = None) -> str:
    """Render Game/Bots/<sid>/prompt.md with finalized roster names.

    Supported placeholders:
    - {{<sid>}} for any bot in the roster, including the current bot
    """
    _validate_bot(bot_sid)
    names = _normalize_roster_names(roster_names)
    template = _load_bot_prompt(bot_sid)

    def replace_bot(match: re.Match[str]) -> str:
        referenced_sid = match.group("sid")
        return bot_roster_name(referenced_sid, names)

    rendered = _PROMPT_BOT_RE.sub(replace_bot, template)
    unresolved = _PROMPT_ANY_PLACEHOLDER_RE.search(rendered)
    if unresolved:
        raise ValueError(
            f"Unsupported prompt placeholder {unresolved.group(0)!r} in bot {bot_sid!r}"
        )
    return rendered


def _validate_default_location(bot_sid: str, location: str) -> None:
    """Validate a bot defaultLocation against standable location keys."""
    if not location:
        return
    valid_locations = set(_leaf_location_keys())
    if location not in valid_locations:
        valid = ", ".join(sorted(valid_locations)) or "none"
        raise ValueError(
            f"Bot {bot_sid!r} has invalid defaultLocation {location!r}. "
            f"Expected one of: {valid}"
        )


def bot_thumb(bot_sid: str) -> str:
    """Return a thumbnail path for a bot portrait, if one exists."""
    bot_data = _load_bot_json(bot_sid)
    image_file = str(bot_data.get("image") or "").strip()
    if not image_file:
        return ""
    image_path = os.path.join(_bots_dir(), bot_sid, image_file)
    if not os.path.isfile(image_path):
        return ""
    return _ensure_thumb(image_path)


def bot_image_data(bot_sid: str) -> str:
    """Return a data URI for a bot portrait thumbnail, if one exists."""
    return _image_data_uri(bot_thumb(bot_sid))


def _bot_picker_choices(current_bot_sid: str = "") -> List[Dict[str, Any]]:
    choices: List[Dict[str, Any]] = []
    current_bot_sid = str(current_bot_sid or "").strip()
    bots_dir = _bots_dir()
    if not os.path.isdir(bots_dir):
        return choices
    for bot_sid in sorted(os.listdir(bots_dir)):
        entry_dir = os.path.join(bots_dir, bot_sid)
        if not os.path.isdir(entry_dir) or bot_sid.startswith(".") or bot_sid == "__pycache__":
            continue
        bot_data = _load_bot_json(bot_sid)
        display_name = str(bot_data.get("displayName") or bot_sid)
        current_marker = "Current" if bot_sid == current_bot_sid else ""
        choices.append({
            "id": bot_sid,
            "text": f"{display_name}{' (current)' if current_marker else ''}",
            "bot_sid": bot_sid,
            "columns": [
                {"type": "image", "src": bot_image_data(bot_sid), "alt": display_name},
                display_name,
                bot_sid,
                current_marker,
            ],
            "column_headers": ["", "Bot", "SID", ""],
        })
    return choices


async def _bot_pick_dialog(
    *,
    title: str = "Bot",
    heading: str = "Select bot",
    current_bot_sid: str = "",
) -> Optional[str]:
    choices = _bot_picker_choices(current_bot_sid)
    if not choices:
        raise RuntimeError("No bots found")
    choice = await modal_menu(
        choices,
        title=title,
        heading=heading,
    )
    if choice is None:
        return None
    bot_sid = str(choice.get("bot_sid") or choice.get("id") or "").strip()
    return bot_sid or None


def _bot_rows() -> List[Dict[str, Any]]:
    """Pure data: list available bots. No client side effects."""
    bots_dir = _bots_dir()
    bots: List[Dict[str, Any]] = []
    if not os.path.isdir(bots_dir):
        return bots
    for entry in sorted(os.listdir(bots_dir)):
        entry_dir = os.path.join(bots_dir, entry)
        if not os.path.isdir(entry_dir) or entry.startswith(".") or entry == "__pycache__":
            continue
        bot_data = _load_bot_json(entry)
        mtimes = []
        for sub_root, _dirs, sub_files in os.walk(entry_dir):
            for filename in sub_files:
                mtimes.append(os.path.getmtime(os.path.join(sub_root, filename)))
        updated = datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M') if mtimes else ''

        provider = bot_data.get("provider", "")
        model = bot_data.get("model", "")
        model_label = f"{provider}: {model}" if provider and model else (model or provider)
        default_location = str(bot_data.get("defaultLocation", "") or "")
        _validate_default_location(entry, default_location)

        bots.append({
            "sid": entry,
            "displayName": bot_data.get("displayName", entry),
            "image": bot_image_data(entry),
            "defaultLocation": default_location,
            "prompt": render_bot_prompt(entry),
            "model": model_label,
            "updated": updated,
        })
    return bots


@public
async def bot_list() -> List[Dict[str, Any]]:
    """List bots — config metadata only."""
    bots = _bot_rows()
    await atlantis.client_data("Bots", bots, column_formatter={
        "prompt": {"type": "markdown", "maxWidth": "80ch"},
    })
    return bots


@visible
async def bot_pick() -> Optional[str]:
    """Pick a bot using the standard bot picker dialog."""
    return await _bot_pick_dialog()


@public
async def prompt_assemble(bot_sid: str, roster_names: Optional[Dict[str, str]] = None) -> str:
    """Render a bot prompt using a finalized roster mapping of bot sid -> desired name."""
    return render_bot_prompt(bot_sid, roster_names)




def bot_default_location(bot_sid: str) -> Optional[str]:
    """Return a bot-specific default location from config.json, if set."""
    location = _load_bot_json(bot_sid).get("defaultLocation", "")
    default_location = str(location).strip()
    _validate_default_location(bot_sid, default_location)
    return default_location or None


def bot_entry_location(bot_sid: str) -> str:
    """Return the entry location for a bot, or raise if not configured."""
    location = bot_default_location(bot_sid)
    if not location:
        raise ValueError(
            f"No defaultLocation configured for bot {bot_sid!r}. "
            f"Set defaultLocation in the bot config.json."
        )
    return location


def _validate_bot(bot_sid: str) -> None:
    """Validate a bot folder."""
    if not os.path.isdir(os.path.join(_bots_dir(), bot_sid)):
        raise ValueError(f"Bot folder not found: {bot_sid}")


def load_bot(bot_sid: str) -> BotConfigT:
    """Load a bot's config as a typed, fully-populated record.

    The single boundary where a loose config.json becomes a known shape. Raises
    if the sid has no folder (foreign-key check) or if a core field is missing —
    a malformed bot fails loudly here instead of rendering a half-blank row.
    """
    _validate_bot(bot_sid)
    raw = _load_bot_json(bot_sid)
    label = f"Bot {bot_sid!r} config.json"
    return BotConfigT(
        sid=bot_sid,
        displayName=_require_str(raw, "displayName", label),
        defaultLocation=_require_str(raw, "defaultLocation", label),
        provider=_require_str(raw, "provider", label),
        model=_require_str(raw, "model", label),
        baseUrl=str(raw.get("baseUrl", "")),
        apiKeyEnv=str(raw.get("apiKeyEnv", "")),
        image=str(raw.get("image", "")),
    )
