"""Casting — runtime binding of bots (or humans) to slots, per game.

Replaces the old `character.py` and the `Game/Characters/` folder.

## Concepts

- **Slot** (`Game/Slots/<slot>/`): a playable unit in a scenario.
  Has a `defaultBot` and a `defaultLocation` baked into its config.
- **Bot** (`Game/Bots/<sid>/`): an AI character with prompt, model,
  image, etc. Reusable across slots and games.
- **Casting**: which occupant currently fills which slot, *per game*. An
  occupant is either an AI bot sid or a human user sid. If no casting
  override exists for a slot, the slot's `defaultBot` is in the chair.

## On-disk

- Slot defaults:  `Game/Slots/<slot>/config.json`     (asset, shared)
- Per-slot prompt flavor for a bot:
                   `Game/Slots/<slot>/casting/<bot>.md`   (asset, shared)
- Per-game override: `Data/games/<game_key>/casting.json`     (runtime)
    `{ "<slot>": { "occupant": "<sid>", "kind": "ai" | "human", "displayName": "..." } }`

## Public surface

- `get_casting(game_key)` → `{ slot: {occupant, kind, displayName, source} }`
- `set_casting(game_key, slot, occupant, kind, displayName="")`
- `clear_casting(game_key, slot)`
- `slot_for_occupant(game_key, sid)` → slot or None  (reverse lookup)
- `load_casting_prompt(slot, bot)` — reads `Slots/<slot>/casting/<bot>.md`
- Legacy shims used by older callers: `_load_characters`, `_find_character`,
  `_character_rows`, `is_bot_driven`, `load_character_prompt`.
"""

import atlantis
import logging
import os
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.common import (
    _load_bot_config,
    _read_json,
    _write_json,
    home_path,
    require_game_dir,
)

logger = logging.getLogger("mcp_server")


# ---------------------------------------------------------------------------
# Slot enumeration / config
# ---------------------------------------------------------------------------

def _slots_dir() -> str:
    return home_path("Game", "Slots")


def _list_slot_keys() -> List[str]:
    d = _slots_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        entry for entry in os.listdir(d)
        if os.path.isdir(os.path.join(d, entry))
        and not entry.startswith(".") and entry != "__pycache__"
    )


def _slot_config(slot: str) -> Dict[str, Any]:
    p = os.path.join(_slots_dir(), slot, "config.json")
    return _read_json(p, {}) or {}


def slot_default_bot(slot: str) -> str:
    return str(_slot_config(slot).get("defaultBot", "") or "")


# ---------------------------------------------------------------------------
# Per-game casting overrides
# ---------------------------------------------------------------------------

def _casting_path(game_key: str) -> str:
    return os.path.join(require_game_dir(game_key), "casting.json")


def _read_overrides(game_key: str) -> Dict[str, Dict[str, Any]]:
    data = _read_json(_casting_path(game_key), {}) or {}
    return data if isinstance(data, dict) else {}


def _write_overrides(game_key: str, data: Dict[str, Dict[str, Any]]) -> None:
    _write_json(_casting_path(game_key), data)


def get_casting(game_key: str) -> Dict[str, Dict[str, Any]]:
    """Return current casting for every slot. Merges per-game overrides on top
    of each slot's defaultBot. Slots with no occupant at all are still
    included with kind="empty" (which only happens when there is no default
    AND no override)."""
    overrides = _read_overrides(game_key)
    out: Dict[str, Dict[str, Any]] = {}
    for slot in _list_slot_keys():
        cfg = _slot_config(slot)
        slot_display = cfg.get("displayName", slot)
        ov = overrides.get(slot)
        if ov and ov.get("occupant"):
            out[slot] = {
                "slot": slot,
                "slotDisplayName": slot_display,
                "occupant": ov["occupant"],
                "kind": ov.get("kind", "ai"),
                "displayName": ov.get("displayName") or _occupant_display_name(ov["occupant"], ov.get("kind", "ai")),
                "source": "override",
            }
            continue
        default_bot = cfg.get("defaultBot", "")
        if default_bot:
            out[slot] = {
                "slot": slot,
                "slotDisplayName": slot_display,
                "occupant": default_bot,
                "kind": "ai",
                "displayName": _occupant_display_name(default_bot, "ai"),
                "source": "default",
            }
            continue
        out[slot] = {
            "slot": slot,
            "slotDisplayName": slot_display,
            "occupant": "",
            "kind": "empty",
            "displayName": "",
            "source": "empty",
        }
    return out


def _occupant_display_name(occupant: str, kind: str) -> str:
    if kind == "ai":
        loaded = _load_bot_config(occupant)
        if loaded:
            cfg, _ = loaded
            return cfg.get("displayName", occupant)
    return occupant


def set_casting(game_key: str, slot: str, occupant: str, kind: str = "ai", displayName: str = "") -> None:
    """Cast a slot. `kind` is "ai" (occupant is a bot sid) or "human"
    (occupant is a user sid)."""
    if slot not in _list_slot_keys():
        raise ValueError(f"Unknown slot: {slot}")
    if kind not in ("ai", "human"):
        raise ValueError(f"Unknown kind: {kind}")
    if kind == "ai" and not _load_bot_config(occupant):
        raise ValueError(f"Unknown bot: {occupant}")
    overrides = _read_overrides(game_key)
    overrides[slot] = {
        "occupant": occupant,
        "kind": kind,
        "displayName": displayName or _occupant_display_name(occupant, kind),
    }
    _write_overrides(game_key, overrides)


def clear_casting(game_key: str, slot: str) -> None:
    """Drop the per-game override so the slot falls back to its defaultBot."""
    overrides = _read_overrides(game_key)
    if slot in overrides:
        del overrides[slot]
        _write_overrides(game_key, overrides)


def slot_for_occupant(game_key: str, sid: str) -> Optional[str]:
    """Reverse lookup: which slot is this sid currently filling in this game?"""
    for slot, info in get_casting(game_key).items():
        if info.get("occupant") == sid and info.get("kind") != "empty":
            return slot
    return None


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------

def load_casting_prompt(slot: str, bot: str) -> str:
    """Read this bot's specific take on this slot.
    Stored at `Game/Slots/<slot>/casting/<bot>.md`. Empty string if missing."""
    path = os.path.join(_slots_dir(), slot, "casting", f"{bot}.md")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# Back-compat alias (old name).
load_character_prompt = lambda sid, role: load_casting_prompt(role, sid)


# ---------------------------------------------------------------------------
# Legacy shims used elsewhere in the codebase
# ---------------------------------------------------------------------------

def is_bot_driven(sid: str) -> bool:
    """True iff `sid` is a known bot AND no live session has claimed its
    chat_slot (i.e. no human is currently typing as this bot)."""
    from dynamic_functions.Home.session import chat_slot_claimed
    if _load_bot_config(sid) is None:
        return False
    return not chat_slot_claimed(sid)


def _load_characters(game_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return one record per filled slot in this game (or, if game_key is None,
    a synthetic view from slot defaults — used by the legacy `character.py`
    shape, where the binding came purely from the filesystem)."""
    if game_key is None:
        # Pre-game: derive from slot defaults only.
        rows: List[Dict[str, Any]] = []
        for slot in _list_slot_keys():
            cfg = _slot_config(slot)
            bot = cfg.get("defaultBot", "")
            if not bot:
                continue
            loaded = _load_bot_config(bot)
            if not loaded:
                continue
            pcfg, _ = loaded
            rows.append({
                "sid": bot,
                "role": slot,
                "displayName": pcfg.get("displayName", bot),
                "prompt": load_casting_prompt(slot, bot),
            })
        return rows

    rows: List[Dict[str, Any]] = []
    for slot, info in get_casting(game_key).items():
        if info.get("kind") == "empty":
            continue
        rows.append({
            "sid": info["occupant"],
            "role": slot,
            "displayName": info.get("displayName", info["occupant"]),
            "prompt": load_casting_prompt(slot, info["occupant"]) if info.get("kind") == "ai" else "",
        })
    return rows


def _find_character(sid: str, game_key: Optional[str] = None) -> Dict[str, Any]:
    """Find a casting record by occupant sid."""
    for ch in _load_characters(game_key):
        if ch["sid"] == sid:
            return ch
    raise ValueError(
        f"No casting found for sid: {sid!r}. Set a defaultBot in a slot, "
        f"or use set_casting() to bind them."
    )


def _character_rows(game_key: str) -> List[Dict[str, Any]]:
    """Pure data: characters with current positions. No client side effects."""
    require_game_dir(game_key)
    from dynamic_functions.Home.location import get_positions
    positions = get_positions(game_key)
    return [
        {**ch, "location": positions.get(ch["sid"], "")}
        for ch in _load_characters(game_key)
    ]


# ---------------------------------------------------------------------------
# Prompt introspection
# ---------------------------------------------------------------------------

def build_casting_prompt(game_key: str, slot: str, caller: str = "") -> Dict[str, Any]:
    """Assemble the system prompt that the AI in this slot would currently
    receive. Returns the prompt plus a breakdown of which file each section
    came from. For human-occupied slots, returns kind="human" with no prompt.
    """
    casting = get_casting(game_key)
    info = casting.get(slot)
    if not info or info.get("kind") == "empty":
        raise ValueError(f"Slot {slot!r} has no occupant in game {game_key!r}.")
    if info.get("kind") == "human":
        return {
            "slot": slot,
            "occupant": info["occupant"],
            "kind": "human",
            "displayName": info.get("displayName", info["occupant"]),
            "prompt": "",
            "sources": {},
            "note": "Slot is filled by a human — no AI system prompt is assembled.",
        }

    bot_sid = info["occupant"]

    from dynamic_functions.Home.prompt_common import (
        build_system_prompt, load_bot, load_appearance,
        load_slot_system_prompt,
    )
    from dynamic_functions.Home.location import compose_setting, position_get
    from dynamic_functions.Home.interactions import read_interaction

    bot_md = load_bot(bot_sid)
    appearance_md = load_appearance(bot_sid)
    job_md = load_slot_system_prompt(slot)
    flavor_md = load_casting_prompt(slot, bot_sid)

    pos = position_get(game_key, bot_sid) or _slot_config(slot).get("defaultLocation", "")
    setting = compose_setting(pos) if pos else ""

    history = read_interaction(game_key, bot_sid, caller) if caller else {}

    prompt = build_system_prompt(
        base_prompt=job_md,
        bot=bot_md,
        appearance=appearance_md,
        character_prompt=flavor_md,
        setting=setting,
        caller=caller,
        prior_interaction_count=int(history.get("count") or 0),
        last_interaction_at=history.get("last_interaction_at", ""),
        first_name=history.get("first_name", ""),
    )

    return {
        "slot": slot,
        "occupant": bot_sid,
        "kind": "ai",
        "displayName": info.get("displayName", bot_sid),
        "location": pos,
        "prompt": prompt,
        "sources": {
            "bot":    f"Game/Bots/{bot_sid}/bot.md"      if bot_md   else "(missing)",
            "appearance": f"Game/Bots/{bot_sid}/appearance.md"   if appearance_md else "(missing)",
            "job":        f"Game/Slots/{slot}/system_prompt.md"          if job_md       else "(missing)",
            "casting":    f"Game/Slots/{slot}/casting/{bot_sid}.md"  if flavor_md    else "(none — optional)",
            "setting":    f"composed from location chain rooted at {pos!r}" if setting else "(no location)",
        },
    }


@visible
async def casting_list(game_key: str) -> List[Dict[str, Any]]:
    """Show the cast list for this game — one row per slot. The lobby view:
    who's in each seat, are they AI or human, is this the slot's default or an
    explicit override. Empty slots show kind="empty".
    """
    casting = get_casting(game_key)
    rows: List[Dict[str, Any]] = []
    for slot_key in _list_slot_keys():
        cfg = _slot_config(slot_key)
        info = casting.get(slot_key, {})
        rows.append({
            "slot": slot_key,
            "slotDisplayName": cfg.get("displayName", slot_key),
            "occupant": info.get("occupant", ""),
            "displayName": info.get("displayName", ""),
            "kind": info.get("kind", "empty"),
        })
    await atlantis.client_data("Casting", rows)
    return rows


@visible
async def casting_prompt(game_key: str, slot: str, caller: str = "") -> Dict[str, Any]:
    """Show the assembled system prompt for the AI currently cast in `slot`.

    Useful for debugging "why is this bot saying that" — you get the exact
    prompt the LLM sees, plus a breakdown of which files each section came
    from. Pass `caller` to also fold in the interaction-context block for a
    specific speaker.
    """
    result = build_casting_prompt(game_key, slot, caller=caller)
    await atlantis.client_data(
        f"Casting prompt — {slot}",
        [{"section": k, "source": v} for k, v in result["sources"].items()],
    )
    await atlantis.client_markdown(
        f"### {result['displayName']} as {slot} "
        f"({'AI: ' + result['occupant'] if result['kind'] == 'ai' else 'human'})\n\n"
        f"```\n{result['prompt']}\n```"
    )
    return result
