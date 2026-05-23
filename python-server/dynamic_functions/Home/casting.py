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
- Iteration helpers: `casting_records(game_key, include_empty=False)`,
  `casting_for_occupant(game_key, sid)`.
- Internal: `is_bot_driven`.
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
    """Return current casting for every slot. casting.json is the source of
    truth — slots with no entry are empty (kind="empty"). The slot's
    `defaultBot` is only a suggestion for an explicit cast; nothing is cast
    until `set_casting` writes it."""
    castings = _read_overrides(game_key)
    out: Dict[str, Dict[str, Any]] = {}
    for slot in _list_slot_keys():
        cfg = _slot_config(slot)
        slot_display = cfg.get("displayName", slot)
        c = castings.get(slot)
        if c and c.get("occupant"):
            out[slot] = {
                "slot": slot,
                "slotDisplayName": slot_display,
                "occupant": c["occupant"],
                "kind": c.get("kind", "ai"),
                "displayName": c.get("displayName") or _occupant_display_name(c["occupant"], c.get("kind", "ai")),
                "entryLocation": c.get("entryLocation", ""),
                "location": c.get("location", ""),
            }
            continue
        out[slot] = {
            "slot": slot,
            "slotDisplayName": slot_display,
            "occupant": "",
            "kind": "empty",
            "displayName": "",
            "entryLocation": "",
            "location": "",
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
    """Cast `occupant` into `slot`. `kind` is "ai" (occupant is a bot sid)
    or "human" (occupant is a user sid). The occupant's entry and current
    location are set to the slot's `defaultLocation`."""
    if slot not in _list_slot_keys():
        raise ValueError(f"Unknown slot: {slot}")
    if kind not in ("ai", "human"):
        raise ValueError(f"Unknown kind: {kind}")
    if kind == "ai" and not _load_bot_config(occupant):
        raise ValueError(f"Unknown bot: {occupant}")
    default_loc = _slot_config(slot).get("defaultLocation", "")
    castings = _read_overrides(game_key)
    castings[slot] = {
        "occupant": occupant,
        "kind": kind,
        "displayName": displayName or _occupant_display_name(occupant, kind),
        "entryLocation": default_loc,
        "location": default_loc,
    }
    _write_overrides(game_key, castings)


def clear_casting(game_key: str, slot: str) -> None:
    """Remove the casting for this slot."""
    castings = _read_overrides(game_key)
    if castings.pop(slot, None) is None:
        return
    _write_overrides(game_key, castings)


# ---------------------------------------------------------------------------
# Location (position) — stored on the casting record
# ---------------------------------------------------------------------------

def casting_location(game_key: str, sid: str) -> Optional[str]:
    """Return the current location of occupant `sid`, or None if not cast."""
    for info in get_casting(game_key).values():
        if info.get("kind") != "empty" and info.get("occupant") == sid:
            return info.get("location") or None
    return None


def set_casting_location(game_key: str, sid: str, location: str) -> None:
    """Update the current location of occupant `sid` in the casting record."""
    castings = _read_overrides(game_key)
    for slot, c in castings.items():
        if c.get("occupant") == sid:
            c["location"] = location
            _write_overrides(game_key, castings)
            return
    raise ValueError(f"No casting found for occupant {sid!r}.")


def occupants_at(game_key: str, location: str) -> List[str]:
    """Return sids of all occupants currently at `location`."""
    return [
        info["occupant"]
        for info in get_casting(game_key).values()
        if info.get("kind") != "empty" and info.get("location") == location
    ]


def casting_at_location(game_key: str, location: str) -> List[Dict[str, Any]]:
    """Return casting records for all occupants at `location`."""
    return [
        dict(info, location=location)
        for info in get_casting(game_key).values()
        if info.get("kind") != "empty" and info.get("location") == location
    ]


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

@visible
async def casting_move(game_key: str, location: str, sid: str = "") -> str:
    """Move an occupant to a location. sid defaults to the caller."""
    from dynamic_functions.Home.location import (
        _connects_to, _is_leaf, _load_location, _location_dir, _require_leaf,
    )
    require_game_dir(game_key)

    if not sid:
        sid = atlantis.get_caller() or ""
    if not sid:
        raise ValueError("Unable to determine character to move (no sid and no caller).")

    entry = casting_for_occupant(game_key, sid)
    display_name = entry.get("displayName", sid)
    display = f"{display_name} ({sid})" if display_name != sid else sid

    current = casting_location(game_key, sid) or ""

    dest = _load_location(location)
    if not dest:
        raise ValueError(f"Unknown location: {location}")
    _require_leaf(location)

    desc = dest.get("displayName", location)

    # No-op when already there
    if current == location:
        await atlantis.client_log(f"\U0001f4cd {display} is already in {desc}")
        return location

    # Enforce adjacency (skip for first move from entry location)
    if current:
        reachable = _connects_to(current)
        if location not in reachable:
            raise ValueError(
                f"Cannot reach {location} from {current}. "
                f"Reachable: {reachable}"
            )

    # Apply movement
    set_casting_location(game_key, sid, location)
    if current:
        current_desc = (_load_location(current) or {}).get("displayName", current)
        await atlantis.client_log(f"\U0001f6b6 {display} moved from {current_desc} to {desc}")
    else:
        await atlantis.client_log(f"\U0001f3db\ufe0f {display} has entered {desc}")
    await atlantis.client_description(
        "Someone has entered.",
        location=location,
    )
    logger.info(f"[game] {sid} moved to {location} (from {current or 'entry'})")
    if atlantis.get_caller() == sid:
        image_name = dest.get("image")
        if image_name:
            import os as _os
            image_path = _os.path.join(_location_dir(location), image_name)
            if _os.path.exists(image_path):
                await atlantis.set_background(image_path)
    from dynamic_functions.Home.chat_callback import greet_entrant
    await greet_entrant(game_key, sid, location)
    return location


@visible
def casting_claim(game_key: str, sid: str) -> Dict[str, Any]:
    """Bind this session to the casting whose occupant is `sid`. Typed text
    comes out of that occupant's mouth. The occupant must already be cast
    via `set_casting`."""
    from dynamic_functions.Home.session import _slot

    slot = slot_for_occupant(game_key, sid)
    if not slot:
        raise ValueError(f"{sid!r} is not cast in any slot for game {game_key!r}.")

    _slot()["casting"] = sid
    return {"casting": sid, "slot": slot, "location": casting_location(game_key, sid) or ""}


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


# ---------------------------------------------------------------------------
# Internal helpers used elsewhere in the codebase
# ---------------------------------------------------------------------------

def is_bot_driven(sid: str) -> bool:
    """True iff `sid` is a known bot AND no live session is driving its
    casting (i.e. no human is currently typing as this bot)."""
    from dynamic_functions.Home.session import casting_is_claimed
    if _load_bot_config(sid) is None:
        return False
    return not casting_is_claimed(sid)


def casting_records(game_key: str, include_empty: bool = False) -> List[Dict[str, Any]]:
    """Flat list of casting records for this game. Each record is the shape
    returned by `get_casting()`: {slot, slotDisplayName, occupant, kind,
    displayName, source}. Empty slots are excluded unless `include_empty=True`."""
    rows: List[Dict[str, Any]] = []
    for info in get_casting(game_key).values():
        if not include_empty and info.get("kind") == "empty":
            continue
        rows.append(info)
    return rows


def casting_for_occupant(game_key: str, sid: str) -> Dict[str, Any]:
    """Return the casting record whose occupant is `sid`. Raises if not cast."""
    for info in get_casting(game_key).values():
        if info.get("kind") != "empty" and info.get("occupant") == sid:
            return info
    raise ValueError(
        f"No casting found for sid: {sid!r}. Set a defaultBot in a slot, "
        f"or use set_casting() to bind them."
    )


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
        prompt_assemble, load_bot, load_appearance,
        load_slot_system_prompt,
    )
    pos = casting_location(game_key, bot_sid) or ""
    prompt = prompt_assemble(game_key, bot_sid, caller)

    # Load individual pieces only for the sources breakdown
    bot_md = load_bot(bot_sid)
    appearance_md = load_appearance(bot_sid)
    job_md = load_slot_system_prompt(slot)
    flavor_md = load_casting_prompt(slot, bot_sid)

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
            "setting":    f"composed from location chain rooted at {pos!r}" if pos else "(no location)",
        },
    }


@visible
async def casting_list(game_key: str) -> List[Dict[str, Any]]:
    """Show the cast list for this game — one row per slot. The lobby view:
    who's in each seat, are they AI or human, where they currently stand,
    is this the slot's default or an explicit override. Empty slots show kind="empty".
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
            "entryLocation": info.get("entryLocation", ""),
            "location": info.get("location", ""),
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
