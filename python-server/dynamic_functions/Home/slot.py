"""Slot tools

A *slot* is a playable unit in the game (the gamer-friendly term for what we
used to call a role). The lobby UI shows a roster of slots; each one is either
filled by an AI persona or claimed by a human. The slot's `role.md` is the job
description that whoever fills the slot is expected to follow.

This file is the renamed successor of `role.py`. For now it still reads from
`Game/Slots/<slot>/` and uses the legacy `system_prompt.md` filename for the
role text — the deeper model shift (defaultPersona, casting.json, etc.) lands
in a follow-up pass.
"""

import atlantis
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.common import home_path

logger = logging.getLogger("mcp_server")


def _slots_dir() -> str:
    return home_path("Game", "Slots")


def _load_slot_json(slot_name: str) -> dict:
    """Read a slot config"""
    sjson = os.path.join(_slots_dir(), slot_name, "config.json")
    if os.path.isfile(sjson):
        with open(sjson) as f:
            return json.load(f)
    return {}


def _slot_rows() -> List[Dict[str, Any]]:
    """Pure data: list available slots. No client side effects."""
    from dynamic_functions.Home.prompt_common import load_slot_system_prompt
    slots_dir = _slots_dir()
    slots: List[Dict[str, Any]] = []
    if not os.path.isdir(slots_dir):
        return slots
    for entry in sorted(os.listdir(slots_dir)):
        entry_dir = os.path.join(slots_dir, entry)
        if not os.path.isdir(entry_dir) or entry.startswith(".") or entry == "__pycache__":
            continue
        slot_data = _load_slot_json(entry)
        mtimes = []
        for sub_root, _dirs, sub_files in os.walk(entry_dir):
            for filename in sub_files:
                mtimes.append(os.path.getmtime(os.path.join(sub_root, filename)))
        updated = datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M') if mtimes else ''
        try:
            system_prompt = load_slot_system_prompt(entry)
        except FileNotFoundError:
            system_prompt = ""
        slots.append({
            "name": entry,
            "displayName": slot_data.get("displayName", entry),
            "defaultLocation": slot_data.get("defaultLocation", ""),
            "defaultPersona": slot_data.get("defaultPersona", ""),
            "purpose": slot_data.get("purpose", ""),
            "openToHumans": bool(slot_data.get("openToHumans", True)),
            "systemPrompt": system_prompt,
            "updated": updated,
        })
    return slots


@visible
async def slot_list(game_key: str = "") -> List[Dict[str, Any]]:
    """List slots (playable units). If `game_key` is given, include current
    casting for each slot (the lobby roster view)."""
    slots = _slot_rows()
    if game_key:
        from dynamic_functions.Home.casting import get_casting
        casting = get_casting(game_key)
        for row in slots:
            info = casting.get(row["name"], {})
            row["occupant"] = info.get("occupant", "")
            row["occupantDisplayName"] = info.get("displayName", "")
            row["occupantKind"] = info.get("kind", "empty")
            row["castingSource"] = info.get("source", "empty")
    await atlantis.client_data("Slots", slots, column_formatter={
        "systemPrompt": {"type": "markdown"},
    })
    return slots


@visible
async def slot_take(game_key: str, slot: str, displayName: str = "") -> Dict[str, Any]:
    """Claim a slot for the calling user (human). Overrides any AI default."""
    import atlantis as _a
    from dynamic_functions.Home.casting import set_casting, get_casting
    user_sid = _a.get_caller() or ""
    if not user_sid:
        raise RuntimeError("No caller identity available.")
    set_casting(game_key, slot, user_sid, kind="human", displayName=displayName or user_sid)
    return get_casting(game_key)[slot]


@visible
async def slot_release(game_key: str, slot: str) -> Dict[str, Any]:
    """Release a slot back to its AI default."""
    from dynamic_functions.Home.casting import clear_casting, get_casting
    clear_casting(game_key, slot)
    return get_casting(game_key).get(slot, {"slot": slot, "kind": "empty"})


def slot_default_location(slot: str) -> Optional[str]:
    """Return a slot-specific default location from config.json, if set."""
    location = _load_slot_json(slot).get("defaultLocation", "")
    return str(location).strip() or None


def _validate_slot(slot: str) -> None:
    """Validate a slot folder"""
    if not os.path.isdir(os.path.join(_slots_dir(), slot)):
        raise ValueError(f"Slot folder not found: {slot}")
