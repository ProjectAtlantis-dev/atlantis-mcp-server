"""Per-game roster tools."""

import atlantis
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .bot import bot_image_path, bot_roster_name, load_bot
from .bot_tool import _initialize_bot_tool_files
from .common import _read_json, _write_json
from .game import require_membership
from .location import _connects_to, _require_leaf, load_location
from dynamic_functions.Home.modal import modal_string
from .scene import _load_scene, _scene_name, _scene_names


# A slot nobody has taken is a role that is open, not a thing that is broken:
# "available" throughout, key and label alike. The key is not persisted — a
# row's state is derived from `ai` (None/True/False) — so it only ever travels
# between the browser, roster_set_slot, and these helpers.
KEY_AVAILABLE = "available"
KEY_AI = "ai"
KEY_HUMAN = "human"
STATE_KEYS = (KEY_AVAILABLE, KEY_AI, KEY_HUMAN)

STATE_AVAILABLE = "Available"
STATE_AI = "AI"
STATE_HUMAN = "Human"
STATE_LABELS = {KEY_AVAILABLE: STATE_AVAILABLE, KEY_AI: STATE_AI, KEY_HUMAN: STATE_HUMAN}
# Callers that hold a label and need the key must look it up here rather than
# lower-casing: that only works while every label is its key capitalized.
STATE_KEY_BY_LABEL = {label: key for key, label in STATE_LABELS.items()}
# A row's state is stored as `ai`: None = nobody, True = bot, False = person.
AI_BY_STATE_KEY = {KEY_AVAILABLE: None, KEY_AI: True, KEY_HUMAN: False}

KITTY_BOT_SID = "kitty"
SIGHTINGS_FILENAME = "sightings.json"


def _number_duplicate_display_names(rows: List[Dict[str, Any]]) -> None:
    used: set[str] = set()
    for row in rows:
        raw_display_name = row.get("displayName")
        if raw_display_name is None:
            continue
        display_name = str(raw_display_name).strip()
        if not display_name:
            continue

        if display_name not in used:
            used.add(display_name)
            continue

        count = 2
        numbered_name = f"{display_name} {count}"
        while numbered_name in used:
            count += 1
            numbered_name = f"{display_name} {count}"
        row["displayName"] = numbered_name
        used.add(numbered_name)


def _scene_slot_state_key(scene: str, index: int, slot: Dict[str, Any]) -> str:
    """The state a scene declares for one slot, defaulting to available.

    A scene describes the shape of a game — who is meant to be a bot and which
    role is meant for a person. Unknown values are rejected rather than ignored,
    so a typo in a scene file surfaces instead of silently producing a roster of
    empty slots.
    """
    raw = str(slot.get("state") or KEY_AVAILABLE).strip().lower()
    if raw not in STATE_KEYS:
        raise ValueError(
            f"Scene {scene!r} row {index} has unknown state {raw!r}; "
            f"expected one of: {', '.join(STATE_KEYS)}"
        )
    return raw


def _scene_roster_rows(scene: str) -> List[Dict[str, Any]]:
    """Convert a static scene into initial per-game roster rows."""
    rows: List[Dict[str, Any]] = []
    for index, slot in enumerate(_load_scene(scene)):
        if not isinstance(slot, dict):
            raise ValueError(f"Scene {scene!r} row {index} must be an object")

        key = str(slot.get("key", "")).strip()
        bot_sid = str(slot.get("bot_sid", "")).strip()
        if not key:
            raise ValueError(f"Scene {scene!r} row {index} is missing key")
        if not bot_sid:
            raise ValueError(f"Scene {scene!r} row {index} is missing bot_sid")

        load_bot(bot_sid)
        rows.append({
            "key": key,
            "bot_sid": bot_sid,
            "ai": AI_BY_STATE_KEY[_scene_slot_state_key(scene, index, slot)],
            "displayName": None,
            "location": None,
            "spawned_at": None,
            "session_key": None,
            "sid": None,
            "user_game_id": None,
            "bound_at": None,
        })
    return rows


def _apply_scene_human_defaults(rows: List[Dict[str, Any]]) -> None:
    """Name the scene's human slots after their role, and hand over the first.

    The player is stepping into a character, so the slot keeps that character's
    name — a human playing the `chad` slot is "Chad", not the caller's sid. Only
    the creator is here when the roster is built, so they take the first such
    slot outright; any further human slots stay unclaimed under the same
    role-derived name until someone binds and renames them.
    """
    claimed = False
    for row in rows:
        if row.get("ai") is not False:
            continue
        name = bot_roster_name(str(row.get("bot_sid") or "").strip())
        if claimed:
            row["displayName"] = name
            continue
        _set_roster_slot_human(row, name)
        claimed = True


def _load_game_roster(game_key: str) -> List[Dict[str, Any]]:
    """Load a game's finalized roster.json, requiring it to already exist."""
    data_dir = require_membership(game_key)
    roster_path = os.path.join(data_dir, "roster.json")
    if not os.path.isfile(roster_path):
        raise RuntimeError("must create scene roster for this game first")

    rows = _read_json(roster_path)
    if not isinstance(rows, list):
        raise ValueError(f"Game {game_key!r} roster.json must be a JSON array")
    return rows


def _find_roster_row(rows: List[Dict[str, Any]], sid_or_key: str) -> Dict[str, Any]:
    """Find one roster row by slot key, human sid, or AI bot sid."""
    needle = str(sid_or_key or "").strip()
    if not needle:
        raise ValueError("sid required")

    matches: List[Dict[str, Any]] = []
    for row in rows:
        candidates = [row.get("key")]
        if row.get("ai") is False:
            candidates.append(row.get("sid"))
        elif row.get("ai") is True:
            candidates.append(row.get("bot_sid"))
        if needle in candidates:
            matches.append(row)

    if not matches:
        raise ValueError(f"Unknown roster sid or slot: {needle!r}")
    if len(matches) > 1:
        keys = ", ".join(row["key"] for row in matches)
        raise ValueError(f"Roster id {needle!r} is ambiguous; use one of these slot keys: {keys}")
    return matches[0]


def _write_game_roster(game_key: str, rows: List[Dict[str, Any]]) -> None:
    data_dir = require_membership(game_key)
    _write_json(os.path.join(data_dir, "roster.json"), rows)


def _sightings_path(game_key: str) -> str:
    return os.path.join(require_membership(game_key), SIGHTINGS_FILENAME)


def _load_sightings(game_key: str) -> Dict[str, List[str]]:
    path = _sightings_path(game_key)
    raw = _read_json(path, {}) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Game {game_key!r} {SIGHTINGS_FILENAME} must be a JSON object")

    sightings: Dict[str, List[str]] = {}
    for bot_sid, human_sids in raw.items():
        if not isinstance(human_sids, list) or not all(isinstance(sid, str) for sid in human_sids):
            raise ValueError(
                f"Game {game_key!r} sighting record for {bot_sid!r} must be a list of human sids"
            )
        sightings[str(bot_sid)] = human_sids
    return sightings


async def _show_kitty_first_sighting(
    game_key: str,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Show Kitty's portrait once when the calling human first shares her location."""
    human_sid = str(atlantis.get_caller() or "").strip()
    if not human_sid:
        return False

    roster = rows if rows is not None else _load_game_roster(game_key)
    human_locations = {
        str(row.get("location") or "").strip()
        for row in roster
        if row.get("ai") is False and row.get("sid") == human_sid and row.get("location")
    }
    if not human_locations:
        return False

    kitty_row = next(
        (
            row for row in roster
            if row.get("ai") is True
            and row.get("bot_sid") == KITTY_BOT_SID
            and row.get("location") in human_locations
        ),
        None,
    )
    if kitty_row is None:
        return False

    sightings = _load_sightings(game_key)
    seen_by = sightings.setdefault(KITTY_BOT_SID, [])
    if human_sid in seen_by:
        return False

    image_path = bot_image_path(KITTY_BOT_SID)
    if not image_path:
        raise FileNotFoundError(f"Kitty portrait is not configured or is missing")

    location = str(kitty_row.get("location") or "")
    await atlantis.client_image(
        image_path,
        sid=KITTY_BOT_SID,
        location=location,
        shell="display",
    )

    seen_by.append(human_sid)
    _write_json(_sightings_path(game_key), sightings)
    return True


def _roster_rows() -> List[Dict[str, Any]]:
    """Pure data: scene roster definitions. No client side effects."""
    rows: List[Dict[str, Any]] = []

    for scene_name in _scene_names():
        for row in _scene_roster_rows(scene_name):
            out = dict(row)
            out["scene_name"] = scene_name
            rows.append(out)
    return rows


def _roster_row_state(row: Dict[str, Any]) -> str:
    state = str(row.get("state") or "").strip().lower()
    if state in STATE_LABELS:
        return STATE_LABELS[state]
    if row.get("ai") is True:
        return STATE_AI
    if row.get("ai") is False or row.get("session_key") or row.get("sid"):
        return STATE_HUMAN
    return STATE_AVAILABLE


def _roster_row_name(row: Dict[str, Any], state: str) -> str:
    if state == STATE_AVAILABLE:
        return ""
    if state == STATE_AI:
        bot_sid = str(row.get("bot_sid") or "").strip()
        if bot_sid:
            return str(row.get("displayName") or bot_roster_name(bot_sid) or bot_sid)
    return str(row.get("displayName") or row.get("sid") or row.get("bot_sid") or "")


def _roster_row_label(row: Dict[str, Any], available_label: str = "-") -> str:
    state = _roster_row_state(row)
    return _roster_row_name(row, state) or available_label


def _caller_roster_row(
    roster: List[Dict[str, Any]],
    session_key: str,
    caller_sid: Optional[str],
) -> Optional[Dict[str, Any]]:
    row = next((row for row in roster if row.get("session_key") == session_key), None)
    if row is not None:
        return row
    if caller_sid:
        return next(
            (
                row for row in roster
                if _roster_row_state(row) == STATE_HUMAN and row.get("sid") == caller_sid
            ),
            None,
        )
    return None


def _display_roster_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project live roster rows into the table order shown to users."""
    columns = [
        "key",
        "bot_sid",
        "state",
        "displayName",
        "sid",
        "location",
        "session_key",
        "bound_at",
        "spawned_at",
    ]
    out = []
    for row in rows:
        state = _roster_row_state(row)
        display_row = dict(row)
        display_row["state"] = state
        display_row["displayName"] = _roster_row_name(row, state)
        out.append({column: display_row.get(column, "") for column in columns})
    return out


@public
async def roster_list(game_key: str) -> List[Dict[str, Any]]:
    """Show this game's live roster.json, including any roster_bind changes."""
    rows = _load_game_roster(game_key)
    display_rows = _display_roster_rows(rows)
    await atlantis.client_data(f"{game_key} roster", display_rows)
    return display_rows


def _reset_roster_slot(target: Dict[str, Any]) -> None:
    target["displayName"] = None
    target["location"] = None
    target["spawned_at"] = None
    target["session_key"] = None
    target["sid"] = None
    target["user_game_id"] = None
    target["bound_at"] = None


def _set_roster_slot_available(target: Dict[str, Any]) -> None:
    _reset_roster_slot(target)
    target["ai"] = None


def _set_roster_slot_ai(target: Dict[str, Any], bot_sid: Optional[str] = None) -> None:
    bot_sid = str(bot_sid or "").strip()
    if bot_sid:
        load_bot(bot_sid)
    _reset_roster_slot(target)
    target["ai"] = True
    if bot_sid:
        target["bot_sid"] = bot_sid


def _set_roster_slot_human(target: Dict[str, Any], display_name: str) -> None:
    session_key = atlantis.get_session_key()
    if not session_key:
        raise RuntimeError("No session key in this call context")
    display_name = str(display_name or "").strip()
    if not display_name:
        raise ValueError("display_name required")
    _reset_roster_slot(target)
    target["session_key"] = session_key
    target["sid"] = atlantis.get_caller() or None
    target["user_game_id"] = atlantis.get_user_game_id()
    target["ai"] = False
    target["displayName"] = display_name
    target["bound_at"] = datetime.now().isoformat(timespec="seconds")


@public
async def roster_set_slot(
    game_key: str,
    slot_key: str,
    state: str,
    display_name: Optional[str] = None,
    bot_sid: Optional[str] = None,
) -> Dict[str, Any]:
    """Set an available roster slot to AI or Human."""
    slot_key = str(slot_key or "").strip()
    if not slot_key:
        raise ValueError("slot_key required")

    state_key = str(state or "").strip().lower()
    if state_key not in STATE_KEYS:
        raise ValueError("state must be one of: available, ai, human")

    rows = _load_game_roster(game_key)
    target = next((row for row in rows if row.get("key") == slot_key), None)
    if target is None:
        raise ValueError(f"Unknown roster slot: {slot_key!r}")

    moved_slots: List[tuple[Dict[str, Any], Optional[str]]] = [(target, target.get("location") or None)]
    if state_key == KEY_AVAILABLE:
        _set_roster_slot_available(target)
    elif state_key == KEY_AI:
        _set_roster_slot_ai(target, bot_sid)
    else:
        _set_roster_slot_human(target, str(display_name or ""))

    _write_game_roster(game_key, rows)
    _initialize_bot_tool_files(game_key, rows)
    await atlantis.client_log(f"roster_set_slot game_key: {game_key!r} slot_key: {slot_key!r} state: {state_key!r}")
    await atlantis.client_data(f"{game_key} roster slot", _display_roster_rows([target])[0])
    await atlantis.client_data(f"{game_key} roster", _display_roster_rows(rows))
    for moved_slot, previous_location in moved_slots:
        if previous_location:
            await _notify_roster_slot_moved(game_key, moved_slot, moved_slot.get("location") or None)
    return target


@public
async def roster_create(game_key: str, scene: str) -> List[Dict[str, Any]]:
    """Create Data/games/<game_key>/roster.json from a static scene file."""
    await atlantis.client_log(f"roster_create game_key: {game_key!r} scene: {scene!r}")
    data_dir = require_membership(game_key)
    scene_name = _scene_name(scene)
    rows = _scene_roster_rows(scene)
    _apply_scene_human_defaults(rows)
    _number_duplicate_display_names(rows)
    _write_json(os.path.join(data_dir, "roster.json"), rows)
    _initialize_bot_tool_files(game_key, rows)
    meta = _read_json(os.path.join(data_dir, "game.json")) or {}
    meta["roster_scene"] = scene_name
    meta["roster_created_at"] = datetime.now().isoformat(timespec="seconds")
    meta.pop("roster", None)
    _write_json(os.path.join(data_dir, "game.json"), meta)
    display_rows = _display_roster_rows(rows)
    await atlantis.client_data(f"{game_key} roster", display_rows)
    return display_rows


@public
async def roster_bind(game_key: str, slot_key: str) -> Dict[str, Any]:
    """Bind the caller's Atlantis session to a slot in this game's roster."""
    session_key = atlantis.get_session_key()
    if not session_key:
        raise RuntimeError("No session key in this call context")
    await atlantis.client_log(
        f"roster_bind game_key: {game_key!r} slot_key: {slot_key!r} session_key: {session_key!r}"
    )

    slot_key = str(slot_key or "").strip()
    if not slot_key:
        raise ValueError("slot_key required")

    rows = _load_game_roster(game_key)
    target = None
    for row in rows:
        if row.get("key") == slot_key:
            target = row

    if target is None:
        raise ValueError(f"Unknown roster slot: {slot_key!r}")

    existing_session = target.get("session_key")
    if existing_session and existing_session != session_key:
        raise RuntimeError(f"Slot {slot_key!r} is already bound")

    display_name = await modal_string(
        f"What name should people call {slot_key}?",
        title="Roster - Human",
        submit_label="Join",
    )
    if display_name is None:
        return {"cancelled": True, "key": slot_key}
    display_name = str(display_name or "").strip()
    if not display_name:
        raise ValueError("display_name required")

    _set_roster_slot_human(target, display_name)
    _write_game_roster(game_key, rows)
    await atlantis.client_log(f"Saved roster binding for {game_key!r} slot {slot_key!r}")
    await atlantis.client_data(f"{game_key} roster slot", _display_roster_rows([target])[0])
    await atlantis.client_data(f"{game_key} roster", _display_roster_rows(rows))
    return target


def _movement_log_label(reason: str) -> str:
    reason = str(reason or "move").strip() or "move"
    labels = {
        "spawn": "spawn",
        "teleport": "teleport",
        "move": "move",
    }
    return labels.get(reason, reason or "move")


async def _notify_roster_slot_moved(game_key: str, target: Dict[str, Any], location: Optional[str]) -> None:
    from .camera import camera_slot_moved

    await camera_slot_moved(game_key, target["key"], location)


async def _describe_roster_slot_entered(target: Dict[str, Any], location: str) -> None:
    display_name = _roster_row_name(target, _roster_row_state(target)) or target.get("key") or "Someone"
    await atlantis.client_description(
        f"{display_name} entered.",
        location=location,
        shell="display",
    )


async def _describe_roster_slot_exited(target: Dict[str, Any], location: str) -> None:
    display_name = _roster_row_name(target, _roster_row_state(target)) or target.get("key") or "Someone"
    await atlantis.client_description(
        f"{display_name} exited.",
        location=location,
        shell="display",
    )


def _require_adjacent_move(previous: str, location: str) -> None:
    if not previous:
        raise RuntimeError("Roster slot has no current location; use roster_spawn or roster_teleport first")
    if previous == location:
        return

    from_previous = {str(name or "").strip() for name in _connects_to(previous)}
    from_location = {str(name or "").strip() for name in _connects_to(location)}
    if location in from_previous or previous in from_location:
        return

    allowed = sorted(name for name in (from_previous | from_location) if name)
    suffix = f" Adjacent locations: {', '.join(allowed)}." if allowed else ""
    raise ValueError(f"Cannot move from {previous!r} to non-adjacent location {location!r}.{suffix}")


async def _roster_move(game_key: str, sid_or_slot: str, location: str, reason: str = "move") -> Dict[str, Any]:
    """Move a roster slot to a Location.

    `sid_or_slot` may be a roster slot key, a bound human sid, or an AI bot sid.
    If a bot sid appears more than once in the roster, use the slot key.
    """
    load_location(location)
    _require_leaf(location)

    rows = _load_game_roster(game_key)
    target = _find_roster_row(rows, sid_or_slot)
    previous = target.get("location") or ""
    movement_reason = str(reason or "move").strip() or "move"
    if movement_reason == "move":
        _require_adjacent_move(previous, location)
    target["location"] = location
    if not target.get("spawned_at") or movement_reason == "spawn":
        target["spawned_at"] = datetime.now().isoformat(timespec="seconds")

    _write_game_roster(game_key, rows)
    log_label = _movement_log_label(movement_reason)
    await atlantis.client_log(
        f"{log_label}: {target.get('displayName', sid_or_slot)} -> {location}"
        + (f" from {previous}" if previous and previous != location else "")
    )
    await atlantis.client_data(f"{game_key} roster slot", _display_roster_rows([target])[0])
    await atlantis.client_data(f"{game_key} roster", _display_roster_rows(rows))
    if previous and previous != location:
        await _describe_roster_slot_exited(target, previous)
    if previous != location:
        await _describe_roster_slot_entered(target, location)
    await _show_kitty_first_sighting(game_key, rows)
    await _notify_roster_slot_moved(game_key, target, location)
    return target


@public
async def roster_move(game_key: str, sid_or_slot: str, location: str) -> Dict[str, Any]:
    """Move a roster slot to an adjacent Location."""
    return await _roster_move(game_key, sid_or_slot, location, reason="move")


@public
async def roster_spawn(game_key: str, sid_or_slot: str, location: str) -> Dict[str, Any]:
    """Spawn a roster slot by moving it to a Location."""
    return await _roster_move(game_key, sid_or_slot, location, reason="spawn")


@public
async def roster_teleport(game_key: str, sid_or_slot: str, location: str) -> Dict[str, Any]:
    """Teleport a roster slot to a Location."""
    return await _roster_move(game_key, sid_or_slot, location, reason="teleport")


@public
async def roster_despawn(game_key: str, sid: str) -> Dict[str, Any]:
    """Remove a roster entry from its current Location."""
    rows = _load_game_roster(game_key)
    target = _find_roster_row(rows, sid)
    previous = target.get("location") or ""
    target["location"] = None
    target["spawned_at"] = None

    _write_game_roster(game_key, rows)
    await atlantis.client_log(
        f"despawn: {target.get('displayName', sid)}"
        + (f" from {previous}" if previous else "")
    )
    await atlantis.client_data(f"{game_key} roster slot", _display_roster_rows([target])[0])
    await atlantis.client_data(f"{game_key} roster", _display_roster_rows(rows))
    if previous:
        await _describe_roster_slot_exited(target, previous)
    await _notify_roster_slot_moved(game_key, target, None)
    return target
