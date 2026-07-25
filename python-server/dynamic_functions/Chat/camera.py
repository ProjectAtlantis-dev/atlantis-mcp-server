"""Cameras — where each terminal is looking.

A *camera* is one terminal's viewpoint: it maps a terminal_key (the PK,
atlantis session_key narrowed to one shell — i.e. sessionKey + shell) to what
that terminal is currently watching.

The mapping is per game and lives in Data/games/<key>/cameras.json. The stored
file starts empty; bindings appear only as terminals are assigned to locations
or roster slots. The *view* (`_camera_rows`) resolves every target to a concrete
Location and outer joins against standable locations — see that function.
"""

import atlantis
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .common import (
    _read_json,
    _write_json,
)
from .game import require_game_dir, require_membership
from .location import (
    _leaf_location_keys,
    _load_location,
    _require_leaf,
    location_image_path,
)
from .roster import _load_game_roster


# ---------------------------------------------------------------------------
# Camera store — terminal_key -> camera target, per game
# ---------------------------------------------------------------------------

def _cameras_path(game_key: str) -> str:
    return os.path.join(require_game_dir(game_key), "cameras.json")


def _load_cameras(game_key: str) -> Dict[str, Dict[str, Any]]:
    return _read_json(_cameras_path(game_key), {}) or {}


def _save_cameras(game_key: str, cameras: Dict[str, Dict[str, Any]]) -> None:
    _write_json(_cameras_path(game_key), cameras)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _location_default_camera_align(location: str) -> str:
    loc = _load_location(location)
    if not loc:
        raise ValueError(f"Unknown location: {location}")
    align = str(loc.get("defaultCameraAlign", "") or "").strip()
    if not align:
        raise ValueError(f"Location {location!r} has no defaultCameraAlign set")
    return align


def _slot_location(game_key: str, slot_key: str) -> Optional[str]:
    slot_key = str(slot_key or "").strip()
    if not slot_key:
        raise ValueError("slot_key required")

    for row in _load_game_roster(game_key):
        if row.get("key") == slot_key:
            return row.get("location")
    raise ValueError(f"Unknown roster slot: {slot_key!r}")


def _resolve_camera_location(game_key: str, entry: Dict[str, Any]) -> Optional[str]:
    target_type = entry.get("target_type")
    if target_type == "location":
        return entry.get("location")
    if target_type == "slot":
        return _slot_location(game_key, entry["slot_key"])
    raise ValueError(f"Unknown camera target_type: {target_type!r}")


async def _paint_location(location: str) -> None:
    background = location_image_path(location)
    align = _location_default_camera_align(location)
    await atlantis.client_log(f"camera paint location={location!r} background={background!r} align={align!r}")
    await atlantis.set_background(background, vertical_align=align)


def _camera_rows(game_key: str) -> List[Dict[str, Any]]:
    """Pure data: outer join of standable locations against resolved cameras."""
    cameras = _load_cameras(game_key)
    rows: List[Dict[str, Any]] = []
    seen_locations: set[str] = set()
    for terminal_key, entry in cameras.items():
        location = _resolve_camera_location(game_key, entry)
        if not location:
            continue
        seen_locations.add(location)
        target_type = entry.get("target_type")
        rows.append({
            "terminal": terminal_key,
            "location": location,
            "roster_slot": entry.get("slot_key", "") if target_type == "slot" else "",
            "updated_at": entry.get("updated_at", ""),
        })

    for location in _leaf_location_keys():
        if location not in seen_locations:
            rows.append({"terminal": "", "location": location, "roster_slot": "", "updated_at": ""})

    return sorted(rows, key=lambda row: (row["location"], row["terminal"], row["roster_slot"]))


async def _render_cameras(game_key: str) -> List[Dict[str, Any]]:
    """Push the camera table to the client and return the rows."""
    rows = _camera_rows(game_key)
    await atlantis.client_data("Cameras", rows, {
        "updated_at": {
            "type": "when"
        }
    })
    return rows


@public
async def camera_list(game_key: str) -> List[Dict[str, Any]]:
    """Show which Location each terminal is watching in this game."""
    require_membership(game_key)
    return await _render_cameras(game_key)


@visible
async def camera_bind(game_key: str, location: str) -> Dict[str, str]:
    """Bind the calling terminal to a Location, establishing its camera.

    The terminal_key (sessionKey + shell) of the calling shell starts watching
    `location`, and the terminal's background is repainted to that location's
    image. This is the invariant: a bound terminal always shows the location it
    is watching. Re-binding the same terminal simply moves its camera — each
    terminal watches exactly one place. The location must be standable (a leaf)
    and have an image; containers, unknown names, and imageless leaves are
    rejected before anything is mutated.

    The location's `defaultCameraAlign` is used when painting the terminal.
    """
    require_membership(game_key)
    terminal_key = atlantis.get_terminal_key()
    if not terminal_key:
        raise RuntimeError("No terminal key in this call context")
    await atlantis.client_log(
        "camera_bind context "
        f"game_key={game_key!r} session_key={atlantis.get_session_key()!r} "
        f"terminal_key={terminal_key!r} location={location!r}"
    )
    loc = _load_location(location)
    if not loc:
        raise ValueError(f"Unknown location: {location}")
    _require_leaf(location)
    location_image_path(location)  # resolve + validate before mutating
    _location_default_camera_align(location)

    cameras = _load_cameras(game_key)
    cameras[terminal_key] = {
        "target_type": "location",
        "location": location,
        "updated_at": _now_iso(),
    }
    _save_cameras(game_key, cameras)
    await atlantis.client_log(
        f"camera_bind saved terminal_key={terminal_key!r} camera_keys={sorted(cameras.keys())!r}"
    )

    await _paint_location(location)
    await _render_cameras(game_key)
    return {"terminal": terminal_key, "target_type": "location", "location": location}


async def _camera_follow_slot(
    game_key: str,
    slot_key: str,
    *,
    replace_session_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    require_membership(game_key)
    terminal_key = atlantis.get_terminal_key()
    if not terminal_key:
        raise RuntimeError("No terminal key in this call context")
    await atlantis.client_log(
        "camera_follow context "
        f"game_key={game_key!r} session_key={atlantis.get_session_key()!r} "
        f"terminal_key={terminal_key!r} slot_key={slot_key!r}"
    )

    slot_key = str(slot_key).strip()
    location = _slot_location(game_key, slot_key)
    if location:
        _require_leaf(location)
        location_image_path(location)  # resolve + validate before mutating
        _location_default_camera_align(location)

    cameras = _load_cameras(game_key)
    changed = False
    session_keys = replace_session_keys or []
    if session_keys:
        for existing_terminal in list(cameras):
            if existing_terminal == terminal_key:
                continue
            if not any(
                existing_terminal == session_key
                or existing_terminal.startswith(f"{session_key}:")
                for session_key in session_keys
            ):
                continue
            del cameras[existing_terminal]
            changed = True

    existing = cameras.get(terminal_key) or {}
    if (
        existing.get("target_type") != "slot"
        or existing.get("slot_key") != slot_key
    ):
        cameras[terminal_key] = {
            "target_type": "slot",
            "slot_key": slot_key,
            "updated_at": _now_iso(),
        }
        changed = True

    if changed:
        _save_cameras(game_key, cameras)
        await atlantis.client_log(
            f"camera_follow saved terminal_key={terminal_key!r} camera_keys={sorted(cameras.keys())!r}"
        )

    await atlantis.client_log(
        f"camera_follow terminal={terminal_key!r} slot={slot_key!r} location={location!r}"
    )
    if location:
        await _paint_location(location)
        await atlantis.client_log(f"camera painted terminal={terminal_key!r} location={location!r}")
    else:
        await atlantis.client_log(
            f"camera_follow pending terminal={terminal_key!r} slot={slot_key!r}; slot has no location yet"
        )
    await _render_cameras(game_key)
    return {
        "terminal": terminal_key,
        "target_type": "slot",
        "slot_key": slot_key,
        "location": location or "",
    }


@visible
async def camera_follow(game_key: str, slot_key: str) -> Dict[str, Any]:
    """Bind the calling terminal to follow a roster slot.

    If the slot has not been placed yet, the follow target is still recorded;
    the terminal is painted once the slot has a current Location.
    """
    return await _camera_follow_slot(game_key, slot_key)


async def camera_slot_moved(game_key: str, slot_key: str, location: Optional[str]) -> List[str]:
    """Refresh camera table after a followed slot moves.

    Atlantis background painting is scoped to the calling terminal, so this
    records the resolved state and refreshes the camera view. A terminal that is
    following the moved slot will repaint on its next camera_follow/camera_bind
    call unless Atlantis exposes a target-terminal paint primitive.
    """
    moved: List[str] = []
    for terminal_key, entry in _load_cameras(game_key).items():
        if entry.get("target_type") != "slot":
            continue
        if entry.get("slot_key") == slot_key:
            moved.append(terminal_key)

    if location and atlantis.get_terminal_key() in moved:
        await _paint_location(location)

    if moved:
        await _render_cameras(game_key)
    return sorted(moved)
