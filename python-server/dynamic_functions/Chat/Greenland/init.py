"""Greenland's setup flow: pick a scene, fill the roster, choose a camera, start."""

import atlantis

from ..roster import STATE_AVAILABLE
from ..runner import _camera_edit, _warn_unfilled_roster, roster_edit, roster_exists, start_owned_stopped_game
from ..scene import _scene_pick_dialog


async def game_init(game_name: str, game_key: str) -> None:
    if not roster_exists(game_name, game_key):
        scene = await _scene_pick_dialog(game_name)
        if not scene:
            raise RuntimeError("Scene selection cancelled")
        await atlantis.client_command(f"@roster_create {scene}")
        await atlantis.client_log(f"game scene: {scene!r}")

    if not await roster_edit(game_name):
        raise RuntimeError("Roster selection cancelled")
    roster = await atlantis.client_command("@roster_list")
    if roster and all(row.get("state") == STATE_AVAILABLE for row in roster):
        await _warn_unfilled_roster()

    if not await _camera_edit(roster=roster, game_name=game_name, game_key=game_key):
        raise RuntimeError("Camera selection cancelled")

    await start_owned_stopped_game(game_name, game_key)
