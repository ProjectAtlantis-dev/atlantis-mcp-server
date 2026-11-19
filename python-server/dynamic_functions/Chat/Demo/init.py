"""Demo's setup flow: one scene, one owner, one camera — nothing to ask."""

import atlantis

from ..camera import camera_follow
from ..roster import bind_caller_to_slot
from ..runner import roster_exists, start_owned_stopped_game

SCENE = "demo"
OWNER_SLOT = "Owner"
# Run before every bot turn (see turn._game_status)
STATUS_TOOL = "$Demo/get_status"


async def game_init(game_name: str, game_key: str) -> None:
    if not roster_exists(game_name, game_key):
        await atlantis.client_command(f"@roster_create {SCENE}")

    # The owner plays themselves, not a character: bind them under their
    # Atlantis username instead of asking for a name.
    caller = atlantis.get_caller()
    if not caller:
        raise RuntimeError("No caller identity in this call context")
    await bind_caller_to_slot(game_name, game_key, OWNER_SLOT, display_name=caller)

    # The owner starts unknown to every bot: they introduce themselves and the
    # bot learns them through remember_visitor, which is how the visitor is authed.

    # No camera, same as Setup.
    #await camera_follow(game_name, game_key, OWNER_SLOT)
    await start_owned_stopped_game(game_name, game_key)
