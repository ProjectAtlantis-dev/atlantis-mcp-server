"""Setup's setup flow: one scene, one owner, one camera — nothing to ask."""

import atlantis

from ..camera import camera_follow
from ..chat_callback import _mark_identity_known
from ..roster import _load_game_roster, bind_caller_to_slot
from ..runner import roster_exists, start_owned_stopped_game

SCENE = "setup"
OWNER_SLOT = "Owner"


async def game_init(game_name: str, game_key: str) -> None:
    if not roster_exists(game_name, game_key):
        await atlantis.client_command(f"@roster_create {SCENE}")

    # The owner plays themselves, not a character: bind them under their
    # Atlantis username instead of asking for a name.
    caller = atlantis.get_caller()
    if not caller:
        raise RuntimeError("No caller identity in this call context")
    await bind_caller_to_slot(game_name, game_key, OWNER_SLOT, display_name=caller)

    # Customer care is not a front desk: the bots know who they are helping
    # from the start instead of learning the name through remember_visitor.
    # Before the start, so a bot joining on start already greets them by name.
    roster = _load_game_roster(game_name, game_key)
    owner = next(row for row in roster if row["key"] == OWNER_SLOT)
    for row in roster:
        if row.get("ai") is True:
            _mark_identity_known(
                game_name,
                game_key,
                bot_sid=row["bot_sid"],
                human_sid=owner["sid"],
                display_name=owner["displayName"],
            )

    await camera_follow(game_name, game_key, OWNER_SLOT)
    await start_owned_stopped_game(game_name, game_key)
