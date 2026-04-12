"""KittyLobby scenario — reads the atlantis_receptionist role from the roster."""

import atlantis
import logging
import os

from dynamic_functions.Data.todo import _write_store
from dynamic_functions.Game.Runtime.common import spawn_bot
from dynamic_functions.Game.Runtime.roles import create_role
from dynamic_functions.Game.Runtime.roster import get_role, assign_bot

logger = logging.getLogger("mcp_server")

BACKGROUND = os.path.join(os.path.dirname(__file__), "builder.jpg")


@game
async def game_callback():
    """KittyLobby scenario — sets up the Atlantis receptionist role from the roster."""

    try:
        user_id = atlantis.get_caller()
        if not user_id:
            raise ValueError("Game started without a caller identity")
        game_id = atlantis.get_game_id()
        if not game_id:
            raise RuntimeError("game callback fired without a game_id in context")

        logger.info(f"KittyLobby game started for user: {user_id}")

        _write_store([], user_id)

        # Assign the bot and register the role for this game
        await assign_bot("atlantis_receptionist", "kitty")
        roster_role = get_role("atlantis_receptionist")
        role = create_role(game_id, **roster_role)
        logger.info(f"Roster: {role['bot']} -> {role['title']}")

        await atlantis.client_command("/silent on")
        await atlantis.client_command("/callback set chat chat_callback")
        await atlantis.client_command("/callback set session session_callback")

    finally:
        await atlantis.client_command("/silent off")

    await atlantis.set_background(BACKGROUND)
    await spawn_bot(role["bot"])
