"""FlowCentralLobby scenario — assigns Atlas to the FlowCentral receptionist role."""

import atlantis
import logging
import os

from dynamic_functions.Data.main import ensure_player_record
from dynamic_functions.Data.todo import _write_store
from dynamic_functions.Game.Runtime.common import spawn_bot
from dynamic_functions.Game.Runtime.roles import get_role
from dynamic_functions.Game.Runtime.roster import assign_role

logger = logging.getLogger("mcp_server")

BACKGROUND = os.path.join(os.path.dirname(__file__), "builder.jpg")


@game
async def game_callback():
    """FlowCentralLobby scenario — sets up the FlowCentral receptionist roster entry."""

    try:
        user_id = atlantis.get_caller()
        if not user_id:
            raise ValueError("Game started without a caller identity")
        game_id = atlantis.get_game_id()
        if not game_id:
            raise RuntimeError("game callback fired without a game_id in context")

        logger.info(f"FlowCentralLobby game started for user: {user_id}")

        _, created_player = ensure_player_record(user_id, location="FlowCentralLobby")
        if created_player:
            logger.info(f"FlowCentralLobby first-time player folder created for user: {user_id}")

        _write_store([], user_id, game_id)

        # Assign the bot inside this game's private role data.
        roster_role = {**get_role("flowcentral_receptionist"), "bot": "atlas"}
        role = assign_role(game_id, user_sid=user_id, **roster_role)
        logger.info(f"Roster: {role['bot']} -> {role['title']}")

        await atlantis.client_command("/silent on")
        await atlantis.client_command("/callback set chat chat_callback")
        await atlantis.client_command("/callback set session session_callback")

    finally:
        await atlantis.client_command("/silent off")

    await atlantis.set_background(BACKGROUND)
    await spawn_bot(role["bot"])
