"""AtlantisLobby scenario — assigns Kitty to the Atlantis receptionist role."""

import atlantis
import logging
import os

from dynamic_functions.Data.main import ensure_player_record
from dynamic_functions.Data.todo import todo_write
from dynamic_functions.Home.Game.common import spawn_bot
from dynamic_functions.Home.Game.location import enter_location
from dynamic_functions.Home.Game.roles import get_role
from dynamic_functions.Home.Game.roster import assign_role

logger = logging.getLogger("mcp_server")

GAME_DIR = os.path.dirname(__file__)
BOTS_DIR = os.path.join(GAME_DIR, "..", "..", "Bots")


@game
async def game_callback():
    """AtlantisLobby scenario — sets up the Atlantis receptionist roster entry."""

    try:
        user_id = atlantis.get_caller()
        if not user_id:
            raise ValueError("Game started without a caller identity")
        game_id = atlantis.get_game_id()
        if not game_id:
            raise RuntimeError("game callback fired without a game_id in context")

        logger.info(f"AtlantisLobby game started for user: {user_id}")

        _, created_player = ensure_player_record(user_id, location="AtlantisLobby")
        if created_player:
            logger.info(f"AtlantisLobby first-time player folder created for user: {user_id}")

        todo_write(f"AtlantisLobby/{user_id}/{game_id}/greeting_todo", [])

        # Assign the bot inside this game's private role data.
        roster_role = {**get_role("atlantis_receptionist"), "bot": "kitty"}
        role = assign_role(game_id, user_sid=user_id, **roster_role)
        logger.info(f"Roster: {role['bot']} -> {role['title']}")

        await atlantis.client_command("/silent on")
        await atlantis.client_command("/callback set chat $Game/Runtime/chat_callback")
        await atlantis.client_command("/callback set session $Game/Runtime/session_callback")

    finally:
        await atlantis.client_command("/silent off")

    await enter_location("AtlantisLobby", GAME_DIR, game_id=game_id, user_sid=user_id)
    await spawn_bot(role["bot"], BOTS_DIR)
