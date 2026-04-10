import atlantis
import logging
import os

from dynamic_functions.Game.Data.main import ensure_player_record, DEFAULT_LOCATION

logger = logging.getLogger("mcp_server")

LOCATION_BACKGROUNDS = {
    "Lobby": os.path.join(os.path.dirname(__file__), "..", "Content", "Lobby", "builder.jpg"),
}


@game
async def game_callback():
    """Initializes a new chat session"""



    try:
        user_id = atlantis.get_caller()
        if not user_id:
            raise ValueError("Game started without a caller identity")
        logger.info(f"Game started for user: {user_id}")

        player_record, is_new_player = ensure_player_record(user_id)
        player_location = player_record.get("where") or DEFAULT_LOCATION
        logger.info(
            f"Game player state for {user_id}: where={player_location}, is_new={is_new_player}"
        )

        owner_id = atlantis.get_owner()

        await atlantis.client_command("/silent on")

        # everything should be in same dir as game
        await atlantis.client_command("/callback set chat chat_callback")

        # Game registration is automatic: DynamicFunctionManager auto-adds
        # this game to the server-wide active dict the moment this very call
        # entered the context. We just guard the contract that nodejs must send game_id.
        if not atlantis.get_game_id():
            raise RuntimeError("game callback fired without a game_id in context — nodejs side must send game_id")

    finally:
        await atlantis.client_command("/silent off")

    image_path = LOCATION_BACKGROUNDS.get(
        player_location,
        LOCATION_BACKGROUNDS[DEFAULT_LOCATION],
    )
    await atlantis.set_background(image_path)
