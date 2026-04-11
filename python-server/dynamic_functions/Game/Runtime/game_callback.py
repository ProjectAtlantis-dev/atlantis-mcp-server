import atlantis
import logging
import os

from dynamic_functions.Game.Data.main import ensure_player_record

logger = logging.getLogger("mcp_server")

LOCATION_BACKGROUNDS = {
    "Lobby": os.path.join(os.path.dirname(__file__), "..", "Content", "Lobby", "builder.jpg"),
}


@game
async def game_callback():
    """Initializes a new chat session at the player's persisted location."""

    try:
        user_id = atlantis.get_caller()
        if not user_id:
            raise ValueError("Game started without a caller identity")
        if not atlantis.get_game_id():
            raise RuntimeError("game callback fired without a game_id in context — nodejs side must send game_id")

        logger.info(f"Game started for user: {user_id}")
        player_record, is_new_player = ensure_player_record(user_id)
        player_location = player_record["where"]
        logger.info(
            f"Game player state for {user_id}: where={player_location}, is_new={is_new_player}"
        )

        await atlantis.client_command("/silent on")

        # Wire up callbacks — everything in same dir as game
        await atlantis.client_command("/callback set chat chat_callback")
        await atlantis.client_command("/callback set session session_callback")

    finally:
        await atlantis.client_command("/silent off")

    image_path = LOCATION_BACKGROUNDS.get(player_location)
    if not image_path:
        raise RuntimeError(f"No background configured for game location: {player_location}")
    await atlantis.set_background(image_path)
