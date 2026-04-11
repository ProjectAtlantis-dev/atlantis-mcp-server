import atlantis
import logging
import os

from dynamic_functions.Computer.query import _connect

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

        # Ask the Computer if we know this guest
        conn = _connect()
        guest = conn.execute("SELECT * FROM guests WHERE username = ?", (user_id,)).fetchone()
        conn.close()

        if guest:
            player_location = guest["location"] or "Lobby"
            logger.info(f"Known guest {user_id}: location={player_location}")
        else:
            player_location = "Lobby"
            logger.info(f"Unknown guest {user_id}: routing to Lobby for check-in")

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
