import atlantis
import logging

logger = logging.getLogger("mcp_server")


@session
async def session_callback():
    """Fires on session reconnect — ensures the game stays in the active set
    so any registered tick continues to fire."""
    game_id = atlantis.get_game_id()
    logger.info(f"session_callback: game_id={game_id}")
