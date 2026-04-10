import atlantis
import logging

logger = logging.getLogger("mcp_server")


@session
async def session_callback():
    """Fires on session reconnect — ensures the game stays in the active set
    so any registered tick continues to fire."""
    game_id = atlantis.get_game_id()
    caller = atlantis.get_caller() or "unknown"
    # NOTE: ensure_active_game() is already called by DynamicFunctionManager
    # for any tool call carrying a game_id, so we don't need to call it here.
    logger.info(f"🔄 Session reconnect: game={game_id} caller={caller}")
