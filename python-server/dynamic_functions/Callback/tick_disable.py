import atlantis
import logging

logger = logging.getLogger("mcp_server")

_TICK_KEY = "tick_task"


@visible
async def tick_disable() -> str:
    """Stop the local tick loop if one is running."""
    task = atlantis.server_shared.get(_TICK_KEY)
    if task is None:
        return "tick not running"
    if task.done():
        atlantis.server_shared.remove(_TICK_KEY)
        return "tick already stopped"
    task.cancel()
    atlantis.server_shared.remove(_TICK_KEY)
    logger.info("tick_disable: tick loop cancelled")
    return "tick disabled"
