import atlantis
import logging

from dynamic_functions.Callback.tick_set import set_tick_enabled

logger = logging.getLogger("mcp_server")

_LOOP_KEY = "tick_loop"


@visible
async def tick_disable() -> str:
    """Disable ticking and cancel the global tick loop if it is running.

    Per-game removal should go through game_deactivate(); this is the global
    enable/disable switch for ticking itself.
    """
    set_tick_enabled(False)
    loop_task = atlantis.server_shared.get(_LOOP_KEY)
    if loop_task is None or loop_task.done():
        return "tick disabled; loop already stopped"

    loop_task.cancel()
    atlantis.server_shared.remove(_LOOP_KEY)
    logger.info("tick_disable: tick loop cancelled")
    return "tick disabled; loop cancelled"
