import atlantis
import logging

logger = logging.getLogger("mcp_server")

_LOOP_KEY = "tick_loop"


@game
async def game_deactivate():
    """Remove the current game from the server-wide active game list.

    If that was the last active game, cancel the tick loop immediately rather
    than waiting for it to notice on its next iteration.
    """
    game_id = atlantis.get_game_id()
    if not game_id:
        raise RuntimeError("game_deactivate() fired without a game_id in context")

    games = atlantis.get_active_games()
    if game_id not in games:
        return f"game {game_id} was not active"

    atlantis.deactivate_game(game_id)
    logger.info(f"game_deactivate: game={game_id} removed (remaining active: {len(games)})")

    if not games:
        loop_task = atlantis.server_shared.get(_LOOP_KEY)
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
        atlantis.server_shared.remove(_LOOP_KEY)
        logger.info("game_deactivate: no active games, tick loop cancelled")

    return f"game {game_id} deactivated (remaining active: {len(games)})"
