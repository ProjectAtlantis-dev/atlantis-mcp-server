import atlantis
import logging

logger = logging.getLogger("mcp_server")
_LOOP_KEY = "tick_loop"


@visible
async def index():
    """
    Chatroom runtime utils
    """
    logger.info(f"Executing placeholder function: index...")

    await atlantis.client_log("index running")

    # Replace this return statement with your function's result
    return f"Placeholder function 'index' executed successfully."
async def list() -> list[dict]:
    """Return the currently active games registered server-side.

    Each entry: {"game_id": str, "caller": str}. Useful for debugging the
    auto-registration / tick fan-out path.
    """
    games = atlantis.get_active_games()
    return [
        {"game_id": gid, "caller": entry.get("caller", "")}
        for gid, entry in games.items()
    ]

async def deactivate():
    """Remove the current game from the server-wide active game list.

    If that was the last active game, cancel the tick loop immediately rather
    than waiting for it to notice on its next iteration.
    """
    game_id = atlantis.get_game_id()
    if not game_id:
        raise RuntimeError("deactivate() fired without a game_id in context")

    games = atlantis.get_active_games()
    if game_id not in games:
        return f"game {game_id} was not active"

    atlantis.deactivate_game(game_id)
    logger.info(f"deactivate: game={game_id} removed (remaining active: {len(games)})")

    if not games:
        loop_task = atlantis.server_shared.get(_LOOP_KEY)
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
        atlantis.server_shared.remove(_LOOP_KEY)
        logger.info("deactivate: no active games, tick loop cancelled")

    return f"game {game_id} deactivated (remaining active: {len(games)})"
