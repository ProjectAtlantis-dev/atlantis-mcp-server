import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def tick_register(toolPath: str) -> str:
    """Register a tick callback for the current game.

    toolPath is a dotted path under dynamic_functions/ (e.g. 'Callback.tick_callback'),
    where the last segment is both the module name and the function to invoke
    each tick. Stored on the current game's active-games entry, so it lives as
    long as the game is active.
    """
    if not toolPath or not isinstance(toolPath, str):
        raise ValueError("tick_register requires a non-empty toolPath string")

    game_id = atlantis.get_game_id()
    if not game_id:
        raise RuntimeError("tick_register() called outside a game context")

    games = atlantis.get_active_games()
    entry = games.get(game_id)
    if entry is None:
        raise RuntimeError(f"game {game_id} is not in the active set")

    entry["tick"] = toolPath
    logger.info(f"tick_register: game={game_id} tick={toolPath}")
    return f"tick registered for game {game_id}: {toolPath}"
