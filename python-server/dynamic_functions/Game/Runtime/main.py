import atlantis
import logging

logger = logging.getLogger("mcp_server")


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

    Each entry: {"game_id": str, "caller": str, "has_tick": bool}.
    Useful for debugging the active game registry.
    """
    games = atlantis.get_active_games()
    return [
        {
            "game_id": gid,
            "caller": entry.get("caller", ""),
            "has_tick": entry.get("tick_callback") is not None,
            "tick_busy": entry.get("tick_busy", False),
        }
        for gid, entry in games.items()
    ]


async def deactivate():
    """Remove the current game from the server-wide active game list.

    Tick loop cleanup is handled automatically by atlantis.deactivate_game().
    """
    game_id = atlantis.get_game_id()
    if not game_id:
        raise RuntimeError("deactivate() fired without a game_id in context")

    games = atlantis.get_active_games()
    if game_id not in games:
        return f"game {game_id} was not active"

    atlantis.deactivate_game(game_id)
    remaining = len(atlantis.get_active_games())
    logger.info(f"deactivate: game={game_id} removed (remaining active: {remaining})")

    return f"game {game_id} deactivated (remaining active: {remaining})"
