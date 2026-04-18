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


@visible
async def list_games() -> list[dict]:
    """List all currently active games with their game_id, caller, and tick info."""
    games = atlantis.get_active_games()
    return [
        {
            "game_id": gid,
            "caller": entry.get("caller", ""),
            "tick": str(entry["tick_callback"]) if entry.get("tick_callback") else "",
            "tick_busy": entry.get("tick_busy", False),
        }
        for gid, entry in games.items()
    ]
