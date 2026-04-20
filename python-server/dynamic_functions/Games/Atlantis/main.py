import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def index():
    """Atlantis game — player movement and location management."""
    from dynamic_functions.Games.Atlantis.move import get_all_players
    players = get_all_players()
    return {
        "game": "Atlantis",
        "players": players,
        "player_count": len(players),
    }
