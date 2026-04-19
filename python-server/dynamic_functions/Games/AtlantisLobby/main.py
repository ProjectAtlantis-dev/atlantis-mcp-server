import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def index():
    """
    AtlantisLobby
    """
    logger.info("Executing AtlantisLobby index")

    await atlantis.client_log("AtlantisLobby index running")

    return "AtlantisLobby index executed successfully."
