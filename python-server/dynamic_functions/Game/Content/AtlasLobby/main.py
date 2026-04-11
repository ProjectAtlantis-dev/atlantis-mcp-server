import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def index():
    """
    FlowCentral lobby
    """
    logger.info("Executing AtlasLobby index")

    await atlantis.client_log("AtlasLobby index running")

    return "AtlasLobby index executed successfully."
