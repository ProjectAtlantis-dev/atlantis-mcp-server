import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def index():
    """
    FlowCentralLobby
    """
    logger.info("Executing FlowCentralLobby index")

    await atlantis.client_log("FlowCentralLobby index running")

    return "FlowCentralLobby index executed successfully."
