import logging

logger = logging.getLogger("mcp_server")


@visible
async def coffee():
    """
    Provides current coffee directions in the FlowCentral lobby.
    """
    logger.info("FlowCentralLobby coffee called")

    return "Coffee is available near the FlowCentral lobby workspace."
