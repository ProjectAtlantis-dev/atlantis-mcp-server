import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def coffee():
    """
    Provides latest directions to find coffee since they keep moving the break room.
    """
    logger.info("Executing coffee function...")

    await atlantis.client_log("coffee running")

    return (
        "Coffee is available midway down the hall, located just past the tunnel elevators"
    )
