import atlantis
import logging


logger = logging.getLogger("dynamic_function")


@visible
async def index():
    """Terrain viewer server tools."""
    logger.info("Opening Terrain server tools")
    await atlantis.client_log("Terrain server tools opened")
    return None
