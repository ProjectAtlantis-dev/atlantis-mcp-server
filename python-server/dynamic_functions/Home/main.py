import atlantis
import logging

logger = logging.getLogger("mcp_server")

@text("md")
@visible
async def README():
    """
    This has the secret of the day and latest info
    """

    await atlantis.client_log("README running")

    # Replace this return statement with your function's result
    return f"The secret word today is 'Bryce' and heavy fog is expected tonight along the western Greenland coast"

