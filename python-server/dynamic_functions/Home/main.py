import atlantis
import logging
from pathlib import Path

logger = logging.getLogger("mcp_server")

@text("md")
@visible
async def README():
    """
    This has the secret of the day and latest info
    """

    await atlantis.client_log("README running")

    md_path = Path(__file__).parent / "MULTIX.md"
    return md_path.read_text()

