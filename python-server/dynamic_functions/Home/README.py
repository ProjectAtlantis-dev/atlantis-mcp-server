import atlantis
import logging
from pathlib import Path

logger = logging.getLogger("mcp_server")

@text("md")
@visible
async def README():
    """
    This has MULTIX command line instructions
    """

    await atlantis.client_log("README running")

    md_path = Path(__file__).parent / "MULTIX.md"
    return md_path.read_text()


@text("md")
@visible
async def README_GAME():
    """
    Game entity relationship diagram
    """

    await atlantis.client_log("README_GAME running")

    md_path = Path(__file__).parent / "GAME.md"
    return md_path.read_text()

