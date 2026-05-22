import atlantis
import logging
from pathlib import Path

logger = logging.getLogger("mcp_server")

# % whoami


@text("md")
@visible
async def README():
    """Show MULTIX instructions"""

    await atlantis.client_log("README running")

    md_path = Path(__file__).parent / "MULTIX.md"
    return md_path.read_text()


@text("md")
@visible
async def README_GAME():
    """Show the game diagram docs"""

    await atlantis.client_log("README_GAME running")

    md_path = Path(__file__).parent / "GAME.md"
    return md_path.read_text()
