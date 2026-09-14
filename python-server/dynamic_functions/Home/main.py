import atlantis
import logging
from pathlib import Path

logger = logging.getLogger("dynamic_function")

# % whoami


@index
@visible
async def index(session_key: str):
    """Docs n stuff"""
    pass


@text("md")
@visible
async def README():
    """Show MULTIX instructions"""

    await atlantis.client_log("README running")

    md_path = Path(__file__).parent / "MULTIX.md"
    return md_path.read_text()


@text("md")
@visible
async def README_LOBSTER():
    """Show Lobster MCP tool instructions."""

    await atlantis.client_log("README_LOBSTER running")

    md_path = Path(__file__).parent / "MULTIX.md"
    return md_path.read_text(encoding="utf-8")
