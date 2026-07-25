import atlantis
from pathlib import Path

from .common import home_path


@text("md")
@visible
async def README():
    """Show atlantis-mcp-chat runtime notes."""

    await atlantis.client_log("README running")

    md_path = Path(home_path("README.md"))
    return md_path.read_text(encoding="utf-8")
