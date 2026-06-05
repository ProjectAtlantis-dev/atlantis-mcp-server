import atlantis
import logging
from pathlib import Path

logger = logging.getLogger("mcp_server")

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

    return """# Atlantis Lobster MCP Tools

Lobster exposes a small local MCP surface that forwards work to the connected Atlantis cloud session.

Tools:
- `readme`: show the Multix help text.
- `command`: send an Atlantis command. If the command has no prefix, `/` is added automatically.
- `chat`: send a plain chat message.

Common command prefixes:
- `/`: Atlantis slash command.
- `@`: tool/function call.
- `~`: routed tool/function call.
"""

import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def chicago():
    """
    This is a placeholder function for 'chicago'
    """
    logger.info(f"Executing placeholder function: chicago...")

    await atlantis.client_log("chicago running")

    # Replace this return statement with your function's result
    return f"Placeholder function 'chicago' executed successfully."

