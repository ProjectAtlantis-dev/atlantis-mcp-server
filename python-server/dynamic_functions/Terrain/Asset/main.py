"""Folder entry point for Terrain asset catalog tools."""

import atlantis


@index
@visible
async def index() -> None:
    """Open the Terrain asset catalog tools."""
    await atlantis.client_log("Terrain asset catalog tools opened")
