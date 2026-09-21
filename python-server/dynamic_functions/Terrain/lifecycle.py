"""Combined lifecycle controls for Terrain services."""

from dynamic_functions.Terrain.Asset import database as asset
from dynamic_functions.Terrain.Database import database
from dynamic_functions.Terrain.Server import server


@visible
async def terrain_start(host: str = "127.0.0.1", port: int = 5180) :
    """Start the terrain database, asset catalog, and viewer server in order."""
    await database.start()
    asset_status = await asset.start()
    server_status = await server.start(host=host, port=port)


@visible
async def terrain_stop() :
    """Stop the viewer server before closing the asset and terrain databases."""
    server_status = await server.stop()
    if not server_status["stopped"] and not server_status["alreadyStopped"]:
        raise RuntimeError("Terrain viewer server did not stop; databases remain open")
    asset_status = await asset.stop()
    await database.stop()
