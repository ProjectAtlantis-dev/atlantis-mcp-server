"""Default game_callback — delegates to the AtlasLobby scenario."""

from dynamic_functions.Game.Content.AtlasLobby.game_callback import game_callback as _atlas_callback


@game
async def game_callback():
    """Default game callback — routes to AtlasLobby game_callback."""
    await _atlas_callback()
