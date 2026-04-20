"""Default game_callback — delegates to the FlowCentralLobby scenario."""

from dynamic_functions.Games.FlowCentralLobby.game_callback import game_callback as _flowcentral_callback


@game
async def game_callback():
    """Default game callback handler for this remote"""

