import atlantis
import logging
import os

logger = logging.getLogger("mcp_server")


@game
async def game_callback():
    """Initializes a new chat session"""

    await atlantis.client_command("/silent on")

    user_id = atlantis.get_caller()
    logger.info(f"Game started for user: {user_id}")

    owner_id = atlantis.get_owner()
    #await atlantis.client_log(f"Owner ID: {owner_id}")  # TEMP

    #kittyPath = f"{owner_id}**Bot.Kitty.OpenRouterGLM**chat"
    #kittyPath = f"{owner_id}**Bot.Kitty.OpenRouterMinimax**chat"
    #await atlantis.client_command("/chat set " + kittyPath)
    # everything should be in same dir as game
    await atlantis.client_command("/callback set chat chat_callback")

    # Tick is managed locally by the MCP server (no remote callback).
    # Game registration is now automatic: DynamicFunctionManager auto-adds
    # this game to the server-wide active dict the moment this very call
    # entered the context, so we don't need to call game_activate() here.
    # We just guard the contract that nodejs must send game_id.
    if not atlantis.get_game_id():
        raise RuntimeError("game callback fired without a game_id in context — nodejs side must send game_id")

    await atlantis.client_command("/silent off")
    image_path = os.path.join(os.path.dirname(__file__), "..", "Content", "Lobby", "builder.jpg")
    await atlantis.set_background(image_path)


