import logging
from pathlib import Path

import atlantis

logger = logging.getLogger("dynamic_function")

# Placeholder background: builder.jpg in the project root
SETUP_BG_PATH = Path(__file__).absolute().parents[3] / "builder.jpg"


@visible
@index
async def index():
    """
    Folder for Setup
    """
    logger.info(f"Executing placeholder function: index...")

    await atlantis.client_log("index running")

    # Replace this return statement with your function's result
    return f"Placeholder function 'index' executed successfully."


@visible
@enter
async def on_enter():
    """
    Runs when a terminal cd enters Setup
    """
    logger.info(f"Executing Setup on_enter...")

    #await atlantis.client_log("enter running")
    await atlantis.set_background(str(SETUP_BG_PATH), vertical_align="center")

    # The app folder name is the Chat game name; preflight and chat read it from the cursor
    await atlantis.client_command("/cursor join", {"game_name": Path(__file__).parent.name})

    return "Setup entered"


@visible
@exit
async def on_exit():
    """
    Runs when a terminal cd leaves Setup
    """
    logger.info(f"Executing Setup on_exit...")

    #await atlantis.client_log("exit running")
    await atlantis.clear_background()

    # Drop the game cursor joined in on_enter
    await atlantis.client_command("/cursor clear")

    return "Setup exited"

