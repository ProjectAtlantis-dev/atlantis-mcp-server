import atlantis
import logging

logger = logging.getLogger("dynamic_function")


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

    await atlantis.client_log("enter running")

    return "Setup entered"


@visible
@exit
async def on_exit():
    """
    Runs when a terminal cd leaves Setup
    """
    logger.info(f"Executing Setup on_exit...")

    await atlantis.client_log("exit running")

    return "Setup exited"

