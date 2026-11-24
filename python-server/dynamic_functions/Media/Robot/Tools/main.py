import logging

logger = logging.getLogger("dynamic_function")

# Tools/ holds the tools a bot uses to set Media up on this machine. Setup's
# installer Robot names this folder, so Chat loads every tool here into each
# of Astrid's rounds (see Setup/Robot/main.py tool_folders).


@visible
@index
async def index():
    """Media setup tools"""
    return "Media robot tools"
