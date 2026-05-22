"""Home app tools"""

import atlantis
from typing import List, Dict, Any

from dynamic_functions.Home.chat_common import logger


@visible
async def index(session_key: str):
    """Game logic engine"""
    pass


@visible
async def show_tools() -> List[Dict[str, Any]]:
    """Show the runtime tool inventory"""
    logger.info("show_tools: no pseudo-tools remain")
    return []

