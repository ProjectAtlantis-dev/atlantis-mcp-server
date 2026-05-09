"""Home app tools"""

import atlantis
from typing import List, Dict, Any

from dynamic_functions.Home.chat_common import logger, get_base_tools


@visible
async def index():
    """Game logic engine"""
    pass


@visible
async def show_tools() -> List[Dict[str, Any]]:
    """Show the runtime tool inventory"""
    tools, lookup = get_base_tools()
    simple: List[Dict[str, Any]] = []
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "any")
            if isinstance(ptype, list):
                ptype = ",".join(ptype)
            parts.append(f"{pname}:{ptype}")
        sig = ", ".join(parts)
        simple.append({
            "name": f"{fn['name']} ({sig})",
            "description": fn.get("description", ""),
        })
    logger.info(f"show_tools: {len(simple)} tools")
    return simple

import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def foo(x):
    """
    Adds 10 to x and returns the result.
    """
    logger.info(f"Executing foo with x={x}")

    await atlantis.client_log(f"foo running with x={x}")

    return x + 10


@visible
async def bar(x, y):
    """
    Returns x + y.
    """
    logger.info(f"Executing bar with x={x}, y={y}")

    await atlantis.client_log(f"bar running with x={x}, y={y}")

    return x + y

