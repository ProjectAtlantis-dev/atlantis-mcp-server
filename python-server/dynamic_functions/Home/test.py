"""Home test tools"""

import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def foo(x: int) -> int:
    """
    Adds 10 to x and returns the result.
    """
    logger.info(f"Executing foo with x={x}")

    await atlantis.client_log(f"foo running with x={x}")

    return x + 10


@visible
async def bar(x: int, y: int) -> int:
    """
    Returns x + y.
    """
    logger.info(f"Executing bar with x={x}, y={y}")

    await atlantis.client_log(f"bar running with x={x}, y={y}")

    return x + y


@visible
async def priorTest(_prior):
    """Examine prior: log length if list, log keys if dict."""
    if isinstance(_prior, list):
        await atlantis.client_log(len(_prior))
    elif isinstance(_prior, dict):
        await atlantis.client_log(list(_prior.keys()))
    else:
        await atlantis.client_log(f"unrecognized prior type: {type(_prior).__name__}")
