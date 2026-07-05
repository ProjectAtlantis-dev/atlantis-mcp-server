"""Cancellation test probe: an interruptible async sleep."""

import asyncio
import logging

import atlantis

logger = logging.getLogger("dynamic_function")


@public
@visible
async def sleep(seconds: float = 30) -> str:
    """Sleep for the given number of seconds, ticking once per second.

    Exists to test the interrupt path: cancel lands on the asyncio.sleep
    await, CancelledError propagates, and the server reports the tool call
    cancelled. Logs make the abort point visible.
    """
    total = float(seconds)
    await atlantis.client_log(f"sleep: starting {total}s")
    elapsed = 0.0
    try:
        while elapsed < total:
            step = min(1.0, total - elapsed)
            await asyncio.sleep(step)
            elapsed += step
            logger.info(f"sleep: {elapsed:.0f}/{total:.0f}s")
    except asyncio.CancelledError:
        logger.warning(f"sleep: CANCELLED at {elapsed:.0f}/{total:.0f}s")
        raise
    return f"slept {total:.0f}s"
