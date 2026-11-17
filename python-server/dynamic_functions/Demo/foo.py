import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@public
async def foo(x: int, y: int) -> int:
    """Add two integers"""
    return x + y
