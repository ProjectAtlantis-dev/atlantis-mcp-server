import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@keyword("foo")
@public
async def foo(x: int, y: int) -> int:
    """Add two integers"""
    return x + y
