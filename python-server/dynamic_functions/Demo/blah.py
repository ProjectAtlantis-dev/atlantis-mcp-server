import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@public
async def blah() -> None:
    """Always raise, to demo how a tool error looks"""
    raise ValueError("you suck bawlz")
