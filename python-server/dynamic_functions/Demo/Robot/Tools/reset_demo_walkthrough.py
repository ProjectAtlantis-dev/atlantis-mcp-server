import logging

import atlantis

logger = logging.getLogger("dynamic_function")


@visible
async def reset_demo_walkthrough() -> str:
    """Start the user's Demo walkthrough over from step 1. Call it only when they ask to."""
    logger.info("Executing reset_demo_walkthrough...")
    # Progress belongs to Demo/Robot, which keeps it; this is the bot's way in
    result = await atlantis.client_command("@$Demo/Robot/reset_walkthrough")
    if not isinstance(result, str):
        raise TypeError(f"@$Demo/Robot/reset_walkthrough returned {type(result).__name__}, expected a string")
    return result
