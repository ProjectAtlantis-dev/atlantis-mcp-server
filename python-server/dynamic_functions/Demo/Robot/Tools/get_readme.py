import logging

import atlantis

logger = logging.getLogger("dynamic_function")


@visible
async def get_readme() -> str:
    """The full Multix guide. Users open it themselves with /cd ../Home, then @readme."""
    logger.info("Executing get_readme...")
    # The commands a user types for it, run in this tool's own shell. The
    # shell is not in terminal mode, so the tool call needs its @.
    await atlantis.client_command("/cd $Home")
    guide = await atlantis.client_command("@readme")
    if not isinstance(guide, str):
        raise TypeError(f"@readme returned {type(guide).__name__}, expected a string")
    return guide
