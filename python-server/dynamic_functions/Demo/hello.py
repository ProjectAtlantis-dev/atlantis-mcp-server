import atlantis
import logging

logger = logging.getLogger("dynamic_function")


# this is a python comment

# this is a command
# % pwd

@visible
async def hello() -> None:
    """Gives a personalized greeting to the current caller."""
    caller = atlantis.get_caller() or "stranger"
    await atlantis.client_log(f"Hello, {caller}!")
