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
    # A description, not client_log: chat view hides log (system) events, so a log never shows in chat
    await atlantis.client_description(f"Hello, {caller}!")
