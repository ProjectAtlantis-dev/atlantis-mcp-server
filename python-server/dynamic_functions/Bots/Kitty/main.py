import os
from DynamicFunctionManager import text

_DIR = os.path.dirname(__file__)


@visible
async def index():
    """Kitty the helpful catgirl"""
    pass


@visible
@text("md")
async def get_system_prompt():
    """Returns the system prompt for this bot."""
    with open(os.path.join(_DIR, "system_prompt.md"), "r") as f:
        return f.read()
