import logging

logger = logging.getLogger("dynamic_function")

# Tools/ holds the bot's own tools for this zone: Chat loads every tool here
# into each bot round (see Chat/turn.py _robot_tools). Walkthrough progress is
# not one of them: the server sees it in the user's terminal (see
# Demo/Robot/main.py _walk), so the bot never reports it; it can only start it
# over when the user asks (reset_demo_walkthrough).


@visible
@index
async def index():
    """Taffy's tools for the Demo zone"""
    return "Demo robot tools"
