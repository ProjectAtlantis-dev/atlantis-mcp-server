import json
import logging
import posixpath
from pathlib import Path

import atlantis

from dynamic_functions.Demo.main import (
    DEMO_STEPS,
    _caller,
    _check_steps,
    _game_path,
    _now,
    _read_json,
    _write_json,
)

logger = logging.getLogger("dynamic_function")

# Robot/ holds the hooks and status that Chat drives for the app folder it sits
# in. That app folder's name is the Chat game name.
APP_NAME = Path(__file__).absolute().parents[1].name
# In each game's folder: who this game's bot has met, {username: {lastSeen}}.
# Bot memory lives with the game, so a new game starts with a bot that has met nobody.
VISITORS_FILE = "visitors.json"


@visible
@index
async def index():
    """Demo's robot: the enter/exit hooks and the status Chat reads every bot turn"""
    return "Demo robot"


@visible
@enter
async def on_enter():
    """
    Runs when entering chat
    """
    logger.info(f"Executing Demo on_enter...")

    # The app folder name is the Chat game name; preflight and chat read it from the cursor
    await atlantis.client_command("/cursor join", {"game_name": APP_NAME})

    # Point the calling terminal at the app folder's chat shell (upserted by Node, keyed by folder)
    app_folder = posixpath.dirname(atlantis.get_script_folder())
    await atlantis.client_command("/chat bind " + json.dumps(app_folder), shell="caller")

    return "Demo entered"


@visible
@exit
async def on_exit():
    """
    Runs when leaving chat
    """
    logger.info(f"Executing Demo on_exit...")

    # Walk out of the room while this folder's chat is still bound
    from dynamic_functions.Chat.roster import roster_leave
    await roster_leave(APP_NAME)

    # Drop the game cursor joined in on_enter
    await atlantis.client_command("/cursor clear")

    # Detach the calling terminal from the app folder's chat shell; the chat shell stays
    await atlantis.client_command("/chat unbind", shell="caller")

    return "Demo exited"


@visible
async def get_status():
    """
    Demo status for the current user: who they are, whether you have met them
    in this game, every walkthrough step with whether it is done, and exactly
    what to do next. Chat runs this before every bot round, so it only reads.
    """
    logger.info(f"Executing get_status...")

    username = _caller()
    steps = _check_steps(username)

    if username in _read_json(_game_path(VISITORS_FILE)):
        seen = f"You have already met {username} in this game; carry on from where you are."
    else:
        seen = f"You have not met {username}. Introduce yourself, then start the walkthrough."

    current = next((number for number, step in enumerate(steps, 1) if not step["done"]), None)
    lines = [f"The user is {username}. {seen}", f"The Demo walkthrough has {len(DEMO_STEPS)} steps:"]
    for number, ((name, _, _), step) in enumerate(zip(DEMO_STEPS, steps), 1):
        if step["done"]:
            label = "done"
        elif number == current:
            label = "NEXT"
        else:
            label = f"waiting on step {step['waitsOn']}"
        lines.append(f"{number}. [{label}] {name}: {step['state']}")
    if current is None:
        lines.append("Next: the walkthrough is complete; offer to show them any other Demo tool.")
    else:
        lines.append(f"Next (step {current}): {steps[current - 1]['fix']}")
    lines.append("If the user would rather explore on their own, search \"list demo tools\" and "
                 "call list_demo_tools. To run one of those tools, search its name, then call it.")
    return "\n".join(lines)


@visible
async def record_visit():
    """
    Record that the bot has met the current user in this game. Chat runs this
    once a bot's reply has gone out.
    """
    username = _caller()
    visitors_path = _game_path(VISITORS_FILE)
    visitors = _read_json(visitors_path)
    visitors[username] = {"lastSeen": _now()}
    _write_json(visitors_path, visitors)
    return f"Recorded {username}'s visit"

