import json
import logging
import posixpath
from pathlib import Path

import atlantis

from dynamic_functions.Setup.main import (
    DATA_PATH,
    SETUP_STEPS,
    _caller,
    _check_steps,
    _now,
    _read_json,
    _write_json,
)

logger = logging.getLogger("dynamic_function")

# Robot/ holds the hooks and status that Chat drives for the app folder it sits
# in. That app folder's name is the Chat game name.
APP_NAME = Path(__file__).absolute().parents[1].name
# Placeholder background: builder.jpg in the project root
SETUP_BG_PATH = Path(__file__).absolute().parents[4] / "builder.jpg"


@visible
@index
async def index():
    """Robot info for Astrid"""
    return "Setup robot"


@visible
@enter
async def on_enter():
    """
    Runs when entering chat
    """
    logger.info(f"Executing Setup on_enter...")

    await atlantis.set_background(str(SETUP_BG_PATH), vertical_align="center")

    # The app folder name is the Chat game name; preflight and chat read it from the cursor
    await atlantis.client_command("/cursor join", {"game_name": APP_NAME})

    # Point the calling terminal at the app folder's chat shell (upserted by Node, keyed by folder)
    script_folder = atlantis.get_script_folder()
    if not script_folder:
        raise RuntimeError("No script folder; on_enter must run as a dynamic function")
    app_folder = posixpath.dirname(script_folder)
    await atlantis.client_command("/chat bind " + json.dumps(app_folder), shell="caller")

    return "Setup entered"


@visible
@exit
async def on_exit():
    """
    Runs when leaving chat
    """
    logger.info(f"Executing Setup on_exit...")

    # Walk out of the room while this folder's chat is still bound
    from dynamic_functions.Chat.roster import roster_leave
    await roster_leave(APP_NAME)

    await atlantis.clear_background()

    # Drop the game cursor joined in on_enter
    await atlantis.client_command("/cursor clear")

    # Detach the calling terminal from the app folder's chat shell; the chat shell stays
    await atlantis.client_command("/chat unbind", shell="caller")

    return "Setup exited"


@visible
async def get_status():
    """
    Setup status for the current user: who they are, whether you have met
    them in this game, every setup step with whether it is done, and exactly what to
    do next. Chat runs this before every bot round, so it only reads.
    """
    logger.info(f"Executing get_status...")

    username = _caller()
    steps = await _check_steps(username)

    if username in _read_json(_game_visitors_path()):
        seen = f"You have already been helping {username} in this game."
    else:
        seen = f"You have not met {username}."

    current = next((number for number, step in enumerate(steps, 1) if not step["done"]), None)
    lines = [f"The user is {username}. {seen}", f"Setup has {len(SETUP_STEPS)} steps:"]
    for number, ((name, _, _), step) in enumerate(zip(SETUP_STEPS, steps), 1):
        if step["done"]:
            label = "done"
        elif number == current:
            label = "NEXT"
        elif step["waitsOn"]:
            label = f"waiting on step {step['waitsOn']}"
        else:
            label = "to do"
        lines.append(f"{number}. [{label}] {name}: {step['state']}")
    if current is None:
        lines.append("Next: setup is complete; tell the user they are ready to use Terrain.")
    else:
        lines.append(f"Next (step {current}): {steps[current - 1]['fix']}")
    return "\n".join(lines)


@visible
async def record_visit():
    """
    Record that the bot has been helping the current user in this game. Chat
    runs this once a bot's reply has gone out.
    """
    username = _caller()
    visitors_path = _game_visitors_path()
    visitors = _read_json(visitors_path)
    visitors[username] = {"lastSeen": _now()}
    _write_json(visitors_path, visitors)
    return f"Recorded {username}'s visit"


def _game_visitors_path() -> Path:
    """This game's record of who its bot has met: {username: {lastSeen}}.

    Bot memory lives with the game, so a new game starts with a bot that has
    met nobody. The game folder must already exist (Chat creates it on join).
    """
    game_uuid = atlantis.get_game_uuid()
    if not game_uuid:
        raise RuntimeError("No game_uuid; this must be called from the Setup chat")
    game_dir = DATA_PATH / "games" / game_uuid
    if not game_dir.is_dir():
        raise FileNotFoundError(f"No game folder for game {game_uuid!r} at {game_dir}")
    return game_dir / "visitors.json"
