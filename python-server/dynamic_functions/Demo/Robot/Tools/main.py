import logging

import atlantis

from dynamic_functions.Demo.Robot.main import (
    DEMO_STEPS,
    DEMO_STEP_KEYS,
    PROGRESS_FILE,
    _caller,
    _check_steps,
    _game_path,
    _now,
    _read_json,
    _write_json,
)

logger = logging.getLogger("dynamic_function")

# Tools/ holds the bot's own tools for this zone: Chat loads every tool here
# into each bot round (see Chat/turn.py _zone_tools).


@visible
@index
async def index():
    """Taffy's tools for the Demo zone"""
    return "Demo robot tools"


@visible
async def confirm_demo_step(step: str):
    """
    Record that the user finished a step of the Demo walkthrough: "ls", "foo",
    "source", "edit", "media", "table", "hello", "editor", "background" or "command".
    Call it only after the user has actually said so.
    """
    logger.info(f"Executing confirm_demo_step({step!r})...")

    if step not in DEMO_STEP_KEYS:
        raise ValueError(f"step must be one of {', '.join(DEMO_STEP_KEYS)}, not {step!r}")
    username = _caller()
    steps = _check_steps(username)
    number = DEMO_STEP_KEYS.index(step) + 1
    if steps[number - 1]["waitsOn"]:
        waits_on = steps[number - 1]["waitsOn"]
        raise RuntimeError(f"Cannot confirm {step!r}: step {waits_on} ({DEMO_STEPS[waits_on - 1][0]}) comes first")
    if step == "edit":
        # The edit is the one step the server can see for itself
        result = await atlantis.client_command("@$Demo/foo", {"x": 3, "y": 4})
        if result != 107:
            raise RuntimeError(f"Cannot confirm 'edit': foo(3, 4) returns {result!r}, not 107")

    progress_path = _game_path(PROGRESS_FILE)
    progress = _read_json(progress_path)
    progress.setdefault(username, {})[step] = _now()
    _write_json(progress_path, progress)
    return f"Recorded that {username} finished {step!r}."
