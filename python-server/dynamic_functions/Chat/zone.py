"""Entering and leaving a zone: an app whose Robot/ folder holds the @enter and
@exit hooks for the Chat game named after the app folder.

Each Robot's hooks call these. They read the app from the hook's own call
context, so they must be awaited from inside the hook, not run as tools.
"""

import json
import logging
import posixpath
from pathlib import Path

import atlantis

from .roster import roster_leave

logger = logging.getLogger("dynamic_function")


def _caller_terminal() -> str:
    """The terminal that called the hook. /chat commands name it: a command sent
    into the terminal itself would pass for the user's own input."""
    shell = atlantis.get_caller_shell_path()
    if not shell:
        raise RuntimeError("No caller shell; zone hooks must run from the user's terminal")
    return shell


def _app_folder() -> str:
    """The app folder the calling Robot sits in, as a /user/remote/... path"""
    script_folder = atlantis.get_script_folder()
    if not script_folder:
        raise RuntimeError("No script folder; zone hooks must run as a dynamic function")
    if posixpath.basename(script_folder) != "Robot":
        raise RuntimeError(f"Zone hooks must live in an app's Robot folder, not {script_folder}")
    return posixpath.dirname(script_folder)


async def zone_enter(background: Path) -> str:
    """Join the chat for the calling Robot's app: show its background, join the
    game named after the app folder and bind the caller's terminal to the
    app folder's chat shell (upserted by Node, keyed by folder)."""
    app_folder = _app_folder()
    app_name = posixpath.basename(app_folder)
    logger.info(f"Entering zone {app_name}...")

    await atlantis.set_background(str(background), vertical_align="center")
    # Preflight and chat read the game name from the cursor
    await atlantis.client_command("/cursor join", {"game_name": app_name})
    await atlantis.client_command(f"/chat bind {json.dumps(app_folder)} {json.dumps(_caller_terminal())}")
    return f"{app_name} entered"


async def zone_exit() -> str:
    """Leave the calling Robot's chat: walk out of the room while the chat is
    still bound, clear the background and cursor, and detach the caller's
    terminal from the chat shell, which stays."""
    app_name = posixpath.basename(_app_folder())
    logger.info(f"Exiting zone {app_name}...")

    await roster_leave(app_name)
    await atlantis.clear_background()
    await atlantis.client_command("/cursor clear")
    await atlantis.client_command(f"/chat unbind {json.dumps(_caller_terminal())}")
    return f"{app_name} exited"
