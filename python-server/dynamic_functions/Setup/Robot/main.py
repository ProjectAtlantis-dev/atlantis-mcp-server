import logging
from pathlib import Path

import atlantis

from dynamic_functions.Setup.main import check_steps, terrain_installed

logger = logging.getLogger("dynamic_function")

# Robot/ holds the hooks and status that Chat drives for the app folder it sits
# in. That app folder's name is the Chat game name. Chat itself keeps who the
# bot has met in each game and tells it how the user types (see Chat/turn.py,
# _robot_status).
#
# Setup is the package installer: it puts packages on this machine, and each
# package's own Robot sets that package up (status, checks, tools), keeping
# its progress globally, since setup describes this machine. A Robot cannot
# install itself, so the install steps stay in Setup/main.py.
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
    from dynamic_functions.Chat.zone import zone_enter
    return await zone_enter(SETUP_BG_PATH)


@visible
@exit
async def on_exit():
    """
    Runs when leaving chat
    """
    from dynamic_functions.Chat.zone import zone_exit
    return await zone_exit()


@visible
async def get_status(command_prefix: str, current_folder: str):
    """
    Setup status for the current user: every setup step with whether it is
    done, and exactly what to do next. Chat runs this before every bot round,
    passing the prefix a Multix command takes where they type and the folder
    they are in. It only reads; Chat tells the bot who the user is and how
    they type.
    """
    logger.info(f"Executing get_status...")
    if command_prefix not in ("", "/"):
        raise ValueError(f"command_prefix must be '' or '/', got {command_prefix!r}")

    steps = await check_steps()
    current = next((number for number, step in enumerate(steps, 1) if not step["done"]), None)
    lines = [f"Setup installs the packages; it has {len(steps)} steps:"]
    for number, step in enumerate(steps, 1):
        if step["done"]:
            label = "done"
        elif number == current:
            label = "NEXT"
        elif step["waitsOn"]:
            label = f"waiting on step {step['waitsOn']}"
        else:
            label = "to do"
        lines.append(f"{number}. [{label}] {step['name']}: {step['state']}")
    if current is None:
        lines.append("Setup's own steps are done.")
    else:
        lines.append(f"Next (step {current}): {steps[current - 1]['fix']}")

    # Each package sets itself up through its own Robot; Setup only installs
    # them, so it shows their status after its own
    for package, robot in (await _package_robots()).items():
        status = await atlantis.client_command(
            f"@{robot}/get_status", {"command_prefix": command_prefix, "current_folder": current_folder})
        if not isinstance(status, str):
            raise TypeError(f"@{robot}/get_status returned {type(status).__name__}, expected a string")
        lines += ["", f"## {package}", status]
    lines += ["", "Work on one Next at a time, Setup's own first, then each package in the order "
                  "shown. Once every section is done, tell the user they are ready to use Terrain."]
    return "\n".join(lines)


@visible
async def tool_folders() -> list[str]:
    """
    The package tool folders Chat loads into each of Astrid's rounds next to
    Setup/Robot/Tools: each package's Robot/Tools.
    """
    return [f"{robot}/Tools" for robot in (await _package_robots()).values()]


async def _package_robots() -> dict[str, str]:
    """Each installed package's Robot, by package name, in setup order.
    Media ships with this server; Terrain is here once step 2 cloned it."""
    robots = {"Media": "$Media/Robot"}
    if await terrain_installed():
        robots["Terrain"] = "$Terrain/Robot"
    return robots
