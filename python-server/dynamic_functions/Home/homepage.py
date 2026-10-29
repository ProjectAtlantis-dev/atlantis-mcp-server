"""Homepage menu for the local Home app."""

import logging
import posixpath
from pathlib import Path

import atlantis
from utils import format_json_log

from .modal import modal_menu

logger = logging.getLogger("dynamic_function")



# % file_callback_auto


@public
async def file_callback_auto():
    """Turn on the file callback only when a file tool is visible."""

    file_tools = await atlantis.client_command("tool find file")
    logger.info(f"tool find file returned:\n{format_json_log(file_tools, colored=True)}")

    # `tool find` returns a list of tool rows; an empty list means none are visible.
    if not isinstance(file_tools, list):
        raise TypeError(f"tool find file returned {type(file_tools).__name__}, expected a list")

    if len(file_tools) == 0:
        logger.info("No file tool visible; leaving the file callback off")
        return None

    await atlantis.client_command("/callback set file auto")
    return None


# % first_menu


@public
async def first_menu():
    """Let the user choose where to go next."""

    script_folder = atlantis.get_script_folder()
    if not script_folder:
        raise RuntimeError("Cannot determine homepage script folder")

    cwd = await atlantis.client_command("pwd")
    logger.info(f"pwd returned:\n{format_json_log(cwd, colored=True)}")

    tree_entries = await atlantis.client_command("tree ../*/first_menu")
    logger.info(f"tree first_menu (from {cwd}) returned:\n{format_json_log(tree_entries, colored=True)}")

    # `first_menu` is a per-folder entry point. Find public entry points at the
    # root of sibling apps and skip this Home menu itself.
    apps = {}
    for entry in tree_entries:
        parts = entry["filename"].split("/")
        if len(parts) != 2 or "Public" not in entry["chatStatus"]:
            continue
        app_path = entry["searchTerm"].rsplit("/", 1)[0]
        if app_path != script_folder:
            apps[parts[0]] = {"id": f"app:{app_path}", "text": entry["description"]}

    # Discovered apps lead, followed by the demo folder and a clean exit.
    items = [apps[folder] for folder in sorted(apps)]
    items.append({"id": "explore_demo_folder", "text": "Explore demo folder"})
    items.append({"id": "do_nothing", "text": "Quit"})

    choice = await modal_menu(
        items,
        title="Home",
        heading="Where do you want to go?",
    )

    choice_id = str(choice["id"])
    if choice_id == "do_nothing":
        return None

    if choice_id.startswith("app:"):
        app_path = choice_id[4:]
        commands = [
            f"/cd {app_path}",
            "pwd",
            "first_menu",
        ]
        logger.info(f"launching app script for '{app_path}':\n{format_json_log(commands, colored=True)}")
        await atlantis.client_command("/script", {"commands": commands})
        return None

    if choice_id == "explore_demo_folder":
        commands = [
            f"/cd {script_folder}",
            "cd ..",
            "cd Demo",
        ]
        await atlantis.client_command("/script", {"commands": commands})

        img_path = Path(__file__).absolute().parents[3] / "sitting_coffee.png"
        await atlantis.client_image(
            str(img_path),
            content="Demo folder coming right up",
            max_width="25vw",
        )

        # Use script so ls runs in the Demo folder after the scripted cd.
        await atlantis.client_command("/script", {"commands": ["ls"]})

    return None


@public
@homepage
async def homepage() -> dict:
    """Return startup commands."""

    script_folder = atlantis.get_script_folder()
    if not script_folder:
        raise RuntimeError("Cannot determine homepage script folder")

    # Callback `auto` only searches the current folder, and the preflight/chat
    # callbacks live in the sibling Chat app
    chat_folder = posixpath.join(posixpath.dirname(script_folder), "Chat")
    logger.info(f"homepage script folder: {script_folder}, chat folder: {chat_folder}")

    # The commands below run after we return, so this is the folder to come back to.
    # Ask the caller's terminal: the exec shell runs this call from the Home folder
    start_folder = await atlantis.client_command("pwd", shell="caller")
    logger.info(f"homepage pwd returned:\n{format_json_log(start_folder, colored=True)}")
    if not isinstance(start_folder, str) or not start_folder.startswith("/"):
        raise TypeError(f"pwd returned {start_folder!r}, expected an absolute folder path")

    return {
        "commands": [
            #"/terminal blur 12",
            f"/path push {script_folder}",
            "/env save",
            f"/cd {chat_folder}",
            "/callback set preflight auto",
            "/callback set chat auto",
            # The file callback lives here in Home, so it needs its own visit
            f"/cd {script_folder}",
            "file_callback_auto",
            f"/cd {start_folder}",
            # Chat callbacks resolve their game from the cursor; new owners start in Setup
            #'/cursor join {game_name: "Setup"}',
            #f"/path push {script_folder}",
            #"/env save",
            #"/terminal on",
            #"app on",
            #"term_default",
            #"user_bg_default",
            #"first_menu",
            #"/finally terminal blur 0",
        ],
    }
