"""Homepage menu for the local Home app."""

import logging
from pathlib import Path

import atlantis
from utils import format_json_log

from .modal import modal_menu

logger = logging.getLogger("dynamic_function")



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
    items.append({"id": "do_nothing", "text": "do nothing"})

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

    return {
        "commands": [
            #"/terminal blur 12",
            f"/cd {script_folder}",
            f"/path push {script_folder}",
            "/env save",
            "/terminal on",
            "app on",
            "term_default",
            "user_bg_default",
            "first_menu",
            "/finally terminal blur 0",
        ],
    }
