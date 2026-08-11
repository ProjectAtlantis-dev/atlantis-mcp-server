"""Homepage menu for the local Home app."""

import logging
from pathlib import Path

import atlantis
from utils import format_json_log

from .modal import modal_menu

logger = logging.getLogger("dynamic_function")



# % first_menu


def _app_menu_items(tree_entries: list, script_folder: str) -> list[dict]:
    """Menu items for sibling app folders that expose a public first_menu.

    `first_menu` is a per-folder entry point: an app declares one at its root
    (e.g. Chat/runner.py) and we `/cd` into the folder to invoke it. This menu
    is itself such an entry point, so `script_folder` names the one to skip.
    """

    items = {}
    for entry in tree_entries:
        parts = entry["filename"].split("/")
        # <app>/<file>.py — the app folder is the first path segment.
        if len(parts) != 2:
            continue
        if "Public" not in entry["chatStatus"]:
            continue
        # searchTerm is the absolute function path; its parent is the app folder.
        app_path = entry["searchTerm"].rsplit("/", 1)[0]
        if app_path == script_folder:
            continue
        items[parts[0]] = {"id": f"app:{app_path}", "text": entry["description"]}
    return [items[folder] for folder in sorted(items)]


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

    # Discovered apps lead; the demo folder is the fallback for someone with
    # nowhere better to go, so it sits last.
    items = _app_menu_items(tree_entries, script_folder)
    items.append({"id": "explore_terrain_folder", "text": "Explore terrain folder"})
    items.append({"id": "explore_demo_folder", "text": "Explore demo folder"})

    choice = await modal_menu(
        items,
        title="Home",
        heading="Where do you want to go?",
    )

    choice_id = str(choice["id"])
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

    if choice_id == "explore_terrain_folder":
        commands = [
            f"/cd {script_folder}",
            "cd ..",
            "cd Terrain",
            "ls",
        ]
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
