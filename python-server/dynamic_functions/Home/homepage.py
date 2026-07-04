"""Homepage menu for the local Home app."""

import logging
from pathlib import Path

import atlantis
from utils import format_json_log

from .modal import modal_menu

logger = logging.getLogger("dynamic_function")



# % first_menu


def _app_menu_items(tree_entries: list) -> list[dict]:
    """Menu items for subfolders that expose a public Home/first_menu."""

    items = {}
    for entry in tree_entries:
        parts = entry["filename"].split("/")
        # <subfolder>/Home/<file>.py — a bare Home/... entry is this menu itself.
        if len(parts) < 3 or parts[-2] != "Home":
            continue
        if "Public" not in entry["chatStatus"]:
            continue
        folder = parts[0]
        # searchTerm is the absolute function path; its parent is the app's Home folder.
        home_path = entry["searchTerm"].rsplit("/", 1)[0]
        items[folder] = {"id": f"app:{home_path}", "text": entry["description"]}
    return [items[folder] for folder in sorted(items)]


@public
async def first_menu():
    """Let the user choose where to go next."""

    items = [
        {"id": "explore_demo_folder", "text": "Explore demo folder"},
    ]

    cwd = await atlantis.client_command("pwd")
    logger.info(f"pwd returned:\n{format_json_log(cwd, colored=True)}")

    tree_entries = await atlantis.client_command("tree ../*/Home/first_menu")
    logger.info(f"tree first_menu (from {cwd}) returned:\n{format_json_log(tree_entries, colored=True)}")
    items.extend(_app_menu_items(tree_entries))

    choice = await modal_menu(
        items,
        title="Home",
        heading="Where do you want to go?",
    )

    if choice is None:
        await atlantis.client_log("Home menu cancelled.")
        return None

    script_folder = atlantis.get_script_folder()
    if not script_folder:
        raise RuntimeError("Cannot determine homepage script folder")

    choice_id = str(choice["id"])
    if choice_id.startswith("app:"):
        home_path = choice_id[4:]
        commands = [
            f"/cd {home_path}",
            "pwd",
            "first_menu",
        ]
        logger.info(f"launching app script for '{home_path}':\n{format_json_log(commands, colored=True)}")
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
            content="Demo folder coming right up, but if you want to do anything cool you need to mount a filesystem...",
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
            "/terminal blur 12",
            f"/cd {script_folder}",
            f"/path push {script_folder}",
            "/terminal on",
            "app on",
            "term_default",
            "user_bg_default",
            "first_menu",
            "/finally terminal blur 0",
        ],
    }
