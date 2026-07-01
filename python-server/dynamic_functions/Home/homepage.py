"""Homepage menu for the local Home app."""

from pathlib import Path

import atlantis

from .modal import modal_menu


# % first_menu


@public
async def first_menu():
    """Let the user choose where to go next."""

    choice = await modal_menu(
        [
            {"id": "explore_demo_folder", "text": "Explore demo folder"},
        ],
        title="Home",
        heading="Where do you want to go?",
    )
    if choice is None:
        await atlantis.client_log("Home menu cancelled.")
        return None

    script_folder = atlantis.get_script_folder()
    await atlantis.client_log(f"first_menu: script_folder={script_folder}")
    if not script_folder:
        raise RuntimeError("Cannot determine homepage script folder")

    choice_id = str(choice["id"])
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
    await atlantis.client_log(f"homepage: script_folder={script_folder}")
    if not script_folder:
        raise RuntimeError("Cannot determine homepage script folder")

    return {
        "commands": [
            f"/cd {script_folder}",
            f"/path unshift {script_folder}",
            "/terminal on",
            "app on",
            "term_default",
            "user_bg_default",
            "first_menu",
        ],
    }
