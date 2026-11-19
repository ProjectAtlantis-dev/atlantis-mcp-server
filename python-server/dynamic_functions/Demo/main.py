import json
import logging
from datetime import datetime
from pathlib import Path

import atlantis

logger = logging.getLogger("dynamic_function")

# Data for the Chat game this app joins
DATA_PATH = Path(__file__).absolute().parents[1] / "Chat" / Path(__file__).parent.name / "Data"


@index
@public
async def index():
    """Demo app"""
    return "Welcome to the Atlantis demo app!"


@visible
@enter
async def on_enter():
    """
    Runs when entering chat
    """
    logger.info(f"Executing Demo on_enter...")

    # The app folder name is the Chat game name; preflight and chat read it from the cursor
    await atlantis.client_command("/cursor join", {"game_name": Path(__file__).parent.name})

    # Point the calling terminal at this folder's chat shell (upserted by Node, keyed by folder)
    await atlantis.client_command("/chat bind " + json.dumps(atlantis.get_script_folder()), shell="caller")

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
    await roster_leave(Path(__file__).parent.name)

    # Drop the game cursor joined in on_enter
    await atlantis.client_command("/cursor clear")

    # Detach the calling terminal from this folder's chat shell; the chat shell stays
    await atlantis.client_command("/chat unbind", shell="caller")

    return "Demo exited"


@visible
async def get_status():
    """
    Demo status for the current user: who they are and whether you have met
    them in this game. Call this first.
    """
    logger.info(f"Executing get_status...")

    username = atlantis.get_caller()
    if not username:
        raise RuntimeError("Demo has no caller identity")
    game_uuid = atlantis.get_game_uuid()
    if not game_uuid:
        raise RuntimeError("get_status has no game_uuid; it must be called from the Demo chat")

    visitors_path = _game_visitors_path(game_uuid)
    visitors = _read_json(visitors_path)
    # Read before this call's record is written, so a repeat call in the same game counts
    if username in visitors:
        seen = f"You have already met {username} in this game; carry on from where you are."
    else:
        seen = (f"You have not met {username}. Introduce yourself and offer to show them "
                f"what the example tools in this Demo app do.")

    visitors[username] = {"lastSeen": datetime.now().isoformat(timespec="seconds")}
    _write_json(visitors_path, visitors)
    return f"The user is {username}. {seen}"


def _game_visitors_path(game_uuid: str) -> Path:
    """This game's record of who its bot has met: {username: {lastSeen}}.

    Bot memory lives with the game, so a new game starts with a bot that has
    met nobody. The game folder must already exist (Chat creates it on join).
    """
    game_dir = DATA_PATH / "games" / game_uuid
    if not game_dir.is_dir():
        raise FileNotFoundError(f"No game folder for game {game_uuid!r} at {game_dir}")
    return game_dir / "visitors.json"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
