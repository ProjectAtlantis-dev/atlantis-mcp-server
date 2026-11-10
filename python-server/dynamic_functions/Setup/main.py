import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import atlantis

logger = logging.getLogger("dynamic_function")

# Placeholder background: builder.jpg in the project root
SETUP_BG_PATH = Path(__file__).absolute().parents[3] / "builder.jpg"


@visible
@index
async def index():
    """
    Folder for Setup
    """
    logger.info(f"Executing placeholder function: index...")

    await atlantis.client_log("index running")

    # Replace this return statement with your function's result
    return f"Placeholder function 'index' executed successfully."


@visible
@enter
async def on_enter():
    """
    Runs when entering chat
    """
    logger.info(f"Executing Setup on_enter...")

    #await atlantis.client_log("enter running")
    await atlantis.set_background(str(SETUP_BG_PATH), vertical_align="center")

    # The app folder name is the Chat game name; preflight and chat read it from the cursor
    await atlantis.client_command("/cursor join", {"game_name": Path(__file__).parent.name})

    # Point the calling terminal at this folder's chat shell (upserted by Node, keyed by folder)
    await atlantis.client_command("/chat bind " + json.dumps(atlantis.get_script_folder()), shell="caller")

    return "Setup entered"


@visible
@exit
async def on_exit():
    """
    Runs when leaving chat
    """
    logger.info(f"Executing Setup on_exit...")

    #await atlantis.client_log("exit running")
    await atlantis.clear_background()

    # Drop the game cursor joined in on_enter
    await atlantis.client_command("/cursor clear")

    # Detach the calling terminal from this folder's chat shell; the chat shell stays
    await atlantis.client_command("/chat unbind", shell="caller")

    return "Setup exited"


TERRAIN_REPO = "https://github.com/ProjectAtlantis-dev/mcp-pkg-terrain"
VIEWER_REPO = "https://github.com/ProjectAtlantis-dev/atlantis-3d-viewer"
# Setup/main.py -> dynamic_functions -> python-server -> checkout root
TERRAIN_PATH = Path(__file__).absolute().parents[1] / "Terrain"
VIEWER_PATH = Path(__file__).absolute().parents[3].parent / "atlantis-3d-viewer"
# Game data for the Chat game this app joins: {username: {game_uuid: {lastSeen, step, state}}}
VISITORS_PATH = Path(__file__).absolute().parents[1] / "Chat" / Path(__file__).parent.name / "Data" / "visitors.json"


@visible
async def get_status():
    """
    Setup status for the current user: who they are, whether you have helped
    them before, what is installed, which setup step they are on and exactly
    what to do next. Call this first, and again after each step.
    """
    logger.info(f"Executing get_status...")

    username = atlantis.get_caller()
    if not username:
        raise RuntimeError("get_status has no caller identity")
    game_uuid = atlantis.get_game_uuid()
    if not game_uuid:
        raise RuntimeError("get_status has no game_uuid; it must be called from the Setup chat")

    terrain = await _checkout_status(TERRAIN_PATH, TERRAIN_REPO, "main.py")
    viewer = await _checkout_status(VIEWER_PATH, VIEWER_REPO, "package.json")
    step, state, next_action = _current_step(terrain, viewer)

    visitors = _read_visitors()
    visits = visitors.setdefault(username, {})
    earlier = sorted(
        (visit for uuid, visit in visits.items() if uuid != game_uuid),
        key=lambda visit: visit["lastSeen"])
    if earlier:
        last = earlier[-1]
        seen = (f"You have helped {username} before: last seen {last['lastSeen']}, "
                f"at step {last['step']} ({SETUP_STEPS[last['step'] - 1]}).")
    else:
        seen = f"You have not seen {username} before."

    visits[game_uuid] = {"lastSeen": datetime.now().isoformat(timespec="seconds"),
                         "step": step, "state": state}
    _write_visitors(visitors)

    return (f"The user is {username}. {seen}\n"
            f"Step 1 of {len(SETUP_STEPS)} ({SETUP_STEPS[0]}) is done: this remote is running.\n"
            f"Current step {step}: {SETUP_STEPS[step - 1]}. {state}\n"
            f"Next: {next_action}")


SETUP_STEPS = [
    "connect the remote",
    "install Terrain",
    "install the 3D viewer",
    "install the 3D viewer dependencies",
    "finished",
]


def _current_step(terrain: dict, viewer: dict) -> tuple[int, str, str]:
    """(step number, what the state is, what to do next) for the first unfinished step"""
    problem = _checkout_problem("Terrain", terrain)
    if problem:
        return 2, problem, _fix(terrain)
    problem = _checkout_problem("The 3D viewer", viewer)
    if problem:
        return 3, problem, _fix(viewer)
    if not (VIEWER_PATH / "node_modules").is_dir():
        return (4, f"The 3D viewer is cloned at {VIEWER_PATH} but its dependencies are not installed.",
                f"have the user run: cd {VIEWER_PATH} && npm install")
    return (5, f"Terrain ({terrain['revision'][:7]}) and the 3D viewer ({viewer['revision'][:7]}) are installed.",
            "setup is complete; tell the user they are ready to use Terrain.")


def _checkout_problem(name: str, status: dict) -> str | None:
    path = status["path"]
    if not Path(path).exists():
        return f"{name} is not installed; {path} does not exist."
    if not status["gitCheckout"]:
        return f"{path} exists but is not a git checkout."
    if not status["fromRepository"]:
        return f"{path} is a git checkout of {', '.join(status['remotes']) or 'no remote'}, not {status['repository']}."
    if not status["installed"]:
        return f"{path} is a checkout of {status['repository']} but is incomplete."
    return None


def _fix(status: dict) -> str:
    if not Path(status["path"]).exists():
        return f"have the user run: git clone {status['repository']} {status['path']}"
    return "do not change that folder; tell the user what you found and escalate."


def _read_visitors() -> dict:
    if not VISITORS_PATH.is_file():
        return {}
    visitors = json.loads(VISITORS_PATH.read_text(encoding="utf-8"))
    if not isinstance(visitors, dict) or not all(isinstance(v, dict) for v in visitors.values()):
        raise ValueError(f"{VISITORS_PATH} must map usernames to objects of visits")
    return visitors


def _write_visitors(visitors: dict) -> None:
    VISITORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = VISITORS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(visitors, indent=2) + "\n", encoding="utf-8")
    tmp.replace(VISITORS_PATH)


def _repo_slug(url: str) -> str:
    # https://github.com/Owner/repo(.git) and git@github.com:Owner/repo(.git) -> owner/repo
    slug = url.strip().removesuffix("/").removesuffix(".git")
    slug = slug.split("github.com", 1)[-1].lstrip(":/")
    return slug.lower()


async def _git(path: Path, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(path), *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {path} (exit {proc.returncode}): {err.decode().strip()}")
    return out.decode().strip()


async def _checkout_status(path: Path, repo: str, marker: str) -> dict:
    status = {"path": str(path), "repository": repo,
              "installed": (path / marker).is_file()}
    # .git is a directory in a normal clone and a file in a worktree or submodule
    status["gitCheckout"] = (path / ".git").exists()
    if not status["gitCheckout"]:
        return status
    remotes = await _git(path, "remote", "-v")
    urls = sorted({line.split()[1] for line in remotes.splitlines() if len(line.split()) > 1})
    status["remotes"] = urls
    status["fromRepository"] = any(_repo_slug(url) == _repo_slug(repo) for url in urls)
    status["revision"] = await _git(path, "rev-parse", "HEAD")
    return status
