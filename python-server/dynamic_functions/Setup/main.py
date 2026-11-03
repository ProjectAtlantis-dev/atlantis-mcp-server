import asyncio
import logging
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
    Runs when a terminal cd enters Setup
    """
    logger.info(f"Executing Setup on_enter...")

    #await atlantis.client_log("enter running")
    await atlantis.set_background(str(SETUP_BG_PATH), vertical_align="center")

    # The app folder name is the Chat game name; preflight and chat read it from the cursor
    await atlantis.client_command("/cursor join", {"game_name": Path(__file__).parent.name})

    return "Setup entered"


@visible
@exit
async def on_exit():
    """
    Runs when a terminal cd leaves Setup
    """
    logger.info(f"Executing Setup on_exit...")

    #await atlantis.client_log("exit running")
    await atlantis.clear_background()

    # Drop the game cursor joined in on_enter
    await atlantis.client_command("/cursor clear")

    return "Setup exited"


@visible
async def get_status():
    """
    Reports what is installed for the Terrain setup: the Terrain folder
    (mcp-pkg-terrain) in dynamic_functions, and the Atlantis 3D viewer
    (atlantis-3d-viewer) beside this MCP server checkout
    """
    logger.info(f"Executing get_status...")

    terrain = await _checkout_status(TERRAIN_PATH, TERRAIN_REPO, "main.py")
    viewer = await _checkout_status(VIEWER_PATH, VIEWER_REPO, "package.json")
    viewer["dependenciesInstalled"] = (VIEWER_PATH / "node_modules").is_dir()

    return {"terrain": terrain, "viewer": viewer}


TERRAIN_REPO = "https://github.com/ProjectAtlantis-dev/mcp-pkg-terrain"
VIEWER_REPO = "https://github.com/ProjectAtlantis-dev/atlantis-3d-viewer"
# Setup/main.py -> dynamic_functions -> python-server -> checkout root
TERRAIN_PATH = Path(__file__).absolute().parents[1] / "Terrain"
VIEWER_PATH = Path(__file__).absolute().parents[3].parent / "atlantis-3d-viewer"


def _repo_slug(url: str) -> str:
    # https://github.com/Owner/repo(.git) and git@github.com:Owner/repo(.git) -> owner/repo
    slug = url.strip().removesuffix("/").removesuffix(".git")
    slug = slug.split("github.com", 1)[-1].lstrip(":/")
    return slug.lower()


async def _git(path: Path, *args: str) -> str | None:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(path), *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    return out.decode().strip() if proc.returncode == 0 else None


async def _checkout_status(path: Path, repo: str, marker: str) -> dict:
    status = {"path": str(path), "repository": repo,
              "installed": (path / marker).is_file()}
    if not path.is_dir():
        return status
    remotes = await _git(path, "remote", "-v")
    urls = sorted({line.split()[1] for line in (remotes or "").splitlines() if len(line.split()) > 1})
    status["remotes"] = urls
    status["fromRepository"] = any(_repo_slug(url) == _repo_slug(repo) for url in urls)
    status["revision"] = await _git(path, "rev-parse", "HEAD")
    return status
