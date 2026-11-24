import asyncio
import logging
from pathlib import Path

import atlantis

logger = logging.getLogger("dynamic_function")


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


# Setup is the package installer: it only puts packages on this machine. Each
# package's own Robot sets that package up once it is here (see
# Setup/Robot/main.py); a Robot cannot install itself, so installing stays here.
TERRAIN_REPO = "https://github.com/ProjectAtlantis-dev/mcp-pkg-terrain"
TERRAIN_PATH = Path(__file__).absolute().parents[1] / "Terrain"


async def _check_remote() -> dict:
    return {"state": "this remote is running.", "fix": None}


async def _check_terrain_checkout() -> dict:
    status = await _checkout_status(TERRAIN_PATH, TERRAIN_REPO, "main.py")
    if not TERRAIN_PATH.exists():
        return {"state": f"Terrain is not installed; {TERRAIN_PATH} does not exist.",
                "fix": f"have the user run: git clone {TERRAIN_REPO} {TERRAIN_PATH}"}
    problem = _checkout_problem("Terrain", status)
    if problem:
        return {"state": problem, "fix": "do not change that folder; tell the user what you found and escalate."}
    return {"state": f"Terrain ({status['revision'][:7]}) is installed at {TERRAIN_PATH}.", "fix": None}


SETUP_STEPS = [
    # (name, steps that must be done first, check)
    ("connect the remote", (), _check_remote),
    ("install Terrain", (), _check_terrain_checkout),
]


async def check_steps() -> list[dict]:
    """One {name, done, state, fix, waitsOn} per SETUP_STEPS entry, in order"""
    steps: list[dict] = []
    for name, needs, check in SETUP_STEPS:
        waits_on = next((number for number in needs if not steps[number - 1]["done"]), None)
        if waits_on is not None:
            steps.append({"name": name, "done": False, "waitsOn": waits_on, "fix": None,
                          "state": f"needs step {waits_on} ({SETUP_STEPS[waits_on - 1][0]}) first."})
            continue
        result = await check()
        steps.append({"name": name, "done": result["fix"] is None, "waitsOn": None, **result})
    return steps


async def terrain_installed() -> bool:
    """Whether Terrain is here, so its Robot can set it up"""
    return (await _check_terrain_checkout())["fix"] is None


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
