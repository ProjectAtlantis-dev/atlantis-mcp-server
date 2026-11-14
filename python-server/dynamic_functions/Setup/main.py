import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

import atlantis
from dotenv import dotenv_values

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
# Terrain reloads this on every start; the viewer server refuses to start without the token
TERRAIN_ENV_PATH = TERRAIN_PATH / ".env"
TERRAIN_ENV_KEYS = ("DATAFORSYNINGEN_TOKEN", "DATAFORSYNINGEN_FTP_USER", "DATAFORSYNINGEN_FTP_PASS")
DATAFORSYNINGEN_URL = "https://dataforsyningen.dk/"
DATAFORSYNINGEN_TOKEN_URL = "https://dataforsyningen.dk/user#token"
# Tile ids the old onboarding used for its live provider checks
TOKEN_CHECK_TILE = "10-328-212"
ELEVATION_CHECK_TILE = "10-334-192"
EGM2008_GRID = "us_nga_egm08_25.tif"
# The viewer's proxy, the old onboarding and the user's restart command all assume this port
VIEWER_PORT = 5173
VIEWER_URL = f"http://localhost:{VIEWER_PORT}/"
VIEWER_LOG_PATH = VIEWER_PATH / "runViteServer.log"
TAKRAM_PACKAGES = ("atmosphere", "clouds", "core", "effects")
HTTP_TIMEOUT_SECONDS = 10
VIEWER_START_TIMEOUT_SECONDS = 60

# Data for the Chat game this app joins
DATA_PATH = Path(__file__).absolute().parents[1] / "Chat" / Path(__file__).parent.name / "Data"
# {username: {game_uuid: {lastSeen, done}}}
VISITORS_PATH = DATA_PATH / "visitors.json"
# {username: {confirmed: {step: when}, restartPractice: null | "declined" | {stoppedAt}}}
PROGRESS_PATH = DATA_PATH / "progress.json"
# Machine facts that are too slow or costly to re-prove on every get_status
CHECKS_PATH = DATA_PATH / "checks.json"

CONFIRMABLE_STEPS = ("terminal tour", "viewer visible", "viewer restart")


@visible
async def get_status():
    """
    Setup status for the current user: who they are, whether you have helped
    them before, every setup step with whether it is done, and exactly what to
    do next. Call this first, and again after each step.
    """
    logger.info(f"Executing get_status...")

    username = _caller()
    game_uuid = atlantis.get_game_uuid()
    if not game_uuid:
        raise RuntimeError("get_status has no game_uuid; it must be called from the Setup chat")

    steps = await _check_steps(username)
    done = [number for number, step in enumerate(steps, 1) if step["done"]]

    visitors = _read_json(VISITORS_PATH)
    visits = visitors.setdefault(username, {})
    # Read before this call's record is written, so a repeat call in the same game counts
    if visits:
        last = max(visits.values(), key=lambda visit: visit["lastSeen"])
        seen = (f"You have helped {username} before: last seen {last['lastSeen']}, "
                f"with {len(last['done'])} steps done.")
    else:
        seen = f"You have not seen {username} before."

    visits[game_uuid] = {"lastSeen": _now(), "done": done}
    _write_json(VISITORS_PATH, visitors)

    current = next((number for number, step in enumerate(steps, 1) if not step["done"]), None)
    lines = [f"The user is {username}. {seen}", f"Setup has {len(SETUP_STEPS)} steps:"]
    for number, ((name, _, _), step) in enumerate(zip(SETUP_STEPS, steps), 1):
        if step["done"]:
            label = "done"
        elif number == current:
            label = "NEXT"
        elif step["waitsOn"]:
            label = f"waiting on step {step['waitsOn']}"
        else:
            label = "to do"
        lines.append(f"{number}. [{label}] {name}: {step['state']}")
    if current is None:
        lines.append("Next: setup is complete; tell the user they are ready to use Terrain.")
    else:
        lines.append(f"Next (step {current}): {steps[current - 1]['fix']}")
    return "\n".join(lines)


@visible
async def confirm_step(step: str):
    """
    Record that the user confirmed a step only they can see: "terminal tour",
    "viewer visible" or "viewer restart". Call it only after the user has
    actually said so. Call get_status after.
    """
    logger.info(f"Executing confirm_step({step!r})...")

    if step not in CONFIRMABLE_STEPS:
        raise ValueError(f"step must be one of {', '.join(CONFIRMABLE_STEPS)}, not {step!r}")
    username = _caller()
    if step in ("viewer visible", "viewer restart"):
        viewer = await _viewer_status()
        if not viewer["ok"]:
            raise RuntimeError(f"Cannot confirm {step!r}: {viewer['state']}")
    progress = _read_json(PROGRESS_PATH)
    user = progress.setdefault(username, {"confirmed": {}, "restartPractice": None})
    if step == "viewer restart" and not isinstance(user["restartPractice"], dict):
        raise RuntimeError("Cannot confirm 'viewer restart': the viewer was not stopped for practice")
    user["confirmed"][step] = _now()
    _write_json(PROGRESS_PATH, progress)
    return f"Recorded that {username} confirmed {step!r}."


@visible
async def install_3d_viewer():
    """
    Install the 3D viewer: clones the atlantis-3d-viewer repo into place if it
    is missing, otherwise reports what is already there. Call get_status after.
    """
    logger.info(f"Executing install_3d_viewer...")

    status = await _checkout_status(VIEWER_PATH, VIEWER_REPO, "package.json")
    if Path(status["path"]).exists():
        problem = _checkout_problem("The 3D viewer", status)
        if problem:
            # Something else is in the way; leave it for a human
            raise RuntimeError(f"{problem} Not touching it.")
        return f"The 3D viewer ({status['revision'][:7]}) is already installed at {VIEWER_PATH}."

    await atlantis.client_log(f"Cloning {VIEWER_REPO} into {VIEWER_PATH}...")
    await _git(VIEWER_PATH.parent, "clone", VIEWER_REPO, str(VIEWER_PATH))

    status = await _checkout_status(VIEWER_PATH, VIEWER_REPO, "package.json")
    problem = _checkout_problem("The 3D viewer", status)
    if problem:
        raise RuntimeError(f"Clone finished but {problem}")
    return (f"Installed the 3D viewer ({status['revision'][:7]}) at {VIEWER_PATH}. "
            f"Next: call install_3d_viewer_dependencies.")


@visible
async def install_3d_viewer_dependencies():
    """
    Install the 3D viewer's dependencies by running the viewer's own setup
    script: the pinned Takram three-geospatial checkout, npm ci, and its
    compatibility test. Takes a few minutes. Call get_status after.
    """
    logger.info(f"Executing install_3d_viewer_dependencies...")

    status = await _checkout_status(VIEWER_PATH, VIEWER_REPO, "package.json")
    problem = _checkout_problem("The 3D viewer", status)
    if problem:
        raise RuntimeError(f"Cannot install the 3D viewer dependencies: {problem}")

    await atlantis.client_log(f"Running the 3D viewer setup in {VIEWER_PATH}; this takes a few minutes...")
    proc = await asyncio.create_subprocess_exec(
        str(VIEWER_PATH / "setup"), cwd=str(VIEWER_PATH),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    output = out.decode().strip()
    logger.info(f"{VIEWER_PATH / 'setup'} exited {proc.returncode}:\n{output}")
    if proc.returncode != 0:
        raise RuntimeError(f"The 3D viewer setup failed (exit {proc.returncode}):\n{_tail(output)}")

    problem = await _viewer_dependencies_problem()
    if problem:
        raise RuntimeError(f"The 3D viewer setup succeeded but {problem}")
    return "Installed the 3D viewer dependencies; its setup and compatibility test passed."


@visible
async def open_dataforsyningen():
    """
    Open the Dataforsyningen website in a new browser tab, where the user
    registers or logs in to get the token and FTP login Terrain needs.
    """
    logger.info(f"Executing open_dataforsyningen...")

    await _open_tab(DATAFORSYNINGEN_URL)
    return f"Opened {DATAFORSYNINGEN_URL} in a new tab of the user's default browser."


@visible
async def check_dataforsyningen_token():
    """
    Prove the Dataforsyningen token in Terrain/.env works by fetching one live
    satellite imagery tile. Call get_status after.
    """
    logger.info(f"Executing check_dataforsyningen_token...")

    missing = _missing_terrain_env_keys()
    if missing is None or missing:
        raise RuntimeError(f"Cannot check the token: {_env_problem(missing)}")

    # Imported here: Terrain is its own checkout and may not exist yet
    from dynamic_functions.Terrain.dataforsyningen import dataforsyningen_fetch
    from dynamic_functions.Terrain.terrain_config import _load_environment

    # The same reload terrain_start does, so this proves what the server will use
    _load_environment()
    try:
        result = await asyncio.to_thread(dataforsyningen_fetch, TOKEN_CHECK_TILE)
    except Exception as exc:
        # The message can carry the request URL, and the URL carries the token
        raise RuntimeError(
            f"Dataforsyningen check failed: {type(exc).__name__} (message withheld; it can "
            f"contain the token)") from None

    status, http_status = result.get("status"), result.get("httpStatus")
    if status == "authentication_error":
        raise RuntimeError(
            f"Dataforsyningen rejected the token (HTTP {http_status}). Have the user check "
            f"or replace DATAFORSYNINGEN_TOKEN in {TERRAIN_ENV_PATH}.")
    if status != "success" or not isinstance(http_status, int) or not 200 <= http_status < 300:
        raise RuntimeError(
            f"Dataforsyningen check returned {status!r} HTTP {http_status}. This is not proof "
            f"the token is bad; resolve this failure before continuing.")

    checks = _read_json(CHECKS_PATH)
    checks["dataforsyningenToken"] = {"digest": _token_digest(), "checkedAt": _now()}
    _write_json(CHECKS_PATH, checks)
    return f"The Dataforsyningen token works: imagery tile {TOKEN_CHECK_TILE} returned HTTP {http_status}."


@visible
async def install_egm2008_grid():
    """
    Install the EGM2008 geoid grid PROJ needs to convert ArcticDEM elevations
    to sea-level heights. Call get_status after.
    """
    logger.info(f"Executing install_egm2008_grid...")

    if _egm2008_available():
        return f"The EGM2008 geoid grid ({EGM2008_GRID}) is already installed."

    await atlantis.client_log(f"Downloading the EGM2008 geoid grid ({EGM2008_GRID})...")
    # This server's own Python, so the grid lands where this server's PROJ looks
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pyproj", "sync", "--file", EGM2008_GRID,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    output = out.decode().strip()
    logger.info(f"pyproj sync --file {EGM2008_GRID} exited {proc.returncode}:\n{output}")
    if proc.returncode != 0:
        raise RuntimeError(f"pyproj sync failed (exit {proc.returncode}):\n{_tail(output)}")
    if not _egm2008_available():
        raise RuntimeError(f"pyproj sync finished but PROJ still has no EGM2008 transform:\n{_tail(output)}")
    return f"Installed the EGM2008 geoid grid ({EGM2008_GRID})."


@visible
async def check_elevation_data():
    """
    Prove Terrain can read real elevation data by fetching one live ArcticDEM
    tile, corrected to sea level with the EGM2008 grid. Nothing is stored.
    Call get_status after.
    """
    logger.info(f"Executing check_elevation_data...")

    if not _egm2008_available():
        raise RuntimeError("Cannot check elevation data: the EGM2008 geoid grid is not installed")

    # Imported here: Terrain is its own checkout and may not exist yet
    from dynamic_functions.Terrain.arctic_dem import arcticdem_fetch

    await atlantis.client_log(f"Fetching ArcticDEM tile {ELEVATION_CHECK_TILE}...")
    result = await asyncio.to_thread(arcticdem_fetch, ELEVATION_CHECK_TILE)
    cells = result["shape"][0] * result["shape"][1]
    if result["minimum"] is None or result["nanCount"] >= cells:
        raise RuntimeError(f"ArcticDEM tile {ELEVATION_CHECK_TILE} came back with no elevation values")

    sample = {"tileId": ELEVATION_CHECK_TILE, "sampleCount": cells - result["nanCount"],
              "minimum": result["minimum"], "maximum": result["maximum"], "checkedAt": _now()}
    checks = _read_json(CHECKS_PATH)
    checks["elevation"] = sample
    _write_json(CHECKS_PATH, checks)
    return (f"Elevation data works: tile {ELEVATION_CHECK_TILE} has {sample['sampleCount']} values "
            f"from {sample['minimum']:.1f} m to {sample['maximum']:.1f} m.")


@visible
async def start_terrain():
    """
    Start Terrain: opens the terrain database and asset catalog and starts the
    viewer server, then checks the server answers. Call get_status after.
    """
    logger.info(f"Executing start_terrain...")

    status = await _checkout_status(TERRAIN_PATH, TERRAIN_REPO, "main.py")
    problem = _checkout_problem("Terrain", status)
    if problem:
        raise RuntimeError(f"Cannot start Terrain: {problem}")

    # Imported here: Terrain is its own checkout and may not exist yet
    from dynamic_functions.Terrain.lifecycle import terrain_start

    await atlantis.client_log("Starting Terrain...")
    await terrain_start()

    terrain = await _terrain_status()
    if not terrain["ok"]:
        raise RuntimeError(f"terrain_start returned but {terrain['state']}")
    return f"Terrain is running; {terrain['state']}"


@visible
async def launch_3d_viewer():
    """
    Start the 3D viewer's dev server on port 5173, pointed at the running
    Terrain server, and leave it running. Call get_status after.
    """
    logger.info(f"Executing launch_3d_viewer...")

    viewer = await _viewer_status()
    if viewer["ok"]:
        return f"The 3D viewer is already running: {viewer['state']}"
    if viewer["pid"] is not None:
        raise RuntimeError(f"Cannot launch the 3D viewer: {viewer['state']}")
    problem = await _viewer_dependencies_problem()
    if problem:
        raise RuntimeError(f"Cannot launch the 3D viewer: {problem}")
    terrain = await _terrain_status()
    if not terrain["ok"]:
        raise RuntimeError(f"Cannot launch the 3D viewer: {terrain['state']}")

    await atlantis.client_log(f"Starting the 3D viewer on port {VIEWER_PORT}...")
    log_file = VIEWER_LOG_PATH.open("w", encoding="utf-8")
    try:
        # Its own session, so it outlives this call and a Ctrl-C in the server's terminal
        proc = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(VIEWER_PORT), "--strictPort"],
            cwd=str(VIEWER_PATH), env={**os.environ, "FLASK_PROXY_TARGET": terrain["url"]},
            stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
    finally:
        log_file.close()

    deadline = time.monotonic() + VIEWER_START_TIMEOUT_SECONDS
    while True:
        await asyncio.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(
                f"The 3D viewer exited with code {proc.returncode}:\n"
                f"{_tail(VIEWER_LOG_PATH.read_text(encoding='utf-8'))}")
        viewer = await _viewer_status()
        if viewer["ok"]:
            return f"The 3D viewer is running (pid {proc.pid}, log {VIEWER_LOG_PATH}): {viewer['state']}"
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"The 3D viewer (pid {proc.pid}) is not serving after {VIEWER_START_TIMEOUT_SECONDS}s: "
                f"{viewer['state']}\n{_tail(VIEWER_LOG_PATH.read_text(encoding='utf-8'))}")


@visible
async def open_3d_viewer():
    """
    Open the running 3D viewer in a new browser tab for the user.
    """
    logger.info(f"Executing open_3d_viewer...")

    viewer = await _viewer_status()
    if not viewer["ok"]:
        raise RuntimeError(f"Cannot open the 3D viewer: {viewer['state']}")
    await _open_tab(VIEWER_URL)
    return f"Opened {VIEWER_URL} in a new tab of the user's default browser."


@visible
async def choose_restart_practice(practice: bool):
    """
    Record the user's answer to whether they want the 3D viewer stopped so
    they can practice restarting it. Call get_status after.
    """
    logger.info(f"Executing choose_restart_practice({practice!r})...")

    username = _caller()
    progress = _read_json(PROGRESS_PATH)
    user = progress.setdefault(username, {"confirmed": {}, "restartPractice": None})
    user["restartPractice"] = "chosen" if practice else "declined"
    _write_json(PROGRESS_PATH, progress)
    return f"Recorded that {username} {'wants' if practice else 'does not want'} restart practice."


@visible
async def stop_3d_viewer():
    """
    Stop this installation's 3D viewer dev server so the user can practice
    restarting it. Terrain itself keeps running. Call get_status after.
    """
    logger.info(f"Executing stop_3d_viewer...")

    viewer = await _viewer_status()
    pid = viewer["pid"]
    if pid is None:
        raise RuntimeError(f"Nothing is listening on port {VIEWER_PORT}; the 3D viewer is already stopped")
    if not viewer["ours"]:
        raise RuntimeError(f"Not stopping it: {viewer['state']}")

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + HTTP_TIMEOUT_SECONDS
    while await _port_listener(VIEWER_PORT) is not None:
        if time.monotonic() > deadline:
            raise RuntimeError(f"The 3D viewer (pid {pid}) is still listening {HTTP_TIMEOUT_SECONDS}s after SIGTERM")
        await asyncio.sleep(0.5)

    username = _caller()
    progress = _read_json(PROGRESS_PATH)
    user = progress.setdefault(username, {"confirmed": {}, "restartPractice": None})
    user["restartPractice"] = {"stoppedAt": _now()}
    _write_json(PROGRESS_PATH, progress)
    return f"Stopped the 3D viewer (pid {pid}). Terrain is still running."


async def _check_steps(username: str) -> list[dict]:
    """One {done, state, fix, waitsOn} per SETUP_STEPS entry, in order"""
    progress = _read_json(PROGRESS_PATH).get(username, {"confirmed": {}, "restartPractice": None})
    steps: list[dict] = []
    for name, needs, check in SETUP_STEPS:
        waits_on = next((number for number in needs if not steps[number - 1]["done"]), None)
        if waits_on is not None:
            steps.append({"done": False, "state": f"needs step {waits_on} ({SETUP_STEPS[waits_on - 1][0]}) first.",
                          "fix": None, "waitsOn": waits_on})
            continue
        result = await check(progress)
        steps.append({"done": result["fix"] is None, "waitsOn": None, **result})
    return steps


async def _check_remote(progress: dict) -> dict:
    return {"state": "this remote is running.", "fix": None}


async def _check_terrain_checkout(progress: dict) -> dict:
    return await _check_checkout("Terrain", TERRAIN_PATH, TERRAIN_REPO, "main.py")


async def _check_viewer_checkout(progress: dict) -> dict:
    return await _check_checkout("The 3D viewer", VIEWER_PATH, VIEWER_REPO, "package.json")


async def _check_checkout(name: str, path: Path, repo: str, marker: str) -> dict:
    status = await _checkout_status(path, repo, marker)
    problem = _checkout_problem(name, status)
    if problem:
        return {"state": problem, "fix": _fix(status)}
    return {"state": f"{name} ({status['revision'][:7]}) is installed at {path}.", "fix": None}


async def _check_terminal_tour(progress: dict) -> dict:
    if "terminal tour" in progress["confirmed"]:
        return {"state": f"the user confirmed it on {progress['confirmed']['terminal tour']}.", "fix": None}
    return {"state": "the user has not looked at Terrain from this terminal yet.", "fix": (
        "the user is already in a terminal: this chat runs in it, and lines starting with / are "
        "shell commands. Have them type /cd $Terrain and then /ls, and ask whether they see the "
        "list of functions in the Terrain folder. Changing folders does not leave this chat. End "
        "your turn and wait. Only after they confirm they see the list, have them type /cd "
        "$Setup to come back, then call confirm_step with step \"terminal tour\".")}


async def _check_terrain_config(progress: dict) -> dict:
    missing = _missing_terrain_env_keys()
    if missing is None or missing:
        return {"state": _env_problem(missing), "fix": (
            f"the user needs a Dataforsyningen account. Call open_dataforsyningen to open the "
            f"site in a browser tab; they choose \"Opret bruger\" to register, or log in. They "
            f"may find or create their API token at {DATAFORSYNINGEN_TOKEN_URL}; otherwise the "
            f"user icon, then \"Administrer token til webservices og API'er\", then \"Opret ny "
            f"token\". Then have them put {', '.join(TERRAIN_ENV_KEYS)} (the token, and their "
            f"account username and password) into {TERRAIN_ENV_PATH} themselves. Tell them "
            f"never to paste these into the chat. End your turn and wait until they say it is "
            f"done.")}
    checked = _read_json(CHECKS_PATH).get("dataforsyningenToken")
    if checked is None or checked["digest"] != _token_digest():
        return {"state": f"{TERRAIN_ENV_PATH} has every key, but this token has not been checked.",
                "fix": "call check_dataforsyningen_token; it fetches one live imagery tile."}
    return {"state": f"{TERRAIN_ENV_PATH} has every key; the token worked on {checked['checkedAt']}.",
            "fix": None}


async def _check_egm2008(progress: dict) -> dict:
    if _egm2008_available():
        return {"state": f"PROJ has the EGM2008 grid ({EGM2008_GRID}).", "fix": None}
    return {"state": f"PROJ has no EGM2008 grid ({EGM2008_GRID}); ArcticDEM elevations would fail.",
            "fix": "call install_egm2008_grid; it downloads the grid for you."}


async def _check_elevation(progress: dict) -> dict:
    checked = _read_json(CHECKS_PATH).get("elevation")
    if checked is None:
        return {"state": "elevation data has not been checked.",
                "fix": "call check_elevation_data; it fetches one live ArcticDEM tile and stores nothing."}
    return {"state": (f"tile {checked['tileId']} returned {checked['sampleCount']} values on "
                      f"{checked['checkedAt']}."), "fix": None}


async def _check_viewer_dependencies(progress: dict) -> dict:
    problem = await _viewer_dependencies_problem()
    if problem:
        return {"state": problem, "fix": (
            "call install_3d_viewer_dependencies; it runs the viewer's own setup for you and "
            "takes a few minutes, so tell the user before you call it.")}
    return {"state": "the pinned Takram checkout and node_modules are installed.", "fix": None}


async def _check_terrain_running(progress: dict) -> dict:
    terrain = await _terrain_status()
    if terrain["ok"]:
        return {"state": terrain["state"], "fix": None}
    return {"state": terrain["state"], "fix": "call start_terrain; it starts everything for you."}


async def _check_viewer_running(progress: dict) -> dict:
    viewer = await _viewer_status()
    if viewer["ok"]:
        return {"state": viewer["state"], "fix": None}
    if viewer["pid"] is not None:
        return {"state": viewer["state"], "fix": (
            "do not stop that process; tell the user what is holding the port and escalate.")}
    return {"state": viewer["state"], "fix": "call launch_3d_viewer; it starts the viewer for you."}


async def _check_viewer_visible(progress: dict) -> dict:
    if "viewer visible" in progress["confirmed"]:
        return {"state": f"the user confirmed it on {progress['confirmed']['viewer visible']}.", "fix": None}
    return {"state": "the user has not confirmed they can see the terrain.", "fix": (
        f"call open_3d_viewer to open {VIEWER_URL} in a browser tab. Have the user pick a "
        f"Greenland town in the viewer's town control and let the terrain load. Ask whether "
        f"they see the landscape with satellite imagery; an empty canvas, a loading screen, an "
        f"error overlay or only sky does not count. End your turn and wait. Only after they "
        f"confirm, call confirm_step with step \"viewer visible\".")}


async def _check_restart_practice(progress: dict) -> dict:
    choice = progress["restartPractice"]
    if choice is None:
        return {"state": "the user has not been asked.", "fix": (
            "ask the user: would you like me to shut down the 3D viewer so you can practice "
            "restarting it yourself? End your turn and wait, then call choose_restart_practice "
            "with their answer.")}
    if choice == "declined":
        return {"state": "the user declined.", "fix": None}
    if choice == "chosen":
        return {"state": "the user wants to practice; the viewer is still running.",
                "fix": "call stop_3d_viewer."}
    confirmed = progress["confirmed"].get("viewer restart")
    if confirmed is not None and confirmed >= choice["stoppedAt"]:
        return {"state": f"the user restarted the viewer and confirmed it on {confirmed}.", "fix": None}
    viewer = await _viewer_status()
    if not viewer["ok"]:
        terrain = await _terrain_status()
        return {"state": f"stopped for practice on {choice['stoppedAt']}; {viewer['state']}", "fix": (
            f"show the user this command to run in their own terminal, then end your turn and "
            f"wait: cd {VIEWER_PATH} && FLASK_PROXY_TARGET={terrain['url']} npm run dev -- "
            f"--port {VIEWER_PORT} --strictPort. Explain Terrain keeps running because the viewer "
            f"loads its data from it. It must come back on port {VIEWER_PORT}; any other port "
            f"is a failed restart.")}
    return {"state": f"the user restarted the viewer: {viewer['state']}", "fix": (
        "call open_3d_viewer and ask the user whether the terrain loads again; end your turn "
        "and wait. Only after they confirm, call confirm_step with step \"viewer restart\".")}


SETUP_STEPS = [
    # (name, steps that must be done first, check)
    ("connect the remote", (), _check_remote),
    ("install Terrain", (), _check_terrain_checkout),
    ("configure Terrain", (2,), _check_terrain_config),
    ("install the EGM2008 geoid grid", (), _check_egm2008),
    ("check elevation data", (2, 4), _check_elevation),
    ("install the 3D viewer", (), _check_viewer_checkout),
    ("install the 3D viewer dependencies", (6,), _check_viewer_dependencies),
    ("start Terrain", (2, 3), _check_terrain_running),
    ("look around Terrain from the terminal", (8,), _check_terminal_tour),
    ("launch the 3D viewer", (7, 8), _check_viewer_running),
    ("see the terrain", (10,), _check_viewer_visible),
    ("practice restarting the viewer", (11,), _check_restart_practice),
]


def _caller() -> str:
    username = atlantis.get_caller()
    if not username:
        raise RuntimeError("Setup has no caller identity")
    return username


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])


async def _open_tab(url: str) -> None:
    """Open url in the default browser through the OS opener, like the old MCP onboarder.

    Not window.open from the page: that targets the tool's exec shell and runs
    without a user click, so browsers block it.
    """
    if not await asyncio.to_thread(webbrowser.open, url, 2):
        raise RuntimeError(f"No browser could be opened for {url}; give the user the link to click")


def _missing_terrain_env_keys() -> list[str] | None:
    """Required keys Terrain/.env leaves blank, or None when the file does not exist"""
    if not TERRAIN_ENV_PATH.is_file():
        return None
    values = dotenv_values(TERRAIN_ENV_PATH)
    return [key for key in TERRAIN_ENV_KEYS if not (values.get(key) or "").strip()]


def _env_problem(missing: list[str] | None) -> str:
    if missing is None:
        return f"{TERRAIN_ENV_PATH} does not exist."
    return f"{TERRAIN_ENV_PATH} has no value for {', '.join(missing)}."


def _token_digest() -> str:
    """Fingerprint of the configured token, so a replaced token gets checked again"""
    token = (dotenv_values(TERRAIN_ENV_PATH).get("DATAFORSYNINGEN_TOKEN") or "").strip()
    return hashlib.sha256(token.encode()).hexdigest()


def _egm2008_available() -> bool:
    from pyproj.transformer import TransformerGroup

    # Built fresh each time so a grid installed since the last check is seen
    return TransformerGroup(4979, 9518, always_xy=True).best_available


async def _viewer_dependencies_problem() -> str | None:
    match = re.search(r"^TAKRAM_REVISION=(\w+)$", (VIEWER_PATH / "setup").read_text(encoding="utf-8"),
                      re.MULTILINE)
    if not match:
        raise RuntimeError(f"{VIEWER_PATH / 'setup'} does not pin TAKRAM_REVISION")
    takram = VIEWER_PATH / "three-geospatial"
    if not (takram / ".git").exists():
        return f"{takram} is not a git checkout."
    revision = await _git(takram, "rev-parse", "HEAD")
    if revision != match.group(1):
        return f"{takram} is at {revision[:7]}, not the pinned {match.group(1)[:7]}."
    for package in TAKRAM_PACKAGES:
        if not (takram / "packages" / package / "src" / "index.ts").is_file():
            return f"{takram} is incomplete: packages/{package}/src/index.ts is missing."
    if not (VIEWER_PATH / "node_modules").is_dir():
        return f"{VIEWER_PATH / 'node_modules'} does not exist."
    return None


async def _terrain_status() -> dict:
    """{ok, state, url}: every part terrain_start brings up is running and /health answers"""
    # Imported here: Terrain is its own checkout and may not exist yet
    from dynamic_functions.Terrain.Asset import database as asset
    from dynamic_functions.Terrain.Database import database
    from dynamic_functions.Terrain.Server import server

    parts = {"terrain database": database.status(),
             "asset catalog": asset.status(),
             "viewer server": server.status()}
    stopped = [name for name, part in parts.items() if not part["running"]]
    if stopped:
        return {"ok": False, "state": f"not running: {', '.join(stopped)}.", "url": None}
    url = parts["viewer server"]["url"]
    try:
        http_status, body = await _http_get(f"{url}/health")
    except urllib.error.URLError as exc:
        return {"ok": False, "state": f"{url}/health is not answering: {exc.reason}", "url": url}
    if not 200 <= http_status < 300 or json.loads(body).get("status") != "healthy":
        return {"ok": False, "state": f"{url}/health returned HTTP {http_status}: {body[:200]!r}", "url": url}
    return {"ok": True, "state": f"the database, asset catalog and server are running; {url}/health is healthy.",
            "url": url}


async def _viewer_status() -> dict:
    """{ok, state, pid, ours}: the viewer serves its page on VIEWER_PORT and proxies to Terrain"""
    pid = await _port_listener(VIEWER_PORT)
    if pid is None:
        return {"ok": False, "state": f"nothing is listening on port {VIEWER_PORT}.", "pid": None, "ours": False}
    cwd = await _process_cwd(pid)
    ours = cwd == VIEWER_PATH.resolve()
    if not ours:
        return {"ok": False, "pid": pid, "ours": False,
                "state": f"port {VIEWER_PORT} is held by pid {pid} running in {cwd}, not {VIEWER_PATH}."}

    title = re.search(r"<title>.*?</title>", (VIEWER_PATH / "index.html").read_text(encoding="utf-8"))
    if not title:
        raise RuntimeError(f"{VIEWER_PATH / 'index.html'} has no <title> to check the viewer page by")
    try:
        http_status, body = await _http_get(VIEWER_URL)
        if not 200 <= http_status < 300 or title.group(0).encode() not in body:
            return {"ok": False, "pid": pid, "ours": True,
                    "state": f"{VIEWER_URL} returned HTTP {http_status} without the viewer page."}
        # Through Vite's /api proxy, so this proves the proxy reaches Terrain
        http_status, body = await _http_get(f"{VIEWER_URL}api/demand-status")
    except urllib.error.URLError as exc:
        return {"ok": False, "pid": pid, "ours": True, "state": f"{VIEWER_URL} is not answering: {exc.reason}"}
    if not 200 <= http_status < 300:
        return {"ok": False, "pid": pid, "ours": True,
                "state": f"the viewer's /api proxy returned HTTP {http_status}; it is not reaching Terrain."}
    return {"ok": True, "pid": pid, "ours": True,
            "state": f"{VIEWER_URL} serves the viewer (pid {pid}) and its /api proxy reaches Terrain."}


async def _http_get(url: str) -> tuple[int, bytes]:
    """(status, body) for any HTTP answer; URLError when nothing answers"""
    def get() -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            # An HTTP error status is still an answer
            return exc.code, exc.read()

    return await asyncio.to_thread(get)


async def _port_listener(port: int) -> int | None:
    proc = await asyncio.create_subprocess_exec(
        "lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    # lsof exits 1 with no output when nothing matches
    if proc.returncode == 1 and not out.strip() and not err.strip():
        return None
    if proc.returncode != 0:
        raise RuntimeError(f"lsof for port {port} failed (exit {proc.returncode}): {err.decode().strip()}")
    pids = sorted({int(line) for line in out.decode().split()})
    if len(pids) != 1:
        raise RuntimeError(f"port {port} has {len(pids)} listening processes: {pids}")
    return pids[0]


async def _process_cwd(pid: int) -> Path:
    proc = await asyncio.create_subprocess_exec(
        "lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"lsof for pid {pid}'s cwd failed (exit {proc.returncode}): {err.decode().strip()}")
    names = [line[1:] for line in out.decode().splitlines() if line.startswith("n")]
    if len(names) != 1:
        raise RuntimeError(f"lsof reported {len(names)} working directories for pid {pid}")
    return Path(names[0]).resolve()


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
        if status["repository"] == VIEWER_REPO:
            return "call install_3d_viewer; it clones the viewer for you."
        return f"have the user run: git clone {status['repository']} {status['path']}"
    return "do not change that folder; tell the user what you found and escalate."


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
