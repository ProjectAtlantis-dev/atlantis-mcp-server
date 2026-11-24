import asyncio
import hashlib
import json
import logging
import subprocess
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

logger = logging.getLogger("dynamic_function")

# Media's Robot sets Media up on this machine: image descriptions from a local
# vision model, and image search through TypeSafe. It has no @enter/@exit:
# Media is not a chat of its own. Setup's installer Robot asks for this
# status and loads Robot/Tools into Astrid's rounds (see Setup/Robot/main.py).

MEDIA_PATH = Path(__file__).absolute().parents[1]
MEDIA_ENV_PATH = MEDIA_PATH / ".env"
# python-server/runVision, which the user starts by hand next to runServer
RUN_VISION_PATH = Path(__file__).absolute().parents[3] / "runVision"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_PATH = Path.home() / "work" / "llama.cpp"
LLAMA_SERVER_PATH = LLAMA_CPP_PATH / "build" / "bin" / "llama-server"
VISION_MODEL_REPO = "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main"
VISION_MODEL_PATH = Path.home() / "work" / "models" / "Qwen3-VL-8B-Instruct-GGUF"
# file name -> approximate bytes, for download progress
VISION_MODEL_FILES = {
    "Qwen3VL-8B-Instruct-Q8_0.gguf": 8_710_000_000,
    "mmproj-Qwen3VL-8B-Instruct-F16.gguf": 1_160_000_000,
}
VISION_DOWNLOAD_LOG_PATH = VISION_MODEL_PATH / "download.log"
CAPTION_SETTINGS = {"CAPTION_BASE_URL": "http://127.0.0.1:8093/v1", "CAPTION_MODEL": "qwen3-vl-8b"}
# jev.py reads TYPESAFE_API_KEY from Media/.env for gallery search
TYPESAFE_KEYS_URL = "https://console.typesafe.ai/keys"
# Lists the account's models: proves the key without spending a System One request
TYPESAFE_MODELS_URL = "https://api.typesafe.ai/v1/models"
HTTP_TIMEOUT_SECONDS = 10
# Machine facts too slow or costly to re-prove every round. Setup describes
# this machine, so this is global, not per game.
CHECKS_PATH = Path(__file__).absolute().parent / "checks.json"


@visible
@index
async def index():
    """Robot info for Media setup"""
    return "Media robot"


@visible
async def get_status(command_prefix: str, current_folder: str):
    """
    Media setup status: every step with whether it is done, and exactly what
    to do next. Setup's installer runs this every bot round, so it only reads.
    """
    logger.info("Executing Media get_status...")
    if command_prefix not in ("", "/"):
        raise ValueError(f"command_prefix must be '' or '/', got {command_prefix!r}")

    steps: list[dict] = []
    for name, needs, check in MEDIA_STEPS:
        waits_on = next((number for number in needs if not steps[number - 1]["done"]), None)
        if waits_on is not None:
            steps.append({"name": name, "done": False, "waitsOn": waits_on, "fix": None,
                          "state": f"needs step {waits_on} ({MEDIA_STEPS[waits_on - 1][0]}) first."})
            continue
        result = await check()
        steps.append({"name": name, "done": result["fix"] is None, "waitsOn": None, **result})

    current = next((number for number, step in enumerate(steps, 1) if not step["done"]), None)
    lines = [f"Media has {len(steps)} steps:"]
    for number, step in enumerate(steps, 1):
        if step["done"]:
            label = "done"
        elif number == current:
            label = "NEXT"
        elif step["waitsOn"]:
            label = f"waiting on step {step['waitsOn']}"
        else:
            label = "to do"
        lines.append(f"{number}. [{label}] {step['name']}: {step['state']}")
    if current is None:
        lines.append("Media is set up.")
    else:
        lines.append(f"Next (Media step {current}): {steps[current - 1]['fix']}")
    return "\n".join(lines)


async def _check_caption_config() -> dict:
    if not caption_configured():
        return {"state": "image descriptions are not configured.", "fix": (
            "tell the user only this: we are setting up Media, which means installing a local vision "
            "model. Then call configure_image_descriptions.")}
    return {"state": "image descriptions use the local vision model.", "fix": None}


async def _check_llama_cpp() -> dict:
    if LLAMA_SERVER_PATH.is_file():
        return {"state": f"llama.cpp is built: {LLAMA_SERVER_PATH}.", "fix": None}
    return {"state": f"llama.cpp is not built ({LLAMA_SERVER_PATH} does not exist).", "fix": (
        "call install_llama_cpp; it clones and builds llama.cpp and takes about five minutes, so "
        "tell the user before you call it.")}


async def _check_vision_model() -> dict:
    state = vision_download_state()
    if state["done"]:
        return {"state": state["state"], "fix": None}
    if state["pid"] is not None:
        return {"state": state["state"], "fix": (
            "the download runs on its own. Tell the user how far it is and carry on with steps that "
            "do not wait on it; this status shows its progress every round.")}
    return {"state": state["state"], "fix": (
        "call download_vision_model; it starts a 10 GB download in the background and returns at once.")}


async def _check_caption_server() -> dict:
    base_url = dotenv_values(MEDIA_ENV_PATH)["CAPTION_BASE_URL"]
    try:
        status, _ = await _http_get(base_url.removesuffix("/v1") + "/health")
    except urllib.error.URLError:
        return {"state": f"the vision server is not running at {base_url}.", "fix": (
            f"have the user open a new terminal and run {RUN_VISION_PATH}, the same way they run "
            f"runServer, and leave it running. It is ready when it prints that the server is "
            f"listening. End your turn and wait until they say it is running.")}
    if status != 200:
        return {"state": f"the vision server at {base_url} is still loading its model (HTTP {status}).",
                "fix": "tell the user it is still loading; this status shows when it is ready."}
    return {"state": f"the vision server is running at {base_url}.", "fix": None}


async def _check_typesafe_key() -> dict:
    key = typesafe_key()
    if key is None:
        return {"state": f"{MEDIA_ENV_PATH} has no TYPESAFE_API_KEY.", "fix": (
            f"gallery search uses Jev, TypeSafe's judgment model, to match images to what the user "
            f"asks for. The user needs a TypeSafe API key. Call open_typesafe_console to open "
            f"{TYPESAFE_KEYS_URL} in a browser tab, where they sign in and create a key. Then have "
            f"them add TYPESAFE_API_KEY=<their key> to {MEDIA_ENV_PATH} themselves. Tell them never "
            f"to paste the key into the chat. End your turn and wait until they say it is done.")}
    checked = read_json(CHECKS_PATH).get("typesafeKey")
    if checked is None or checked["digest"] != key_digest(key):
        return {"state": f"{MEDIA_ENV_PATH} has a TYPESAFE_API_KEY, but this key has not been checked.",
                "fix": "call check_typesafe_key; it lists the account's TypeSafe models."}
    return {"state": f"the TypeSafe key worked on {checked['checkedAt']}.", "fix": None}


MEDIA_STEPS = [
    # (name, steps that must be done first, check)
    ("set up image descriptions", (), _check_caption_config),
    ("install llama.cpp", (1,), _check_llama_cpp),
    ("download the vision model", (1,), _check_vision_model),
    ("start image descriptions", (2, 3), _check_caption_server),
    ("set up image search", (), _check_typesafe_key),
]


# ---------------------------------------------------------------------------
# Shared with Robot/Tools
# ---------------------------------------------------------------------------

def caption_configured() -> bool:
    """True once Media/.env points at the local vision server"""
    if not MEDIA_ENV_PATH.is_file():
        return False
    base_url = dotenv_values(MEDIA_ENV_PATH).get("CAPTION_BASE_URL")
    if not base_url:
        return False
    if base_url != CAPTION_SETTINGS["CAPTION_BASE_URL"]:
        raise RuntimeError(f"{MEDIA_ENV_PATH} sets CAPTION_BASE_URL={base_url}, which Media setup does not know")
    return True


def vision_download_state() -> dict:
    present = [name for name in VISION_MODEL_FILES if (VISION_MODEL_PATH / name).is_file()]
    if len(present) == len(VISION_MODEL_FILES):
        return {"done": True, "pid": None, "state": f"the vision model is in {VISION_MODEL_PATH}."}
    proc = subprocess.run(["pgrep", "-f", f"curl.*{VISION_MODEL_REPO}"], capture_output=True, text=True)
    pids = proc.stdout.split()
    have = sum((VISION_MODEL_PATH / name).stat().st_size for name in present)
    have += sum(part.stat().st_size for part in VISION_MODEL_PATH.glob("*.part")) if VISION_MODEL_PATH.is_dir() else 0
    total = sum(VISION_MODEL_FILES.values())
    progress = f"{have / 1e9:.1f} of about {total / 1e9:.1f} GB"
    if pids:
        return {"done": False, "pid": int(pids[0]), "state": f"downloading the vision model: {progress}."}
    if VISION_DOWNLOAD_LOG_PATH.is_file() and have:
        return {"done": False, "pid": None, "state": (
            f"the vision model download stopped at {progress}:\n{tail(VISION_DOWNLOAD_LOG_PATH.read_text(errors='replace'), 5)}")}
    return {"done": False, "pid": None, "state": f"the vision model is not downloaded ({VISION_MODEL_PATH})."}


def typesafe_key() -> str | None:
    """Media/.env's TYPESAFE_API_KEY, or None while it is missing or blank"""
    if not MEDIA_ENV_PATH.is_file():
        return None
    return (dotenv_values(MEDIA_ENV_PATH).get("TYPESAFE_API_KEY") or "").strip() or None


def key_digest(key: str) -> str:
    """Fingerprint of a key, so a replaced key gets checked again"""
    return hashlib.sha256(key.encode()).hexdigest()


async def open_tab(url: str) -> None:
    """Open url in the default browser through the OS opener.

    Not window.open from the page: that targets the tool's exec shell and runs
    without a user click, so browsers block it.
    """
    if not await asyncio.to_thread(webbrowser.open, url, 2):
        raise RuntimeError(f"No browser could be opened for {url}; give the user the link to click")


async def git(path: Path, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(path), *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {path} (exit {proc.returncode}): {err.decode().strip()}")
    return out.decode().strip()


def tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


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
