"""Setting Media up on this machine: image descriptions from a local vision
model, and image search through TypeSafe. The checks these act on live in
Media/Robot/main.py, whose status runs every round, so none of these needs a
status call after."""

import asyncio
import json
import logging
import shutil
import subprocess
import urllib.error
import urllib.request

import atlantis
from dotenv import set_key

from dynamic_functions.Chat.jev import JEV_ENV_PATH, typesafe_key
from dynamic_functions.Media.Robot.main import (
    CAPTION_SETTINGS,
    CHECKS_PATH,
    HTTP_TIMEOUT_SECONDS,
    LLAMA_CPP_PATH,
    LLAMA_CPP_REPO,
    LLAMA_SERVER_PATH,
    MEDIA_ENV_PATH,
    TYPESAFE_KEYS_URL,
    TYPESAFE_MODELS_URL,
    VISION_DOWNLOAD_LOG_PATH,
    VISION_MODEL_FILES,
    VISION_MODEL_PATH,
    VISION_MODEL_REPO,
    git,
    key_digest,
    now,
    open_tab,
    read_json,
    tail,
    vision_download_state,
    write_json,
)

logger = logging.getLogger("dynamic_function")


@visible
async def configure_image_descriptions():
    """
    Point Media at the local vision model (runVision) that gives uploaded
    images their text descriptions. Writes Media/.env.
    """
    logger.info("Executing configure_image_descriptions...")

    MEDIA_ENV_PATH.touch()
    # set_key edits only these keys; anything else in Media/.env stays.
    for key, value in CAPTION_SETTINGS.items():
        set_key(MEDIA_ENV_PATH, key, value, quote_mode="never")
    return f"Image descriptions now use the local vision model: wrote {MEDIA_ENV_PATH}."


@visible
async def install_llama_cpp():
    """
    Install llama.cpp for local image descriptions: clones it into ~/work/llama.cpp
    and builds llama-server. Takes about five minutes.
    """
    logger.info("Executing install_llama_cpp...")

    if LLAMA_SERVER_PATH.is_file():
        return f"llama.cpp is already built: {LLAMA_SERVER_PATH}."
    missing = [tool for tool in ("git", "cmake") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Building llama.cpp needs {' and '.join(missing)}; on a Mac: brew install {' '.join(missing)}")
    if not LLAMA_CPP_PATH.exists():
        await atlantis.client_log(f"Cloning {LLAMA_CPP_REPO} into {LLAMA_CPP_PATH}...")
        LLAMA_CPP_PATH.parent.mkdir(parents=True, exist_ok=True)
        await git(LLAMA_CPP_PATH.parent, "clone", "--depth", "1", LLAMA_CPP_REPO, str(LLAMA_CPP_PATH))
    elif not (LLAMA_CPP_PATH / "CMakeLists.txt").is_file():
        raise RuntimeError(f"{LLAMA_CPP_PATH} exists but is not a llama.cpp checkout. Not touching it.")

    await atlantis.client_log(f"Building llama-server in {LLAMA_CPP_PATH}; this takes about five minutes...")
    for args in (["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
                 ["cmake", "--build", "build", "--config", "Release", "-j", "--target", "llama-server"]):
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=str(LLAMA_CPP_PATH), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        output = out.decode(errors="replace").strip()
        logger.info(f"{' '.join(args)} exited {proc.returncode}:\n{tail(output)}")
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(args)} failed (exit {proc.returncode}):\n{tail(output)}")
    if not LLAMA_SERVER_PATH.is_file():
        raise RuntimeError(f"The build finished but {LLAMA_SERVER_PATH} does not exist")
    return f"Built {LLAMA_SERVER_PATH}."


@visible
async def download_vision_model():
    """
    Start downloading the Qwen3-VL-8B vision model (about 10 GB) into
    ~/work/models in the background and return at once. The Media status
    shows its progress every round; do not wait on it.
    """
    logger.info("Executing download_vision_model...")

    state = vision_download_state()
    if state["done"]:
        return f"The vision model is already downloaded to {VISION_MODEL_PATH}."
    if state["pid"] is not None:
        return f"The download is already running: {state['state']}"
    VISION_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    # Each file lands under .part and is renamed only when complete, so a file
    # with its final name is always whole. curl -C - resumes an interrupted .part.
    script = " && ".join(
        f"curl -fL --retry 5 -C - -o {name}.part {VISION_MODEL_REPO}/{name} && mv {name}.part {name}"
        for name in VISION_MODEL_FILES if not (VISION_MODEL_PATH / name).is_file()
    )
    log_file = VISION_DOWNLOAD_LOG_PATH.open("w", encoding="utf-8")
    try:
        # Its own session, so it outlives this call and a restart of this server
        subprocess.Popen(["sh", "-c", script], cwd=str(VISION_MODEL_PATH), stdout=log_file,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    finally:
        log_file.close()
    return f"Started downloading the vision model into {VISION_MODEL_PATH}; the Media status shows progress."


@visible
async def open_typesafe_console():
    """
    Open the TypeSafe console in a new browser tab, where the user signs in
    and creates the API key gallery search needs.
    """
    logger.info("Executing open_typesafe_console...")

    await open_tab(TYPESAFE_KEYS_URL)
    return f"Opened {TYPESAFE_KEYS_URL} in a new tab of the user's default browser."


@visible
async def check_typesafe_key():
    """
    Prove the TypeSafe API key in Chat/.env works by listing the account's
    models.
    """
    logger.info("Executing check_typesafe_key...")

    key = typesafe_key()
    if key is None:
        raise RuntimeError(f"Cannot check the key: {JEV_ENV_PATH} has no TYPESAFE_API_KEY")

    def get() -> tuple[int, bytes]:
        request = urllib.request.Request(TYPESAFE_MODELS_URL, headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    http_status, body = await asyncio.to_thread(get)
    if http_status == 401:
        raise RuntimeError(
            f"TypeSafe rejected the key (HTTP 401). Have the user check or replace TYPESAFE_API_KEY "
            f"in {JEV_ENV_PATH}.")
    if not 200 <= http_status < 300:
        raise RuntimeError(
            f"TypeSafe returned HTTP {http_status}: {body[:200]!r}. This is not proof the key is bad; "
            f"resolve this failure before continuing.")
    models = [model["name"] for model in json.loads(body)["models"]]

    checks = read_json(CHECKS_PATH)
    checks["typesafeKey"] = {"digest": key_digest(key), "checkedAt": now()}
    write_json(CHECKS_PATH, checks)
    return f"The TypeSafe key works; the account can use {', '.join(models)}."
