"""Text descriptions of uploaded images, kept in a JSON sidecar beside each image.

images/<digest>_<name>.png  ->  images/<digest>_<name>.png.json
    {"description": "...", "model": "...", "image": "<digest>_<name>.png"}

Images are named by content digest (see storage.save_upload), so a sidecar
never goes stale: new bytes mean a new file name and a new sidecar.

The describer is any OpenAI-compatible chat endpoint with vision, configured
in Media/.env (read on every call, so edits apply without a restart):

    CAPTION_BASE_URL=http://127.0.0.1:8093/v1     # llama.cpp runVision, or https://openrouter.ai/api/v1
    CAPTION_MODEL=qwen3-vl-8b                     # or an OpenRouter vision model id
    CAPTION_API_KEY_ENV=OPENROUTER_API_KEY        # optional: env var holding the key;
                                                  # omit for a local server that takes none

Uploads caption in the background (start_caption) so the upload returns at
once. A failed caption is logged with its traceback and leaves no sidecar,
so the next gallery view tries it again.
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI
from PIL import Image, ImageOps

logger = logging.getLogger("dynamic_function")

_MEDIA_ENV = Path(__file__).resolve().parent / ".env"

# Longest side sent to the model. Vision encoders resize anyway; this keeps a
# 250 MB upload from becoming a 330 MB base64 request.
CAPTION_MAX_SIDE = 1536
CAPTION_MAX_TOKENS = 400
CAPTION_PROMPT = (
    "Describe this image in a short paragraph so it can be found by search and "
    "understood without seeing it. Cover the main subjects, what they are doing, "
    "the setting, notable colors or style, and transcribe any legible text. "
    "State only what is visible; do not guess names or speculate."
)

# Strong references to running caption tasks (the event loop keeps only weak
# ones), keyed by image path so a path is never captioned twice at once.
_running: dict[Path, asyncio.Task] = {}


def sidecar_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.name + ".json")


def read_caption(image_path: Path) -> dict | None:
    """Return the image's sidecar, or None if it has not been captioned yet."""
    path = sidecar_path(image_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _settings() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) from Media/.env."""
    if not _MEDIA_ENV.is_file():
        raise RuntimeError(f"Image captioning needs {_MEDIA_ENV} with CAPTION_BASE_URL and CAPTION_MODEL")
    env = dotenv_values(_MEDIA_ENV)
    base_url = env.get("CAPTION_BASE_URL")
    model = env.get("CAPTION_MODEL")
    if not base_url or not model:
        raise RuntimeError(f"{_MEDIA_ENV} must set CAPTION_BASE_URL and CAPTION_MODEL")
    key_env = env.get("CAPTION_API_KEY_ENV")
    if key_env:
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"CAPTION_API_KEY_ENV names {key_env}, which is not set in the environment")
    else:
        # The OpenAI client insists on a key; a local llama.cpp server ignores it.
        api_key = "no-key"
    return base_url, model, api_key


def _jpeg_data_url(image_path: Path) -> str:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((CAPTION_MAX_SIDE, CAPTION_MAX_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


async def describe_image(image_path: Path) -> tuple[str, str]:
    """Ask the configured vision model about one image; return (description, model)."""
    base_url, model, api_key = _settings()
    data_url = await asyncio.to_thread(_jpeg_data_url, image_path)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=CAPTION_MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }],
    )
    choice = response.choices[0]
    description = (choice.message.content or "").strip()
    if not description:
        raise RuntimeError(f"{model} returned no description for {image_path.name} (finish_reason={choice.finish_reason!r})")
    return description, model


def _write_sidecar(image_path: Path, record: dict) -> None:
    """Write atomically, so a reader never sees half a sidecar."""
    path = sidecar_path(image_path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".caption_", suffix=".json", delete=False
    ) as temp:
        json.dump(record, temp, ensure_ascii=False, indent=2)
        temporary_path = Path(temp.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


async def caption_image(image_path: Path) -> dict:
    """Describe one image and store its sidecar; return the stored record."""
    description, model = await describe_image(image_path)
    record = {"description": description, "model": model, "image": image_path.name}
    _write_sidecar(image_path, record)
    logger.info(f"caption: {model} described {image_path.name} in {len(description)} chars")
    return record


def _finished(image_path: Path, task: asyncio.Task) -> None:
    _running.pop(image_path, None)
    if task.cancelled():
        logger.warning(f"caption: cancelled for {image_path}")
        return
    error = task.exception()
    if error is not None:
        logger.error(f"caption: failed for {image_path}", exc_info=error)


async def ensure_caption(image_path: Path) -> dict:
    """Return the image's sidecar, waiting on a running caption or making one now."""
    caption = read_caption(image_path)
    if caption is not None:
        return caption
    running = _running.get(image_path)
    if running is not None:
        return await asyncio.shield(running)
    return await caption_image(image_path)


def start_caption(image_path: Path) -> None:
    """Caption an image in the background unless it has a sidecar or is already running."""
    if image_path in _running or sidecar_path(image_path).is_file():
        return
    task = asyncio.create_task(caption_image(image_path), name=f"caption {image_path.name}")
    _running[image_path] = task
    task.add_done_callback(lambda done: _finished(image_path, done))
