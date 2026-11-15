"""Client for TypeSafe's System One API (Jev), mirroring atlantis_nodejs
src/selftyped/Jev.ts: plain HTTP, answers keyed by the caller's question ids.

Configured in Media/.env (gitignored; read on every call, so edits apply
without a restart):

    TYPESAFE_API_KEY=apikey_...
    JEV_MODEL=jev-latest      # optional
"""

import logging
import time
from pathlib import Path

import aiohttp
from dotenv import dotenv_values

logger = logging.getLogger("dynamic_function")

JEV_URL = "https://api.typesafe.ai/v1/systemone"
_MEDIA_ENV = Path(__file__).resolve().parent / ".env"


async def system_one(session: aiohttp.ClientSession, state, questions: dict) -> dict:
    """POST one System One request and return its answers by question id."""
    env = dotenv_values(_MEDIA_ENV) if _MEDIA_ENV.is_file() else {}
    api_key = env.get("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError(f"TYPESAFE_API_KEY is not set in {_MEDIA_ENV}")
    model = env.get("JEV_MODEL") or "jev-latest"
    start = time.monotonic()
    async with session.post(
        JEV_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "state": state, "questions": questions},
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"Jev request failed: HTTP {response.status} {await response.text()}")
        body = await response.json()
    usage = body.get("usage") or {}
    logger.info(
        f"Jev {body.get('model')}: {len(questions)} questions, {usage.get('input_tokens')} input tokens, "
        f"{(time.monotonic() - start) * 1000:.0f}ms"
    )
    return body["answers"]
