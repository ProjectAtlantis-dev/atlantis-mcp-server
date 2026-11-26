"""Client for TypeSafe's System One API (Jev), mirroring atlantis_nodejs
src/selftyped/Jev.ts: plain HTTP, answers keyed by the caller's question ids.

Chat owns it so every app can judge with it (Media's gallery search, a
Robot's behavior chooser). Configured in Chat/.env (gitignored; read on every
call, so edits apply without a restart):

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
JEV_ENV_PATH = Path(__file__).resolve().parent / ".env"


def typesafe_key() -> str | None:
    """Chat/.env's TYPESAFE_API_KEY, or None while it is missing or blank"""
    if not JEV_ENV_PATH.is_file():
        return None
    return (dotenv_values(JEV_ENV_PATH).get("TYPESAFE_API_KEY") or "").strip() or None


async def system_one(session: aiohttp.ClientSession, state, questions: dict) -> dict:
    """POST one System One request and return its answers by question id."""
    api_key = typesafe_key()
    if api_key is None:
        raise RuntimeError(f"TYPESAFE_API_KEY is not set in {JEV_ENV_PATH}")
    model = dotenv_values(JEV_ENV_PATH).get("JEV_MODEL") or "jev-latest"
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
