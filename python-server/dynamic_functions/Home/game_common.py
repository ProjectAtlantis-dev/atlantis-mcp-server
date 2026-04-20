"""Shared helpers for game callbacks — bot config loading and spawning."""

import atlantis
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from dynamic_functions.Data.main import game_dir

logger = logging.getLogger("mcp_server")

BOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "Bots")


def game_data_dir(game_id: Optional[str] = None, *, create: bool = True) -> str:
    """Return the data directory for the current game."""
    actual_game_id = game_id if game_id is not None else atlantis.get_game_id()
    if not actual_game_id:
        raise RuntimeError("game_data_dir requires an active game")
    return game_dir(actual_game_id, create=create)


def _load_bot_config(bot_sid: str, bots_dir: str = BOTS_DIR) -> Optional[Tuple[Dict[str, Any], str]]:
    """Find config.json for a bot by sid under bots_dir. Returns (config, folder_name) or None."""
    for entry in os.listdir(bots_dir):
        config_path = os.path.join(bots_dir, entry, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("sid") == bot_sid:
                cfg["_botDir"] = os.path.join(bots_dir, entry)
                return cfg, entry
    return None


async def spawn_bot(bot_sid: str, bots_dir: str = BOTS_DIR) -> None:
    """Spawn a bot: show their face image and announce them."""
    loaded = _load_bot_config(bot_sid, bots_dir)
    if not loaded:
        logger.warning(f"No config.json found for bot sid: {bot_sid}")
        return
    cfg, folder = loaded

    display_name = cfg.get("displayName", folder)

    bot_dir = os.path.join(bots_dir, folder)
    face_candidates = [f for f in os.listdir(bot_dir) if "face" in f.lower() and f.lower().endswith((".jpg", ".png", ".webp"))]
    if face_candidates:
        face_path = os.path.join(bot_dir, face_candidates[0])
        await atlantis.client_image(face_path)
        logger.info(f"Spawned {display_name}: showed face image")

    # Say hello as a chat message so the bot shows up in the transcript.
    greeting = cfg.get("greeting", f"Hi, I'm {display_name}.")
    stream_id = await atlantis.stream_start(bot_sid, display_name)
    await atlantis.stream(greeting, stream_id)
    await atlantis.stream_end(stream_id)
