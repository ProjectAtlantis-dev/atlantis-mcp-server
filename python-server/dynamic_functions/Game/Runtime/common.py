"""Shared helpers for game callbacks — bot config loading and spawning."""

import atlantis
import json
import logging
import os

logger = logging.getLogger("mcp_server")

BOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Bot", "Content")


# =========================================================================
# Bot config helpers
# =========================================================================

def _load_all_bot_configs():
    """Load all config.json files from Bot/Content/*/."""
    configs = []
    for entry in os.listdir(BOTS_DIR):
        config_path = os.path.join(BOTS_DIR, entry, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            cfg["_folder"] = entry
            configs.append(cfg)
    return configs


def _load_bot_config(bot_sid):
    """Find config.json for a bot by sid."""
    for entry in os.listdir(BOTS_DIR):
        config_path = os.path.join(BOTS_DIR, entry, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("sid") == bot_sid:
                return cfg, entry  # cfg + folder name
    return None, None


def pick_bot_for_location(location):
    """Pick a bot assigned to this location via config.json 'location' field.
    Returns config dict or None."""
    import random
    configs = _load_all_bot_configs()
    matches = [c for c in configs if c.get("location") == location]
    if not matches:
        return None
    return random.choice(matches)


async def spawn_bot(bot_sid):
    """Spawn a bot: show their face image and announce them."""
    cfg, folder = _load_bot_config(bot_sid)
    if not cfg:
        logger.warning(f"No config.json found for bot sid: {bot_sid}")
        return

    display_name = cfg.get("displayName", folder)

    # Look for a face image in the bot's content folder
    bot_dir = os.path.join(BOTS_DIR, folder)
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
