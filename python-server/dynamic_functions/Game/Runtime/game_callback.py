import atlantis
import json
import logging
import os

from dynamic_functions.Data.main import get_guest
from dynamic_functions.Data.todo import _write_store

logger = logging.getLogger("mcp_server")

LOCATION_BACKGROUNDS = {
    "AtlasLobby": os.path.join(os.path.dirname(__file__), "..", "Content", "AtlasLobby", "builder.jpg"),
    "KittyLobby": os.path.join(os.path.dirname(__file__), "..", "Content", "KittyLobby", "builder.jpg"),
}

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
    cfg = random.choice(matches)
    return cfg


async def _spawn_bot(bot_sid):
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
    if bot_sid == "atlas":
        greeting = "Hi, I'm Atlas. Welcome to FlowCentral."
    else:
        greeting = "*waves* Hey there~ Welcome to the lobby! 🐱"
    stream_id = await atlantis.stream_start(bot_sid, display_name)
    await atlantis.stream(greeting, stream_id)
    await atlantis.stream_end(stream_id)


@game
async def game_callback():
    """Initializes a new chat session at the player's persisted location."""

    try:
        user_id = atlantis.get_caller()
        if not user_id:
            raise ValueError("Game started without a caller identity")
        if not atlantis.get_game_id():
            raise RuntimeError("game callback fired without a game_id in context — nodejs side must send game_id")

        logger.info(f"Game started for user: {user_id}")

        # Fresh game = fresh todo list
        _write_store([], user_id)
        logger.info(f"Cleared todo store for {user_id}")

        # Check if we know this guest
        guest = get_guest(user_id)

        if guest:
            player_location = guest.get("location") or "AtlasLobby"
            if not player_location:
                player_location = "AtlasLobby"
            logger.info(f"Known guest {user_id}: location={player_location}")
        else:
            player_location = "AtlasLobby"
            logger.info(f"Unknown guest {user_id}: routing to AtlasLobby for check-in")

        await atlantis.client_command("/silent on")

        # Wire up callbacks — everything in same dir as game
        await atlantis.client_command("/callback set chat chat_callback")
        await atlantis.client_command("/callback set session session_callback")

    finally:
        await atlantis.client_command("/silent off")

    image_path = LOCATION_BACKGROUNDS.get(player_location)
    if not image_path:
        raise RuntimeError(f"No background configured for game location: {player_location}")
    await atlantis.set_background(image_path)

    # Always spawn Atlas at the player's location
    logger.info(f"Spawning Atlas at {player_location}")
    await _spawn_bot("atlas")
