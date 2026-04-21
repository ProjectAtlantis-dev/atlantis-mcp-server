import atlantis
import importlib
import json
import logging
import os
from typing import List, Dict, Any

from dynamic_functions.Home.bot_common import logger, get_base_tools
from dynamic_functions.Data.main import get_positions

LOCATIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Locations')
BOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Bots')

def _require_game():
    if not atlantis.get_game_id():
        raise RuntimeError("No active game — this tool requires a running game session.")


@visible
async def index():
    """Multix CLI readme"""
    pass


@visible
async def show_tools() -> List[Dict[str, Any]]:
    """Show the current runtime tool inventory for this session."""
    tools, lookup = get_base_tools()
    simple: List[Dict[str, Any]] = []
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "any")
            if isinstance(ptype, list):
                ptype = ",".join(ptype)
            parts.append(f"{pname}:{ptype}")
        sig = ", ".join(parts)
        simple.append({
            "name": f"{fn['name']} ({sig})",
            "description": fn.get("description", ""),
        })
    logger.info(f"show_tools: {len(simple)} tools")
    return simple


@visible
async def bot_list() -> List[Dict[str, str]]:
    """List all bots with their name and image (base64-encoded)."""
    import base64
    bots = []
    for entry in sorted(os.listdir(BOTS_DIR)):
        config_path = os.path.join(BOTS_DIR, entry, 'config.json')
        if not os.path.isfile(config_path):
            continue
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        image_data = ''
        bot_dir = os.path.join(BOTS_DIR, entry)
        image_file = cfg.get('image', '')
        if image_file:
            image_path = os.path.join(bot_dir, image_file)
            if os.path.isfile(image_path):
                ext = os.path.splitext(image_file)[1].lower()
                mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext.lstrip('.'), 'jpeg')
                with open(image_path, 'rb') as img:
                    b64 = base64.b64encode(img.read()).decode('ascii')
                image_data = f'data:image/{mime};base64,{b64}'
        # Grab first paragraph of system prompt
        blurb = ''
        prompt_file = cfg.get('systemPrompt', 'system_prompt.md')
        prompt_path = os.path.join(bot_dir, prompt_file)
        if os.path.isfile(prompt_path):
            with open(prompt_path, 'r') as pf:
                text = pf.read().strip()
            # Grab first few paragraphs
            paras = text.split('\n\n')
            blurb = '\n\n'.join(paras[:3]).strip()
        bots.append({
            'sid': cfg.get('sid', entry.lower()),
            'name': cfg.get('displayName', entry),
            'image': image_data,
            'description': blurb,
        })
    return bots


@visible
async def location_list() -> List[Dict[str, str]]:
    """List all locations with their name and image (base64-encoded)."""
    import base64
    locations = []
    for fname in sorted(os.listdir(LOCATIONS_DIR)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(LOCATIONS_DIR, fname), 'r') as f:
            data = json.load(f)
        image_data = ''
        image_file = data.get('image', '')
        if image_file:
            image_path = os.path.join(LOCATIONS_DIR, image_file)
            if os.path.isfile(image_path):
                ext = os.path.splitext(image_file)[1].lower()
                mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext.lstrip('.'), 'jpeg')
                with open(image_path, 'rb') as img:
                    b64 = base64.b64encode(img.read()).decode('ascii')
                image_data = f'data:image/{mime};base64,{b64}'
        locations.append({
            'name': data.get('name', fname[:-5]),
            'description': data.get('description', data.get('name', fname[:-5])),
            'image': image_data,
        })
    return locations


@visible
async def position_list() -> List[Dict[str, str]]:
    """Show current player positions for the active game."""
    game = _get_current_game()
    if not game:
        raise ValueError("No game set. Call game_set() first.")
    game_id = atlantis.get_game_id()
    if not game_id:
        raise ValueError("No active game_id in context")
    positions = get_positions(game_id)
    return [{"sid": sid, "location": loc} for sid, loc in positions.items()]


GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Games')
_GAME_MAP_FILE = os.path.join(os.path.dirname(__file__), 'game_map.json')
_GAME_MAP_SHARED_KEY = 'game_map'


def _load_game_map() -> dict:
    """Return the full {game_id: game_name} map. Cached in server_shared; falls back to disk."""
    cached = atlantis.server_shared.get(_GAME_MAP_SHARED_KEY)
    if cached is not None:
        return cached
    if os.path.isfile(_GAME_MAP_FILE):
        with open(_GAME_MAP_FILE, 'r') as f:
            mapping = json.load(f)
    else:
        mapping = {}
    atlantis.server_shared.set(_GAME_MAP_SHARED_KEY, mapping)
    return mapping


def _save_game_map(mapping: dict) -> None:
    """Persist the game map to disk and cache."""
    with open(_GAME_MAP_FILE, 'w') as f:
        json.dump(mapping, f)
    atlantis.server_shared.set(_GAME_MAP_SHARED_KEY, mapping)


def _get_current_game() -> str:
    """Return the game name for the current game_id. Empty string if not set."""
    game_id = str(atlantis.get_game_id() or '')
    if not game_id:
        return ''
    return _load_game_map().get(game_id, '')


@visible
async def game_list() -> List[str]:
    """List available games in the Games folder."""
    games = []
    for entry in sorted(os.listdir(GAMES_DIR)):
        path = os.path.join(GAMES_DIR, entry)
        if os.path.isdir(path) and not entry.startswith(('.', '_')):
            games.append(entry)
    return games


@visible
async def game_status() -> dict:
    """Show current game lock status, including game_id and locked game folder."""
    game_id = str(atlantis.get_game_id() or '')
    game = _get_current_game()
    return {
        "game_id": game_id if game_id else None,
        "game_name": game if game else "NOT ASSIGNED",
    }


@visible
async def game_set(name: str) -> str:
    """Lock this MCP server to a specific game (e.g. 'Atlantis' or 'FlowCentral').

    The choice is persisted to disk and cached in server_shared so it
    survives restarts and never needs to be set again.
    """
    game_id = str(atlantis.get_game_id() or '')
    if not game_id:
        raise ValueError("No active game_id in context. Cannot set game without a game session.")

    # Check if this game_id is already mapped
    current = _get_current_game()
    if current:
        if current == name:
            await atlantis.client_log(f"Game already set to '{name}' (game_id: {game_id})")
            return name
        raise ValueError(f"Game ID {game_id} is already locked to '{current}'.")

    # Validate the game exists
    available = await game_list()
    if name not in available:
        raise ValueError(f"Unknown game '{name}'. Available: {available}")

    # Persist to disk and cache
    mapping = _load_game_map()
    mapping[game_id] = name
    _save_game_map(mapping)

    await atlantis.client_log(f"Game locked: game_id {game_id} → '{name}'")


@visible
async def game_move(location: str = "") -> None:
    """Move the current player to a location in the active game.

    Delegates to Games/{current_game}/move.move_to().
    Call game_set() first to lock this server to a game.
    """
    game = _get_current_game()
    if not game:
        raise ValueError("No game set. Call game_set() first (e.g. game_set('Atlantis')).")

    mod_name = f"dynamic_functions.Games.{game}.move"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        raise ValueError(f"Game '{game}' has no move module (expected {mod_name})")

    await mod.move_to(location)

