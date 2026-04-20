import atlantis
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
        face_files = [f for f in os.listdir(bot_dir) if 'face' in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
        if face_files:
            image_file = face_files[0]
            image_path = os.path.join(bot_dir, image_file)
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
async def position_list() -> Dict[str, str]:
    """Show current player positions for the active game."""
    game_id = atlantis.get_game_id()
    if not game_id:
        raise ValueError("No active game in context")
    return get_positions(game_id)


@visible
async def game_list() -> list[dict]:
    """List all currently active games with their game_id, caller, and tick info."""
    games = atlantis.get_active_games()
    return [
        {
            "game_id": gid,
            "caller": entry.get("caller", ""),
            "tick": str(entry["tick_callback"]) if entry.get("tick_callback") else "",
            "tick_busy": entry.get("tick_busy", False),
        }
        for gid, entry in games.items()
    ]
