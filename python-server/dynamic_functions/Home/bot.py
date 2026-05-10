"""Bot tools"""

import base64
import json
import os
from datetime import datetime
from typing import List, Dict

from dynamic_functions.Home.common import _bots_dir, _load_bot_config, _ensure_thumb


def _bot_thumb_in(bot_sid: str, bots_dir: str) -> str:
    """Get a bot thumbnail path"""
    loaded = _load_bot_config(bot_sid, bots_dir)
    if not loaded:
        return ''
    cfg, folder = loaded
    image_file = cfg.get('image', '')
    if not image_file:
        return ''
    image_path = os.path.join(bots_dir, folder, image_file)
    if not os.path.isfile(image_path):
        return ''
    return _ensure_thumb(image_path)


def _collect_bots(bots_dir: str) -> List[Dict[str, str]]:
    """List bots from a directory"""
    bots = []
    if not os.path.isdir(bots_dir):
        return bots
    for entry in sorted(os.listdir(bots_dir)):
        entry_dir = os.path.join(bots_dir, entry)
        if not os.path.isdir(entry_dir):
            continue
        if entry.startswith(".") or entry == "__pycache__":
            continue
        config_path = os.path.join(entry_dir, 'config.json')
        if not os.path.isfile(config_path):
            continue
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        image_data = ''
        sid = cfg.get('sid', entry.lower())
        thumb = _bot_thumb_in(sid, bots_dir)
        if thumb:
            ext = os.path.splitext(thumb)[1].lower()
            mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext.lstrip('.'), 'jpeg')
            with open(thumb, 'rb') as img:
                b64 = base64.b64encode(img.read()).decode('ascii')
            image_data = f'data:image/{mime};base64,{b64}'
        provider = cfg.get('provider', '')
        model = cfg.get('model', '')
        model_label = f"{provider}: {model}" if provider and model else (model or provider)
        latest = max(
            (os.path.getmtime(os.path.join(entry_dir, f)) for f in os.listdir(entry_dir)
             if os.path.isfile(os.path.join(entry_dir, f))),
            default=os.path.getmtime(config_path),
        )
        bots.append({
            'sid': cfg.get('sid', entry.lower()),
            'displayName': cfg.get('displayName', entry),
            'model': model_label,
            'image': image_data,
            'updated': datetime.fromtimestamp(latest).strftime('%Y-%m-%d %H:%M'),
        })
    return bots


@visible
async def bot_spawn(game_key: str, sid: str, role: str, location: str = "") -> None:
    """Spawn a bot and place it at a location"""
    from dynamic_functions.Home.character import character_bot
    from dynamic_functions.Home.location import move_character, get_positions
    await character_bot(game_key, sid, role)
    if get_positions().get(sid) is None:
        await move_character(game_key, sid, location or "", is_bot=True)


@visible
async def bot_list() -> List[Dict[str, str]]:
    """List bots"""
    return _collect_bots(_bots_dir())
