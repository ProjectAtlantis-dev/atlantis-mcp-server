"""Bot tools"""

import base64
import json
import os
from datetime import datetime
from typing import List, Dict

from dynamic_functions.Home.common import _bots_dir, _ensure_thumb


@visible
async def bot_spawn(game_key: str, sid: str, role: str, location: str = "") -> None:
    """Spawn a bot and place it at a location"""
    from dynamic_functions.Home.character import character_set
    from dynamic_functions.Home.common import _load_bot_config, _available_bot_sids
    from dynamic_functions.Home.location import character_move, get_positions
    if _load_bot_config(sid) is None:
        raise ValueError(f"Unknown bot sid: {sid!r}. Must match a bot in Bots/ (e.g. {_available_bot_sids()})")
    await character_set(game_key, sid, role)
    if get_positions(game_key).get(sid) is None:
        await character_move(game_key, location or "", sid=sid)


@visible
async def bot_list() -> List[Dict[str, str]]:
    """List bots"""
    bots_dir = _bots_dir()
    bots: List[Dict[str, str]] = []
    if not os.path.isdir(bots_dir):
        return bots
    for entry in sorted(os.listdir(bots_dir)):
        entry_dir = os.path.join(bots_dir, entry)
        if not os.path.isdir(entry_dir) or entry.startswith(".") or entry == "__pycache__":
            continue
        config_path = os.path.join(entry_dir, 'config.json')
        if not os.path.isfile(config_path):
            continue
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        sid = cfg.get('sid', entry.lower())

        image_data = ''
        image_file = cfg.get('image', '')
        if image_file:
            image_path = os.path.join(entry_dir, image_file)
            if os.path.isfile(image_path):
                thumb = _ensure_thumb(image_path)
                if thumb:
                    ext = os.path.splitext(thumb)[1].lower().lstrip('.')
                    mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext, 'jpeg')
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
            'sid': sid,
            'displayName': cfg.get('displayName', entry),
            'model': model_label,
            'image': image_data,
            'updated': datetime.fromtimestamp(latest).strftime('%Y-%m-%d %H:%M'),
        })
    return bots
