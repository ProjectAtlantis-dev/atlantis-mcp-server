"""Bot tools"""

import atlantis
import base64
import json
import os
from datetime import datetime
from typing import List, Dict

from dynamic_functions.Home.common import _bots_dir, _ensure_thumb
from dynamic_functions.Home.prompt_common import load_persona, load_appearance




def _bot_rows() -> List[Dict[str, str]]:
    """Pure data: list bots. No client side effects."""
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
        sid = entry  # folder name is the sid

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
            'persona': load_persona(sid),
            'appearance': load_appearance(sid),
            'updated': datetime.fromtimestamp(latest).strftime('%Y-%m-%d %H:%M'),
        })
    return bots


@visible
async def bot_list() -> List[Dict[str, str]]:
    """List bots"""
    bots = _bot_rows()
    await atlantis.client_data("Bots", bots, column_formatter={
        "persona": {"type": "markdown"},
        "appearance": {"type": "markdown"},
    })
    return bots
