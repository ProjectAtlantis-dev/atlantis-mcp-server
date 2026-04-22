"""Bot listing tools."""

import base64
import json
import os
from typing import List, Dict

from dynamic_functions.Home.common import _bots_dir, bot_thumb


@visible
async def bot_list() -> List[Dict[str, str]]:
    """List all bots with their name and image (base64-encoded)."""
    bots = []
    bots_dir = _bots_dir()
    for entry in sorted(os.listdir(bots_dir)):
        config_path = os.path.join(bots_dir, entry, 'config.json')
        if not os.path.isfile(config_path):
            continue
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        image_data = ''
        sid = cfg.get('sid', entry.lower())
        thumb = bot_thumb(sid)
        if thumb:
            ext = os.path.splitext(thumb)[1].lower()
            mime = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}.get(ext.lstrip('.'), 'jpeg')
            with open(thumb, 'rb') as img:
                b64 = base64.b64encode(img.read()).decode('ascii')
            image_data = f'data:image/{mime};base64,{b64}'
        # Grab first paragraph of system prompt
        blurb = ''
        prompt_file = cfg.get('systemPrompt', 'system_prompt.md')
        prompt_path = os.path.join(bots_dir, entry, prompt_file)
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
