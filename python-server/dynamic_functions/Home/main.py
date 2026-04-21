import atlantis
import importlib
import json
import logging
import os
from typing import List, Dict, Any

from dynamic_functions.Home.bot_common import logger, get_base_tools
from dynamic_functions.Data.main import get_positions
from dynamic_functions.Home.game_common import (
    _load_characters, _load_bot_config, _bots_dir, _locations_dir, GAMES_DIR,
)

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
    bots_dir = _bots_dir()
    for entry in sorted(os.listdir(bots_dir)):
        config_path = os.path.join(bots_dir, entry, 'config.json')
        if not os.path.isfile(config_path):
            continue
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        image_data = ''
        bot_dir = os.path.join(bots_dir, entry)
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
    locations_dir = _locations_dir()
    for fname in sorted(os.listdir(locations_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(locations_dir, fname), 'r') as f:
            data = json.load(f)
        image_data = ''
        image_file = data.get('image', '')
        if image_file:
            image_path = os.path.join(locations_dir, image_file)
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
async def game_set(name: str) -> None:
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
            return
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


def _get_move_module():
    """Return the move module for the current game."""
    game = _get_current_game()
    if not game:
        raise ValueError("No game set. Call game_set() first (e.g. game_set('Atlantis')).")
    mod_name = f"dynamic_functions.Games.{game}.move"
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError:
        raise ValueError(f"Game '{game}' has no move module (expected {mod_name})")


@visible
async def game_show() -> str:
    """Render a live ER diagram of game state as HTML tables with SVG connectors.

    If no game is set, shows BOT, LOCATION, GAME, and all ROLEs across all games.
    CHARACTER and POSITION require an active game.
    """
    game_name = _get_current_game()
    game_id = atlantis.get_game_id()

    # --- Gather data ---
    bot_rows = []
    if game_name:
        # Show bots for the current game
        bots_dir = _bots_dir()
        if os.path.isdir(bots_dir):
            for entry in sorted(os.listdir(bots_dir)):
                cfg_path = os.path.join(bots_dir, entry, "config.json")
                if not os.path.isfile(cfg_path):
                    continue
                with open(cfg_path) as f:
                    cfg = json.load(f)
                bot_rows.append({"game": game_name, "sid": cfg.get("sid", entry), "displayName": cfg.get("displayName", entry)})
    else:
        # No game set — show all bots across all games
        if os.path.isdir(GAMES_DIR):
            for gname in sorted(os.listdir(GAMES_DIR)):
                bots_dir = os.path.join(GAMES_DIR, gname, "Bots")
                if not os.path.isdir(bots_dir):
                    continue
                for entry in sorted(os.listdir(bots_dir)):
                    cfg_path = os.path.join(bots_dir, entry, "config.json")
                    if not os.path.isfile(cfg_path):
                        continue
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                    bot_rows.append({"game": gname, "sid": cfg.get("sid", entry), "displayName": cfg.get("displayName", entry)})

    loc_rows = []
    if game_name:
        locations_dir = _locations_dir()
        if os.path.isdir(locations_dir):
            for fname in sorted(os.listdir(locations_dir)):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(locations_dir, fname)) as f:
                    data = json.load(f)
                loc_rows.append({
                    "game": game_name,
                    "name": data.get("name", fname[:-5]),
                    "description": data.get("description", ""),
                    "connects_to": data.get("connects_to", []),
                })
    else:
        if os.path.isdir(GAMES_DIR):
            for gname in sorted(os.listdir(GAMES_DIR)):
                locations_dir = os.path.join(GAMES_DIR, gname, "Locations")
                if not os.path.isdir(locations_dir):
                    continue
                for fname in sorted(os.listdir(locations_dir)):
                    if not fname.endswith(".json"):
                        continue
                    with open(os.path.join(locations_dir, fname)) as f:
                        data = json.load(f)
                    loc_rows.append({
                        "game": gname,
                        "name": data.get("name", fname[:-5]),
                        "description": data.get("description", ""),
                        "connects_to": data.get("connects_to", []),
                    })

    # --- ROLE: if game set, show that game's roles; otherwise show all games' roles ---
    role_rows = []
    if game_name:
        roles_dir = os.path.join(GAMES_DIR, game_name, "Roles")
        if os.path.isdir(roles_dir):
            for rname in sorted(os.listdir(roles_dir)):
                rjson = os.path.join(roles_dir, rname, "role.json")
                if not os.path.isfile(rjson):
                    continue
                with open(rjson) as f:
                    rdata = json.load(f)
                role_rows.append({"game": game_name, "name": rname, "title": rdata.get("title", rname)})
    else:
        if os.path.isdir(GAMES_DIR):
            for gname in sorted(os.listdir(GAMES_DIR)):
                roles_dir = os.path.join(GAMES_DIR, gname, "Roles")
                if not os.path.isdir(roles_dir):
                    continue
                for rname in sorted(os.listdir(roles_dir)):
                    rjson = os.path.join(roles_dir, rname, "role.json")
                    if not os.path.isfile(rjson):
                        continue
                    with open(rjson) as f:
                        rdata = json.load(f)
                    role_rows.append({"game": gname, "name": rname, "title": rdata.get("title", rname)})

    # --- CHARACTER and POSITION: only available with an active game ---
    char_rows = []
    pos_rows = []
    if game_name and game_id:
        characters = _load_characters()
        for ch in characters:
            char_rows.append({
                "sid": ch["sid"],
                "role": ch.get("role", "?"),
                "isBot": ch.get("isBot", True),
                "humanName": ch.get("humanName", ""),
            })
        positions = get_positions(game_id)
        pos_rows = [{"sid": sid, "location": loc} for sid, loc in sorted(positions.items())]

    # --- Helper to build an HTML table ---
    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def _table(entity_id, title, headers, rows, disabled=False):
        """Return HTML for one entity table."""
        cls = "er-entity er-disabled" if disabled else "er-entity"
        h = "".join(f"<th>{_esc(c)}</th>" for c in headers)
        body = ""
        for row in rows:
            body += "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        if not rows:
            body = f'<tr><td colspan="{len(headers)}" style="color:#888;font-style:italic">empty</td></tr>'
        return (
            f'<div class="{cls}" id="{entity_id}">'
            f'<div class="er-title">{_esc(title)}</div>'
            f'<table><tr>{h}</tr>{body}</table></div>'
        )

    # --- Build tables ---
    # GAME: show current game or all available games
    if game_name:
        game_rows = [[game_name]]
    else:
        game_rows = []
        if os.path.isdir(GAMES_DIR):
            for entry in sorted(os.listdir(GAMES_DIR)):
                if os.path.isdir(os.path.join(GAMES_DIR, entry)) and not entry.startswith(('.', '_')):
                    game_rows.append([entry])
    tables = []
    tables.append(_table("ent-game", "GAME", ["name"], game_rows))
    tables.append(_table("ent-bot", "BOT", ["game", "sid", "displayName"],
        [[b["game"], b["sid"], b["displayName"]] for b in bot_rows]))
    tables.append(_table("ent-location", "LOCATION", ["game", "name", "description", "connects_to"],
        [[l["game"], l["name"], l["description"], ", ".join(l["connects_to"])] for l in loc_rows]))
    tables.append(_table("ent-role", "ROLE", ["game", "name", "title"],
        [[r["game"], r["name"], r["title"]] for r in role_rows]))
    no_game = not game_name
    tables.append(_table("ent-character", "CHARACTER", ["sid", "role", "isBot", "humanName"],
        [[c["sid"], c["role"], c["isBot"], c.get("humanName", "")] for c in char_rows],
        disabled=no_game))
    tables.append(_table("ent-position", "POSITION", ["sid", "location"],
        [[p["sid"], p["location"]] for p in pos_rows],
        disabled=no_game))

    # --- Relationships (from -> to, label) ---
    relationships = [
        ("ent-location", "ent-location", "connects to"),
        ("ent-game", "ent-role", "has"),
        ("ent-bot", "ent-character", "sid"),
        ("ent-role", "ent-character", "role"),
        ("ent-character", "ent-position", "sid"),
        ("ent-location", "ent-position", "location"),
    ]

    rels_json = json.dumps(relationships)

    html = f"""
<style>
  .er-container {{
    position: relative;
    display: flex;
    flex-wrap: wrap;
    gap: 32px;
    padding: 24px;
    align-items: flex-start;
  }}
  .er-entity {{
    background: #1e1e2e;
    border: 1px solid #555;
    border-radius: 6px;
    min-width: 160px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  .er-title {{
    background: #3b3b5c;
    color: #e0e0ff;
    font-weight: bold;
    padding: 6px 10px;
    text-align: center;
    border-radius: 6px 6px 0 0;
    font-size: 13px;
    letter-spacing: 1px;
  }}
  .er-entity table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    color: #ccc;
  }}
  .er-entity th {{
    background: #2a2a40;
    color: #aaa;
    padding: 4px 8px;
    text-align: left;
    border-bottom: 1px solid #444;
    font-weight: normal;
    font-size: 11px;
  }}
  .er-entity td {{
    padding: 3px 8px;
    border-bottom: 1px solid #333;
  }}
  .er-entity tr:last-child td {{
    border-bottom: none;
  }}
  .er-disabled {{
    opacity: 0.35;
  }}
  .er-svg {{
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
  }}
</style>
<div class="er-wrapper" style="position:relative">
  <svg class="er-svg" id="er-svg"></svg>
  <div class="er-container" id="er-container">
    {''.join(tables)}
  </div>
</div>
<script>
(function() {{
  const rels = {rels_json};
  const svg = document.getElementById('er-svg');
  const container = document.getElementById('er-container');

  function drawLines() {{
    const cRect = container.getBoundingClientRect();
    svg.setAttribute('width', container.scrollWidth);
    svg.setAttribute('height', container.scrollHeight);
    svg.innerHTML = '';

    rels.forEach(function(rel) {{
      const fromEl = document.getElementById(rel[0]);
      const toEl = document.getElementById(rel[1]);
      if (!fromEl || !toEl) return;

      const fRect = fromEl.getBoundingClientRect();
      const tRect = toEl.getBoundingClientRect();

      // Connector points: right-center of source, left-center of target
      let x1 = fRect.right - cRect.left;
      let y1 = fRect.top + fRect.height / 2 - cRect.top;
      let x2 = tRect.left - cRect.left;
      let y2 = tRect.top + tRect.height / 2 - cRect.top;

      // Self-referencing: curve below
      if (rel[0] === rel[1]) {{
        const cx = x1 + 40;
        const cy = Math.max(y1, y2) + 50;
        x2 = fRect.right - cRect.left;
        y2 = fRect.top + fRect.height * 0.75 - cRect.top;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', `M ${{x1}} ${{y1}} C ${{cx}} ${{cy}}, ${{cx}} ${{cy}}, ${{x2}} ${{y2}}`);
        path.setAttribute('stroke', '#888');
        path.setAttribute('stroke-width', '1.5');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke-dasharray', '6,3');
        svg.appendChild(path);
      }} else {{
        // If target is to the left, flip
        if (tRect.left < fRect.left) {{
          x1 = fRect.left - cRect.left;
          x2 = tRect.right - cRect.left;
        }}
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', '#888');
        line.setAttribute('stroke-width', '1.5');
        svg.appendChild(line);
      }}

      // Label at midpoint
      const lx = (x1 + x2) / 2;
      const ly = (y1 + y2) / 2 - 6;
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', lx);
      text.setAttribute('y', ly);
      text.setAttribute('fill', '#aaa');
      text.setAttribute('font-size', '10');
      text.setAttribute('text-anchor', 'middle');
      text.textContent = rel[2];
      svg.appendChild(text);
    }});
  }}

  // Draw after layout settles
  setTimeout(drawLines, 100);
  window.addEventListener('resize', drawLines);
}})();
</script>
"""

    await atlantis.client_html(html)
    if not game_name:
        await atlantis.client_log("Game not yet set")


@visible
async def game_move_bot(sid: str, location: str = "") -> str:
    """Move a bot character to a location in the active game.

    sid must be a registered bot character (via character_bot()).
    Location is optional for first-time entry (spawns in default lobby).

    Delegates to Games/{current_game}/move.move_bot().
    Call game_set() first to lock this server to a game.
    """
    mod = _get_move_module()
    return await mod.move_bot(sid, location or "")


@visible
async def game_move_human(sid: str, location: str = "") -> str:
    """Move a human character to a location in the active game.

    sid must be a registered human character (via character_human()).
    Location is optional for first-time entry (spawns in default lobby).

    Delegates to Games/{current_game}/move.move_human().
    Call game_set() first to lock this server to a game.
    """
    mod = _get_move_module()
    return await mod.move_human(sid, location or "")

