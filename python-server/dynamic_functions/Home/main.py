import atlantis
import importlib
import json
import logging
import os
import uuid
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
async def game_show() -> None:
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

    uid = uuid.uuid4().hex[:8]

    def _table(entity_id, title, headers, rows, disabled=False):
        """Return HTML for one entity table."""
        scoped_id = f"{entity_id}-{uid}"
        cls = f"er-entity-{uid} er-disabled-{uid}" if disabled else f"er-entity-{uid}"
        h = "".join(f"<th>{_esc(c)}</th>" for c in headers)
        body = ""
        for row in rows:
            body += "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        if not rows:
            body = f'<tr><td colspan="{len(headers)}" style="color:#888;font-style:italic">empty</td></tr>'
        return (
            f'<div class="{cls}" id="{scoped_id}">'
            f'<div class="er-title-{uid}">{_esc(title)}</div>'
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
        (f"ent-game-{uid}", f"ent-bot-{uid}", "has"),
        (f"ent-game-{uid}", f"ent-location-{uid}", "has"),
        (f"ent-game-{uid}", f"ent-role-{uid}", "has"),
        (f"ent-location-{uid}", f"ent-location-{uid}", "connects to"),
        (f"ent-bot-{uid}", f"ent-character-{uid}", "sid"),
        (f"ent-role-{uid}", f"ent-character-{uid}", "role"),
        (f"ent-character-{uid}", f"ent-position-{uid}", "sid"),
        (f"ent-location-{uid}", f"ent-position-{uid}", "location"),
    ]

    rels_json = json.dumps(relationships)

    html = f"""
<style>
  #er-wrapper-{uid} {{
    position: relative;
    padding: 24px;
    width: 100%;
    box-sizing: border-box;
  }}
  #er-wrapper-{uid} .er-entity-{uid} {{
    min-width: 150px;
  }}
  #er-wrapper-{uid} #er-stage-{uid} {{
    position: relative;
  }}
  #er-wrapper-{uid} #er-stage-{uid}.er-measuring-{uid} {{
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: flex-start;
  }}
  #er-wrapper-{uid} #er-stage-{uid}.er-laid-out-{uid} .er-entity-{uid} {{
    position: absolute;
  }}
  #er-wrapper-{uid} .er-entity-{uid} {{
    background: #1e1e2e;
    border: 1px solid #555;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  #er-wrapper-{uid} .er-title-{uid} {{
    background: #3b3b5c;
    color: #e0e0ff;
    font-weight: bold;
    padding: 6px 10px;
    text-align: center;
    border-radius: 6px 6px 0 0;
    font-size: 13px;
    letter-spacing: 1px;
  }}
  #er-wrapper-{uid} .er-entity-{uid} table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    color: #ccc;
  }}
  #er-wrapper-{uid} .er-entity-{uid} th {{
    background: #2a2a40;
    color: #aaa;
    padding: 4px 8px;
    text-align: left;
    border-bottom: 1px solid #444;
    font-weight: normal;
    font-size: 11px;
  }}
  #er-wrapper-{uid} .er-entity-{uid} td {{
    padding: 3px 8px;
    border-bottom: 1px solid #333;
  }}
  #er-wrapper-{uid} .er-entity-{uid} tr:last-child td {{
    border-bottom: none;
  }}
  #er-wrapper-{uid} .er-disabled-{uid} {{
    opacity: 0.35;
  }}
  #er-wrapper-{uid} #er-svg-{uid} {{
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
    overflow: visible;
  }}
</style>
<div class="er-wrapper" id="er-wrapper-{uid}">
  <div id="er-stage-{uid}" class="er-measuring-{uid}">
    {''.join(tables)}
    <svg id="er-svg-{uid}" xmlns="http://www.w3.org/2000/svg"></svg>
  </div>
</div>
"""

    await atlantis.client_html(html)

    # Load ELK.js library via client_script (not via injected <script> tags)
    elk_loader = (
        'if (!window.ELK) {'
        '  var resolve; var p = new Promise(function(r) { resolve = r; });'
        '  var xhr = new XMLHttpRequest();'
        '  xhr.open("GET", "https://cdn.jsdelivr.net/npm/elkjs@0.9.3/lib/elk.bundled.js", true);'
        '  xhr.onload = function() {'
        '    if (xhr.status === 200) {'
        '      var _define = window.define;'
        '      try { window.define = undefined; (0, eval)(xhr.responseText); }'
        '      catch(e) { console.error("[ER] ELK eval failed", e); }'
        '      finally { window.define = _define; }'
        '    }'
        '    resolve();'
        '  };'
        '  xhr.onerror = function() { console.error("[ER] failed to fetch elkjs"); resolve(); };'
        '  xhr.send();'
        '  await p;'
        '}'
    )
    await atlantis.client_script(f'(async function() {{ {elk_loader} }})()')

    # Now run the layout logic — ELK is available on window
    layout_script = (
        f'(async function() {{'
        f'  await new Promise(function(r) {{ requestAnimationFrame(function() {{ requestAnimationFrame(r); }}); }});'
        f'  var uid = "{uid}";'
        f'  var rels = {rels_json};'
        f'  var stage = document.getElementById("er-stage-" + uid);'
        f'  var svg = document.getElementById("er-svg-" + uid);'
        f'  if (!stage || !svg) {{ console.error("[ER] stage/svg not found", uid); return; }}'
        f'  if (!window.ELK) {{ console.error("[ER] ELK not loaded"); return; }}'
        f'  var SVG_NS = "http://www.w3.org/2000/svg";'
        f'  var entities = stage.querySelectorAll(".er-entity-{uid}");'
        f'  var wrapper = document.getElementById("er-wrapper-" + uid);'
        f'  var availW = wrapper ? wrapper.clientWidth - 48 : 800;'
        f'  var nodes = [];'
        f'  entities.forEach(function(el) {{'
        f'    var r = el.getBoundingClientRect();'
        f'    nodes.push({{ id: el.id, width: Math.ceil(r.width), height: Math.ceil(r.height) }});'
        f'  }});'
        f'  var edges = rels.map(function(rel, i) {{'
        f'    return {{'
        f'      id: "e" + i,'
        f'      sources: [rel[0]],'
        f'      targets: [rel[1]],'
        f'      labels: [{{ text: rel[2], width: rel[2].length * 6 + 4, height: 12 }}]'
        f'    }};'
        f'  }});'
        f'  var graph = {{'
        f'    id: "root",'
        f'    layoutOptions: {{'
        f'      "elk.algorithm": "layered",'
        f'      "elk.direction": "DOWN",'
        f'      "elk.edgeRouting": "ORTHOGONAL",'
        f'      "elk.spacing.nodeNode": "60",'
        f'      "elk.spacing.edgeNode": "25",'
        f'      "elk.spacing.edgeEdge": "15",'
        f'      "elk.layered.spacing.nodeNodeBetweenLayers": "80",'
        f'      "elk.layered.spacing.edgeNodeBetweenLayers": "30",'
        f'      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",'
        f'      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES"'
        f'    }},'
        f'    children: nodes,'
        f'    edges: edges'
        f'  }};'
        f'  var elk = new ELK();'
        f'  elk.layout(graph).then(function(g) {{'
        f'    stage.classList.remove("er-measuring-{uid}");'
        f'    stage.classList.add("er-laid-out-{uid}");'
        f'    var W = Math.ceil(g.width) + 20;'
        f'    var H = Math.ceil(g.height) + 20;'
        f'    var useW = Math.max(W, availW);'
        f'    var scaleX = W > 0 ? useW / W : 1;'
        f'    g.children.forEach(function(n) {{'
        f'      var el = document.getElementById(n.id);'
        f'      if (!el) return;'
        f'      el.style.left = Math.round(n.x * scaleX) + "px";'
        f'      el.style.top = n.y + "px";'
        f'      el.style.width = n.width + "px";'
        f'    }});'
        f'    stage.style.width = useW + "px";'
        f'    stage.style.height = H + "px";'
        f'    svg.setAttribute("width", useW);'
        f'    svg.setAttribute("height", H);'
        f'    svg.setAttribute("viewBox", "0 0 " + useW + " " + H);'
        f'    while (svg.firstChild) svg.removeChild(svg.firstChild);'
        f'    (g.edges || []).forEach(function(e) {{'
        f'      (e.sections || []).forEach(function(sec) {{'
        f'        var pts = [sec.startPoint].concat(sec.bendPoints || []).concat([sec.endPoint]);'
        f'        var d = "M " + pts.map(function(p) {{ return Math.round(p.x * scaleX) + "," + p.y; }}).join(" L ");'
        f'        var path = document.createElementNS(SVG_NS, "path");'
        f'        path.setAttribute("d", d);'
        f'        path.setAttribute("fill", "none");'
        f'        path.setAttribute("stroke", "#888");'
        f'        path.setAttribute("stroke-width", "1.5");'
        f'        svg.appendChild(path);'
        f'      }});'
        f'      (e.labels || []).forEach(function(lbl) {{'
        f'        var bg = document.createElementNS(SVG_NS, "rect");'
        f'        bg.setAttribute("x", Math.round(lbl.x * scaleX) - 2);'
        f'        bg.setAttribute("y", lbl.y - 1);'
        f'        bg.setAttribute("width", lbl.width + 4);'
        f'        bg.setAttribute("height", lbl.height + 2);'
        f'        bg.setAttribute("fill", "#1e1e2e");'
        f'        bg.setAttribute("opacity", "0.85");'
        f'        svg.appendChild(bg);'
        f'        var t = document.createElementNS(SVG_NS, "text");'
        f'        t.setAttribute("x", Math.round(lbl.x * scaleX));'
        f'        t.setAttribute("y", lbl.y + lbl.height - 2);'
        f'        t.setAttribute("fill", "#aaa");'
        f'        t.setAttribute("font-size", "10");'
        f'        t.setAttribute("font-family", "sans-serif");'
        f'        t.textContent = lbl.text;'
        f'        svg.appendChild(t);'
        f'      }});'
        f'    }});'
        f'  }}).catch(function(err) {{ console.error("[ER] ELK layout failed", err); }});'
        f'}})()')

    await atlantis.client_script(layout_script)

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

