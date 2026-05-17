"""Game state tools"""

import atlantis
import json
import os
import uuid
from typing import Dict, Any

from dynamic_functions.Home.location import _location_rows
from dynamic_functions.Home.character import _character_rows
from dynamic_functions.Home.modal import modal_string
from dynamic_functions.Home.role import _role_rows
from dynamic_functions.Home.bot import _bot_rows



@button("New Game")
@public
async def game_button() -> Dict[str, Any]:
    """Create a new game session"""
    settings = await atlantis.client_command("@game_new")
    #await atlantis.client_command("cursor join")
    return settings


@public
async def game_new() -> Dict[str, Any]:
    """Create a new game session"""
    from dynamic_functions.Home.common import create_game_dir, game_dir, _write_json

    for _ in range(10):
        game_key = uuid.uuid4().hex
        data_dir = game_dir(game_key)
        if not os.path.exists(data_dir):
            break
    else:
        raise RuntimeError("Unable to allocate a unique game_key")

    data_dir = create_game_dir(game_key)
    join_password = uuid.uuid4().hex
    _write_json(os.path.join(data_dir, 'game.json'), {
        'key': game_key,
        'join_password': join_password,
        'owner': atlantis.get_caller() or '',
        'user_game_id': atlantis.get_user_game_id(),
    })

    # Spawn the Receptionist (Kitty) at her role's default location.
    #await bot_spawn(game_key, 'kitty', 'Receptionist')

    # Register the chat callback bound to this game_key so it survives restart
    # via the boot-time re-registration scan.
    # await atlantis.client_command(f'/callback set chat chat_callback {game_key}')

    await atlantis.client_log(f"Game created: {game_key}")

    return {
        "game_key": game_key,
        "join_password": join_password,
    }


@visible
async def game_list() -> list:
    """List existing games, newest first"""
    from datetime import datetime
    from dynamic_functions.Home.common import home_path, _read_json
    games_root = home_path("Data", "games")
    if not os.path.isdir(games_root):
        return []
    entries = []
    for name in os.listdir(games_root):
        path = os.path.join(games_root, name)
        if not os.path.isdir(path):
            continue
        try:
            ts = os.path.getctime(path)
        except OSError:
            continue
        meta = _read_json(os.path.join(path, 'game.json')) or {}
        entries.append({
            "game_key": name,
            "user_game_id": meta.get("user_game_id"),
            "owner": meta.get("owner", ""),
            "created": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
            "_ts": ts,
        })
    entries.sort(key=lambda e: e["_ts"], reverse=True)
    for e in entries:
        del e["_ts"]
    return entries


@visible
async def game_status(game_key: str) -> dict:
    """Show current game status"""
    from dynamic_functions.Home.common import require_game_dir
    require_game_dir(game_key)
    return {"game_key": game_key}


@button("Join Game")
@public
async def game_join(game_key: str) -> None:
    """Join a game"""
    pass


@visible
async def game_show(game_key: str) -> None:
    """Show the game state diagram"""
    from dynamic_functions.Home.common import require_game_dir
    require_game_dir(game_key)

    bot_rows = _bot_rows()
    loc_rows = _location_rows()
    role_rows = _role_rows()
    char_rows = _character_rows(game_key)

    # Build an HTML table
    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    uid = uuid.uuid4().hex[:8]

    def _table(entity_id, title, headers, rows):
        """Build one entity table"""
        scoped_id = f"{entity_id}-{uid}"
        cls = f"er-entity-{uid}"
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

    tables = []
    tables.append(_table("ent-game", "GAME", ["key"], [[game_key]]))
    tables.append(_table("ent-bot", "BOT", ["sid", "displayName", "model", "updated"],
        [[b["sid"], b["displayName"], b["model"], b["updated"]] for b in bot_rows]))
    tables.append(_table("ent-location", "LOCATION", ["name", "displayName", "connects_to", "updated"],
        [[l["name"], l["displayName"], ", ".join(l["connects_to"]), l["updated"]] for l in loc_rows]))
    tables.append(_table("ent-role", "ROLE", ["name", "displayName", "defaultLocation", "systemPrompt", "updated"],
        [[r["name"], r["displayName"], r.get("defaultLocation", ""), "system_prompt.md" if r.get("systemPrompt") else "", r["updated"]] for r in role_rows]))
    tables.append(_table("ent-character", "CHARACTER", ["sid", "role", "displayName", "prompt", "location"],
        [[c["sid"], c["role"], c["displayName"], "prompt.md" if c.get("prompt") else "", c["location"]] for c in char_rows]))

    # Relationships
    relationships = [
        (f"ent-location-{uid}", f"ent-location-{uid}", "connects to"),
        (f"ent-location-{uid}", f"ent-role-{uid}", "defaultLocation"),
        (f"ent-bot-{uid}", f"ent-character-{uid}", "Characters/<sid>/<Role>/prompt.md"),
        (f"ent-role-{uid}", f"ent-character-{uid}", "Characters/<sid>/<Role>/prompt.md"),
        (f"ent-game-{uid}", f"ent-character-{uid}", "position"),
        (f"ent-location-{uid}", f"ent-character-{uid}", "location"),
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

    # Load ELK with client_script
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

    # Run the ELK layout
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
        f'      el.style.width = Math.round(n.width * scaleX) + "px";'
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
