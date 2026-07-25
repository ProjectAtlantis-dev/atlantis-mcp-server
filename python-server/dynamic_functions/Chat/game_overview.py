import atlantis
import json
import os
import uuid

from .bot import _bot_rows
from .camera import _camera_rows
from .common import _read_json
from .game import _game_roster_scene, require_membership
from .location import _location_rows
from .roster import _display_roster_rows, _load_game_roster
from .scene import _scene_rows


@visible
async def game_overview(game_key: str) -> None:
    data_dir = require_membership(game_key)

    meta = _read_json(os.path.join(data_dir, "game.json")) or {}
    roster_scene = _game_roster_scene(meta)
    bot_rows = _bot_rows()
    loc_rows = _location_rows()
    camera_rows = _camera_rows(game_key)
    scene_rows = _scene_rows()
    roster_path = os.path.join(data_dir, "roster.json")
    roster_rows = _display_roster_rows(_load_game_roster(game_key)) if os.path.isfile(roster_path) else []

    def _trunc(s, n=40):
        s = str(s or "")
        return s[:n] if len(s) > n else s

    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    uid = uuid.uuid4().hex[:8]

    def _table(entity_id, title, headers, rows, dynamic=False, row_classes=None, tone=""):
        scoped_id = f"{entity_id}-{uid}"
        cls = f"er-entity-{uid}"
        if dynamic:
            cls += f" er-dynamic-{uid}"
        if tone:
            cls += f" er-tone-{tone}-{uid}"
        h = "".join(f"<th>{_esc(c)}</th>" for c in headers)
        body = ""
        for i, row in enumerate(rows):
            rc = row_classes[i] if row_classes else ""
            tr = f'<tr class="{rc}">' if rc else "<tr>"
            body += tr + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        if not rows:
            body = f'<tr><td colspan="{len(headers)}" style="color:#888;font-style:italic">empty</td></tr>'
        return (
            f'<div class="{cls}" id="{scoped_id}">'
            f'<div class="er-title-{uid}">{_esc(title)}</div>'
            f'<table><tr>{h}</tr>{body}</table></div>'
        )

    tables = []
    tables.append(_table("ent-game", "GAME", ["key", "roster_scene"], [[game_key, roster_scene]], dynamic=True))
    tables.append(_table("ent-scene", "SCENE", ["name", "slots", "description"],
        [[s["name"], s["slots"], _trunc(s.get("description", ""))] for s in scene_rows]))
    tables.append(_table("ent-roster", "ROSTER", ["key", "bot_sid", "ai", "displayName", "sid", "location", "session_key", "bound_at", "spawned_at"],
        [[
            r.get("key", ""),
            r.get("bot_sid", ""),
            r.get("ai", ""),
            r.get("displayName", ""),
            r.get("sid", ""),
            r.get("location", ""),
            r.get("session_key", ""),
            r.get("bound_at", ""),
            r.get("spawned_at", ""),
        ] for r in roster_rows],
        dynamic=True,
        tone="green"))
    tables.append(_table("ent-bot", "BOT", ["sid", "displayName", "defaultLocation", "model"],
        [[b["sid"], b["displayName"], b["defaultLocation"], b.get("model", "")] for b in bot_rows]))
    tables.append(_table("ent-location", "LOCATION", ["name", "displayName", "parent", "connects_to", "description"],
        [[l["name"], l["displayName"], l.get("parent", ""), l["connects_to"], _trunc(l.get("description", ""))] for l in loc_rows],
        row_classes=["" if l["is_leaf"] else f"er-nonleaf-{uid}" for l in loc_rows]))
    tables.append(_table("ent-camera", "CAMERA", ["terminal", "location", "roster_slot"],
        [[c["terminal"], c["location"], c.get("roster_slot", "")] for c in camera_rows], dynamic=True))

    relationships = [
        (f"ent-game-{uid}", f"ent-roster-{uid}", "has roster"),
        (f"ent-game-{uid}", f"ent-camera-{uid}", "has cameras"),
        (f"ent-game-{uid}", f"ent-scene-{uid}", "roster_scene"),
        (f"ent-roster-{uid}", f"ent-bot-{uid}", "bot_sid"),
        (f"ent-roster-{uid}", f"ent-location-{uid}", "location"),
        (f"ent-location-{uid}", f"ent-location-{uid}", "connects to"),
        (f"ent-location-{uid}", f"ent-location-{uid}", "parent"),
        (f"ent-bot-{uid}", f"ent-location-{uid}", "defaultLocation"),
        (f"ent-location-{uid}", f"ent-camera-{uid}", "location"),
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
  #er-wrapper-{uid} .er-dynamic-{uid} {{
    background: #1e2e22;
    border-color: #4a7a5a;
  }}
  #er-wrapper-{uid} .er-dynamic-{uid} .er-title-{uid} {{
    background: #2f5c42;
    color: #d0ffe0;
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
  #er-wrapper-{uid} .er-dynamic-{uid} th {{
    background: #223a2c;
    color: #a8c8b0;
  }}
  #er-wrapper-{uid} .er-tone-green-{uid} {{
    background: #16281d;
    border-color: #3f8f58;
  }}
  #er-wrapper-{uid} .er-tone-green-{uid} .er-title-{uid} {{
    background: #27613b;
    color: #d8ffe1;
  }}
  #er-wrapper-{uid} .er-tone-green-{uid} th {{
    background: #203827;
    color: #a7d8b4;
  }}
  #er-wrapper-{uid} .er-tone-orange-{uid} {{
    background: #302114;
    border-color: #b36a2c;
  }}
  #er-wrapper-{uid} .er-tone-orange-{uid} .er-title-{uid} {{
    background: #8b4b1f;
    color: #ffe1c2;
  }}
  #er-wrapper-{uid} .er-tone-orange-{uid} th {{
    background: #442b18;
    color: #e5b083;
  }}
  #er-wrapper-{uid} .er-entity-{uid} td {{
    padding: 3px 8px;
    border-bottom: 1px solid #333;
    white-space: pre-line;
  }}
  #er-wrapper-{uid} .er-entity-{uid} tr:last-child td {{
    border-bottom: none;
  }}
  #er-wrapper-{uid} .er-nonleaf-{uid} td {{
    color: #666;
    font-style: italic;
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
