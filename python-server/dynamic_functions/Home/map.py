"""Interactive location map — shows current location and adjacent locations with images."""

import atlantis
import base64
import json
import os
import uuid
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.common import (
    _load_location, _locations_dir, _connects_to,
    _load_bot_config, location_thumb,
)
from dynamic_functions.Home.character import _load_characters
from dynamic_functions.Data.main import get_player_position, get_players_at


def _require_game():
    if not atlantis.get_game_id():
        raise RuntimeError("No active game — this tool requires a running game session.")


def _location_image_b64(loc_name: str) -> str:
    """Return a data-URI for a location's thumbnail, or empty string.

    Uses disk-persisted thumbnails from common.location_thumb()
    so we never base64-encode multi-MB originals.
    """
    thumb = location_thumb(loc_name)
    if not thumb:
        return ""
    ext = os.path.splitext(thumb)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
    with open(thumb, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _character_display_name(ch: dict) -> str:
    if ch.get("isBot", True):
        loaded = _load_bot_config(ch["sid"])
        return loaded[0].get("displayName", ch["sid"]) if loaded else ch["sid"]
    return ch.get("humanName", ch["sid"])


def _characters_at(game_id: str, location: str) -> List[str]:
    """Return display names of characters at a location."""
    sids = get_players_at(game_id, location)
    if not sids:
        return []
    characters = _load_characters()
    names = []
    for ch in characters:
        if ch["sid"] in sids:
            names.append(_character_display_name(ch))
    return names


@visible
async def map(location: str = "") -> None:
    """Show a visual map of the current location and its immediately adjacent locations.

    Each location is rendered as a card with its image, name, and any characters present.
    The current location is highlighted. Requires an active game (call game_set() first).

    Args:
        location: Override which location to center the map on.
                  Defaults to the caller's current position.
    """
    _require_game()
    game_id = atlantis.get_game_id()

    # Determine center location
    if not location:
        sid = atlantis.get_caller()
        if sid:
            location = get_player_position(game_id, sid)
        if not location:
            raise ValueError(
                "Cannot determine current location. Either pass a location name "
                "or move your character first with go()."
            )

    center_loc = _load_location(location)
    if not center_loc:
        raise ValueError(f"Unknown location: {location}")

    # Gather adjacent locations
    adjacent_names = center_loc.get("connects_to", [])
    all_names = [location] + adjacent_names

    # Build node data
    nodes = []
    for loc_name in all_names:
        loc_data = _load_location(loc_name)
        if not loc_data:
            continue
        is_current = loc_name == location
        chars = _characters_at(game_id, loc_name)
        image_b64 = _location_image_b64(loc_name)
        nodes.append({
            "name": loc_name,
            "description": loc_data.get("description", loc_name),
            "image": image_b64,
            "is_current": is_current,
            "characters": chars,
            # Show which exits this adjacent location has (beyond back to center)
            "further_connects": [
                c for c in loc_data.get("connects_to", [])
                if c != location and c not in adjacent_names
            ],
        })

    # Build edges: center connects to each adjacent
    edges = []
    for adj_name in adjacent_names:
        edges.append((location, adj_name))

    # Also show edges between adjacent locations if they connect to each other
    for i, a in enumerate(adjacent_names):
        for b in adjacent_names[i + 1:]:
            a_connects = _connects_to(a)
            if b in a_connects:
                edges.append((a, b))

    uid = uuid.uuid4().hex[:8]

    # --- Build HTML ---
    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def _node_html(node: dict) -> str:
        name = node["name"]
        node_id = f"map-node-{name}-{uid}"
        is_current = node["is_current"]

        cls = f"map-node-{uid}"
        if is_current:
            cls += f" map-current-{uid}"

        # Image
        img_html = ""
        if node["image"]:
            img_html = f'<img src="{node["image"]}" class="map-img-{uid}" />'
        else:
            img_html = f'<div class="map-img-placeholder-{uid}">🗺️</div>'

        # Characters present
        chars_html = ""
        if node["characters"]:
            chars_list = ", ".join(_esc(c) for c in node["characters"])
            chars_html = f'<div class="map-chars-{uid}">👤 {chars_list}</div>'

        # Label
        label = _esc(node["description"])
        if is_current:
            label = f"📍 {label}"

        # Further exits hint
        further_html = ""
        if node["further_connects"]:
            dirs = ", ".join(_esc(c) for c in node["further_connects"])
            further_html = f'<div class="map-further-{uid}">→ {dirs}</div>'

        return (
            f'<div class="{cls}" id="{node_id}">'
            f'  {img_html}'
            f'  <div class="map-label-{uid}">{label}</div>'
            f'  {chars_html}'
            f'  {further_html}'
            f'</div>'
        )

    nodes_html = "".join(_node_html(n) for n in nodes)

    # Edges as JSON for the layout script
    edges_json = json.dumps([
        (f"map-node-{src}-{uid}", f"map-node-{dst}-{uid}")
        for src, dst in edges
    ])

    html = f"""
<style>
  #map-wrapper-{uid} {{
    position: relative;
    padding: 16px;
    width: 100%;
    box-sizing: border-box;
  }}
  #map-stage-{uid} {{
    position: relative;
  }}
  #map-stage-{uid}.map-measuring-{uid} {{
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: flex-start;
    justify-content: center;
  }}
  #map-stage-{uid}.map-laid-out-{uid} .map-node-{uid} {{
    position: absolute;
  }}

  .map-node-{uid} {{
    background: #1e1e2e;
    border: 2px solid #444;
    border-radius: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    overflow: hidden;
    width: 180px;
    text-align: center;
    transition: transform 0.2s;
  }}
  .map-current-{uid} {{
    border-color: #f0c040 !important;
    box-shadow: 0 0 20px rgba(240,192,64,0.5), 0 2px 12px rgba(0,0,0,0.4) !important;
  }}
  .map-img-{uid} {{
    width: 100%;
    height: 120px;
    object-fit: cover;
    display: block;
  }}
  .map-img-placeholder-{uid} {{
    width: 100%;
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 36px;
    background: #2a2a40;
  }}
  .map-label-{uid} {{
    padding: 8px 10px 4px;
    color: #e0e0ff;
    font-weight: bold;
    font-size: 14px;
  }}
  .map-current-{uid} .map-label-{uid} {{
    color: #f0c040;
  }}
  .map-chars-{uid} {{
    padding: 2px 10px 4px;
    color: #aac8aa;
    font-size: 11px;
  }}
  .map-further-{uid} {{
    padding: 2px 10px 8px;
    color: #888;
    font-size: 10px;
    font-style: italic;
  }}

  #map-svg-{uid} {{
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
    overflow: visible;
  }}
</style>
<div id="map-wrapper-{uid}">
  <div id="map-stage-{uid}" class="map-measuring-{uid}">
    {nodes_html}
    <svg id="map-svg-{uid}" xmlns="http://www.w3.org/2000/svg"></svg>
  </div>
</div>
"""

    await atlantis.client_html(html)

    # Load ELK
    elk_loader = (
        'if (!window.ELK) {'
        '  var resolve; var p = new Promise(function(r) { resolve = r; });'
        '  var xhr = new XMLHttpRequest();'
        '  xhr.open("GET", "https://cdn.jsdelivr.net/npm/elkjs@0.9.3/lib/elk.bundled.js", true);'
        '  xhr.onload = function() {'
        '    if (xhr.status === 200) {'
        '      var _define = window.define;'
        '      try { window.define = undefined; (0, eval)(xhr.responseText); }'
        '      catch(e) { console.error("[MAP] ELK eval failed", e); }'
        '      finally { window.define = _define; }'
        '    }'
        '    resolve();'
        '  };'
        '  xhr.onerror = function() { console.error("[MAP] failed to fetch elkjs"); resolve(); };'
        '  xhr.send();'
        '  await p;'
        '}'
    )
    await atlantis.client_script(f'(async function() {{ {elk_loader} }})()')

    # Layout script
    layout_script = (
        f'(async function() {{'
        f'  await new Promise(function(r) {{ requestAnimationFrame(function() {{ requestAnimationFrame(r); }}); }});'
        f'  var uid = "{uid}";'
        f'  var edgeData = {edges_json};'
        f'  var stage = document.getElementById("map-stage-" + uid);'
        f'  var svg = document.getElementById("map-svg-" + uid);'
        f'  if (!stage || !svg) {{ console.error("[MAP] stage/svg not found"); return; }}'
        f'  if (!window.ELK) {{ console.error("[MAP] ELK not loaded"); return; }}'
        f'  var SVG_NS = "http://www.w3.org/2000/svg";'
        f'  var cards = stage.querySelectorAll(".map-node-{uid}");'
        f'  var nodes = [];'
        f'  cards.forEach(function(el) {{'
        f'    var r = el.getBoundingClientRect();'
        f'    nodes.push({{ id: el.id, width: Math.ceil(r.width), height: Math.ceil(r.height) }});'
        f'  }});'
        f'  var edges = edgeData.map(function(pair, i) {{'
        f'    return {{ id: "me" + i, sources: [pair[0]], targets: [pair[1]] }};'
        f'  }});'
        f'  var graph = {{'
        f'    id: "root",'
        f'    layoutOptions: {{'
        f'      "elk.algorithm": "layered",'
        f'      "elk.direction": "RIGHT",'
        f'      "elk.edgeRouting": "ORTHOGONAL",'
        f'      "elk.spacing.nodeNode": "40",'
        f'      "elk.spacing.edgeNode": "20",'
        f'      "elk.spacing.edgeEdge": "15",'
        f'      "elk.layered.spacing.nodeNodeBetweenLayers": "60",'
        f'      "elk.layered.spacing.edgeNodeBetweenLayers": "25",'
        f'      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",'
        f'      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES"'
        f'    }},'
        f'    children: nodes,'
        f'    edges: edges'
        f'  }};'
        f'  var elk = new ELK();'
        f'  elk.layout(graph).then(function(g) {{'
        f'    stage.classList.remove("map-measuring-{uid}");'
        f'    stage.classList.add("map-laid-out-{uid}");'
        f'    var W = Math.ceil(g.width) + 20;'
        f'    var H = Math.ceil(g.height) + 20;'
        f'    g.children.forEach(function(n) {{'
        f'      var el = document.getElementById(n.id);'
        f'      if (!el) return;'
        f'      el.style.left = n.x + "px";'
        f'      el.style.top = n.y + "px";'
        f'    }});'
        f'    stage.style.width = W + "px";'
        f'    stage.style.height = H + "px";'
        f'    svg.setAttribute("width", W);'
        f'    svg.setAttribute("height", H);'
        f'    svg.setAttribute("viewBox", "0 0 " + W + " " + H);'
        f'    while (svg.firstChild) svg.removeChild(svg.firstChild);'
        # Draw edges
        f'    (g.edges || []).forEach(function(e) {{'
        f'      (e.sections || []).forEach(function(sec) {{'
        f'        var pts = [sec.startPoint].concat(sec.bendPoints || []).concat([sec.endPoint]);'
        f'        var d = "M " + pts.map(function(p) {{ return p.x + "," + p.y; }}).join(" L ");'
        f'        var path = document.createElementNS(SVG_NS, "path");'
        f'        path.setAttribute("d", d);'
        f'        path.setAttribute("fill", "none");'
        f'        path.setAttribute("stroke", "#f0c040");'
        f'        path.setAttribute("stroke-width", "2");'
        f'        path.setAttribute("stroke-dasharray", "6,4");'
        f'        path.setAttribute("opacity", "0.6");'
        f'        svg.appendChild(path);'
        # Arrowhead at endpoint
        f'        var last = pts[pts.length - 1];'
        f'        var prev = pts.length > 1 ? pts[pts.length - 2] : last;'
        f'        var angle = Math.atan2(last.y - prev.y, last.x - prev.x);'
        f'        var arrLen = 8;'
        f'        var arr = document.createElementNS(SVG_NS, "polygon");'
        f'        var ax = last.x, ay = last.y;'
        f'        var p1x = ax - arrLen * Math.cos(angle - 0.4);'
        f'        var p1y = ay - arrLen * Math.sin(angle - 0.4);'
        f'        var p2x = ax - arrLen * Math.cos(angle + 0.4);'
        f'        var p2y = ay - arrLen * Math.sin(angle + 0.4);'
        f'        arr.setAttribute("points", ax+","+ay+" "+p1x+","+p1y+" "+p2x+","+p2y);'
        f'        arr.setAttribute("fill", "#f0c040");'
        f'        arr.setAttribute("opacity", "0.6");'
        f'        svg.appendChild(arr);'
        f'      }});'
        f'    }});'
        f'  }}).catch(function(err) {{ console.error("[MAP] ELK layout failed", err); }});'
        f'}})()')

    await atlantis.client_script(layout_script)
    await atlantis.client_log(f"📍 Map centered on: {center_loc.get('description', location)}")
