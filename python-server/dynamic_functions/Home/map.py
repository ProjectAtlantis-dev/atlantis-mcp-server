"""Location map tool — renders the full facility layout with parent containment."""

import atlantis
import base64
import json
import os
import uuid
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.location import (
    _load_location, _locations_dir, _child_locations, _is_leaf,
    location_thumb,
)


def _location_image_b64(loc_name: str) -> str:
    """Get a location thumbnail data URI"""
    thumb = location_thumb(loc_name)
    if not thumb:
        return ""
    ext = os.path.splitext(thumb)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
    with open(thumb, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _all_location_names() -> List[str]:
    loc_dir = _locations_dir()
    if not os.path.isdir(loc_dir):
        return []
    return sorted([
        entry for entry in os.listdir(loc_dir)
        if os.path.isfile(os.path.join(loc_dir, entry, "config.json"))
    ])


@visible
async def location_map() -> None:
    """Show the full facility map: containers wrap their child rooms; edges are adjacency (connects_to)."""
    all_names = _all_location_names()
    if not all_names:
        await atlantis.client_log("No locations defined.")
        return

    # Partition into leaves (standable rooms) and containers (groupings).
    leaves: List[str] = []
    containers: List[str] = []
    parent_of: Dict[str, str] = {}
    for name in all_names:
        loc = _load_location(name) or {}
        parent_of[name] = (loc.get("parent") or "")
        if _is_leaf(name):
            leaves.append(name)
        else:
            containers.append(name)

    # Build leaf node data for HTML rendering.
    leaf_nodes: List[Dict[str, Any]] = []
    for name in leaves:
        loc = _load_location(name) or {}
        leaf_nodes.append({
            "name": name,
            "displayName": loc.get("displayName", name),
            "image": _location_image_b64(name),
        })

    # Adjacency edges between leaves (skip dangling targets).
    edges: List[tuple] = []
    leaf_set = set(leaves)
    for name in leaves:
        loc = _load_location(name) or {}
        for neighbor in loc.get("connects_to", []):
            if neighbor in leaf_set and (neighbor, name) not in edges:
                edges.append((name, neighbor))

    # Edges must be attached to the lowest-common-ancestor container of their
    # endpoints, not to the root graph. With elk.hierarchyHandling=INCLUDE_CHILDREN,
    # an edge declared on root is routed through root's coordinate space — so a
    # Lobby↔Hallway edge (both inside Atlantis) gets dragged out of the Atlantis
    # container and back in, producing long detours and bends. Grouping by LCA
    # keeps each edge local to the smallest container that contains both ends.
    # "" means root (no shared ancestor).
    def _ancestors(name: str) -> List[str]:
        chain: List[str] = []
        cur = parent_of.get(name) or ""
        while cur:
            chain.append(cur)
            cur = parent_of.get(cur) or ""
        return chain

    def _lca(a: str, b: str) -> str:
        anc_b = _ancestors(b)
        anc_b_set = set(anc_b)
        for x in _ancestors(a):
            if x in anc_b_set:
                return x
        return ""

    # lca_container_name -> list of (src_leaf, dst_leaf); "" key = root.
    edges_by_lca: Dict[str, List[tuple]] = {}
    for src, dst in edges:
        key = _lca(src, dst)
        edges_by_lca.setdefault(key, []).append((src, dst))

    uid = uuid.uuid4().hex[:8]

    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def _node_html(node: dict) -> str:
        name = node["name"]
        node_id = f"map-node-{name}-{uid}"
        img_html = (
            f'<img src="{node["image"]}" class="map-img-{uid}" />'
            if node["image"]
            else f'<div class="map-img-placeholder-{uid}">🗺️</div>'
        )
        label = _esc(node["displayName"])
        return (
            f'<div class="map-node-{uid}" id="{node_id}">'
            f'  {img_html}'
            f'  <div class="map-label-{uid}">{label}</div>'
            f'</div>'
        )

    nodes_html = "".join(_node_html(n) for n in leaf_nodes)

    # Serialize the hierarchy + edges for the layout script.
    hierarchy = {
        "containers": [
            {
                "id": f"map-group-{c}-{uid}",
                "name": c,
                "displayName": (_load_location(c) or {}).get("displayName", c),
                "parent": (f"map-group-{parent_of[c]}-{uid}" if parent_of.get(c) else ""),
                "children": [f"map-node-{n}-{uid}" for n in _child_locations(c) if n in leaf_set]
                            + [f"map-group-{n}-{uid}" for n in _child_locations(c) if n not in leaf_set],
            }
            for c in containers
        ],
        "rootLeaves": [f"map-node-{n}-{uid}" for n in leaves if not parent_of.get(n)],
    }
    hierarchy_json = json.dumps(hierarchy)

    def _edge_pair(src: str, dst: str) -> List[str]:
        return [f"map-node-{src}-{uid}", f"map-node-{dst}-{uid}"]

    # JS expects: { "<container-id-or-__root__>": [[srcId, dstId], ...] }
    edges_by_container = {
        (f"map-group-{k}-{uid}" if k else "__root__"): [_edge_pair(s, d) for s, d in v]
        for k, v in edges_by_lca.items()
    }
    edges_by_container_json = json.dumps(edges_by_container)

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
  #map-stage-{uid}.map-laid-out-{uid} .map-node-{uid},
  #map-stage-{uid}.map-laid-out-{uid} .map-group-{uid} {{
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
    z-index: 2;
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
    padding: 8px 10px 10px;
    color: #e0e0ff;
    font-weight: bold;
    font-size: 14px;
  }}

  .map-group-{uid} {{
    background: rgba(80, 60, 120, 0.10);
    border: 1.5px dashed #6a4c9a;
    border-radius: 14px;
    z-index: 1;
  }}
  .map-group-label-{uid} {{
    position: absolute;
    top: 6px;
    left: 12px;
    color: #b9a4d8;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: bold;
  }}

  #map-svg-{uid} {{
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
    overflow: visible;
    z-index: 3;
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

    # ELK hierarchical layout: build nested children, run layout, then place
    # leaf cards + render container backgrounds.
    layout_script = (
        f'(async function() {{'
        f'  await new Promise(function(r) {{ requestAnimationFrame(function() {{ requestAnimationFrame(r); }}); }});'
        f'  var uid = "{uid}";'
        f'  var hier = {hierarchy_json};'
        f'  var edgesByContainer = {edges_by_container_json};'
        f'  var edgeSeq = 0;'
        f'  function edgesFor(id) {{'
        f'    var pairs = edgesByContainer[id] || [];'
        f'    return pairs.map(function(p) {{'
        f'      return {{ id: "me" + (edgeSeq++), sources: [p[0]], targets: [p[1]] }};'
        f'    }});'
        f'  }}'
        f'  var stage = document.getElementById("map-stage-" + uid);'
        f'  var svg = document.getElementById("map-svg-" + uid);'
        f'  if (!stage || !svg) {{ console.error("[MAP] stage/svg not found"); return; }}'
        f'  if (!window.ELK) {{ console.error("[MAP] ELK not loaded"); return; }}'
        f'  var SVG_NS = "http://www.w3.org/2000/svg";'
        # Measure leaf cards
        f'  var leafSizes = {{}};'
        f'  stage.querySelectorAll(".map-node-{uid}").forEach(function(el) {{'
        f'    var r = el.getBoundingClientRect();'
        f'    leafSizes[el.id] = {{ width: Math.ceil(r.width), height: Math.ceil(r.height) }};'
        f'  }});'
        # Build container index
        f'  var containerById = {{}};'
        f'  hier.containers.forEach(function(c) {{ containerById[c.id] = c; }});'
        # Recursive node builder
        f'  function buildNode(id) {{'
        f'    if (id in leafSizes) {{'
        f'      return Object.assign({{ id: id }}, leafSizes[id]);'
        f'    }}'
        f'    var c = containerById[id];'
        f'    if (!c) return null;'
        f'    var children = c.children.map(buildNode).filter(function(x) {{ return x; }});'
        f'    return {{'
        f'      id: id,'
        f'      layoutOptions: {{'
        # INCLUDE_CHILDREN must be set on every container, not just root —
        # without it, child containers don't propagate cross-hierarchy routing
        # and edges crossing nested boundaries get mangled.
        f'        "elk.algorithm": "layered",'
        f'        "elk.direction": "RIGHT",'
        f'        "elk.hierarchyHandling": "INCLUDE_CHILDREN",'
        f'        "elk.padding": "[top=28,left=18,bottom=18,right=18]",'
        f'        "elk.spacing.nodeNode": "30"'
        f'      }},'
        f'      children: children,'
        f'      edges: edgesFor(id)'
        f'    }};'
        f'  }}'
        # Roots = top-level containers (no parent) + orphan leaves (no parent)
        f'  var rootChildren = [];'
        f'  hier.containers.forEach(function(c) {{'
        f'    if (!c.parent) {{ var n = buildNode(c.id); if (n) rootChildren.push(n); }}'
        f'  }});'
        f'  hier.rootLeaves.forEach(function(id) {{'
        f'    if (id in leafSizes) rootChildren.push(Object.assign({{ id: id }}, leafSizes[id]));'
        f'  }});'
        f'  var rootEdges = edgesFor("__root__");'
        f'  var graph = {{'
        f'    id: "root",'
        f'    layoutOptions: {{'
        f'      "elk.algorithm": "layered",'
        f'      "elk.direction": "RIGHT",'
        f'      "elk.hierarchyHandling": "INCLUDE_CHILDREN",'
        f'      "elk.edgeRouting": "ORTHOGONAL",'
        f'      "elk.spacing.nodeNode": "40",'
        f'      "elk.layered.spacing.nodeNodeBetweenLayers": "60"'
        f'    }},'
        f'    children: rootChildren,'
        f'    edges: rootEdges'
        f'  }};'
        f'  var elk = new ELK();'
        f'  elk.layout(graph).then(function(g) {{'
        f'    stage.classList.remove("map-measuring-{uid}");'
        f'    stage.classList.add("map-laid-out-{uid}");'
        f'    var W = Math.ceil(g.width) + 20;'
        f'    var H = Math.ceil(g.height) + 20;'
        # Walk hierarchy, accumulating absolute positions
        f'    function place(node, offX, offY) {{'
        f'      var ax = offX + (node.x || 0);'
        f'      var ay = offY + (node.y || 0);'
        f'      if (node.id in leafSizes) {{'
        f'        var el = document.getElementById(node.id);'
        f'        if (el) {{ el.style.left = ax + "px"; el.style.top = ay + "px"; }}'
        f'      }} else if (node.id in containerById) {{'
        # Render container background
        f'        var box = document.createElement("div");'
        f'        box.className = "map-group-{uid}";'
        f'        box.id = node.id;'
        f'        box.style.left = ax + "px";'
        f'        box.style.top = ay + "px";'
        f'        box.style.width = Math.ceil(node.width) + "px";'
        f'        box.style.height = Math.ceil(node.height) + "px";'
        f'        var lbl = document.createElement("div");'
        f'        lbl.className = "map-group-label-{uid}";'
        f'        lbl.textContent = containerById[node.id].displayName;'
        f'        box.appendChild(lbl);'
        f'        stage.appendChild(box);'
        f'      }}'
        f'      (node.children || []).forEach(function(ch) {{ place(ch, ax, ay); }});'
        f'    }}'
        f'    (g.children || []).forEach(function(n) {{ place(n, 0, 0); }});'
        f'    stage.style.width = W + "px";'
        f'    stage.style.height = H + "px";'
        f'    svg.setAttribute("width", W);'
        f'    svg.setAttribute("height", H);'
        f'    svg.setAttribute("viewBox", "0 0 " + W + " " + H);'
        f'    while (svg.firstChild) svg.removeChild(svg.firstChild);'
        # Draw edges (recursively flatten — edges live on the LCA container)
        f'    function drawEdges(node, offX, offY) {{'
        f'      var ax = offX + (node.x || 0);'
        f'      var ay = offY + (node.y || 0);'
        f'      (node.edges || []).forEach(function(e) {{'
        f'        (e.sections || []).forEach(function(sec) {{'
        f'          var pts = [sec.startPoint].concat(sec.bendPoints || []).concat([sec.endPoint]);'
        f'          var d = "M " + pts.map(function(p) {{ return (ax + p.x) + "," + (ay + p.y); }}).join(" L ");'
        f'          var path = document.createElementNS(SVG_NS, "path");'
        f'          path.setAttribute("d", d);'
        f'          path.setAttribute("fill", "none");'
        f'          path.setAttribute("stroke", "#f0c040");'
        f'          path.setAttribute("stroke-width", "2");'
        f'          path.setAttribute("opacity", "0.7");'
        f'          svg.appendChild(path);'
        f'        }});'
        f'      }});'
        f'      (node.children || []).forEach(function(ch) {{ drawEdges(ch, ax, ay); }});'
        f'    }}'
        f'    drawEdges(g, 0, 0);'
        f'  }}).catch(function(err) {{ console.error("[MAP] ELK layout failed", err); }});'
        f'}})()')

    await atlantis.client_script(layout_script)
