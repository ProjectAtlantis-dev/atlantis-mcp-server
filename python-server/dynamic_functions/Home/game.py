"""Game state management — list, status, set, and the ER diagram viewer."""

import atlantis
import html as html_lib
import json
import os
import shlex
import uuid
from typing import List, Dict, Any

from dynamic_functions.Home.common import GAMES_DIR
from dynamic_functions.Home.location import get_positions
from dynamic_functions.Home.character import _load_characters, role_list
from dynamic_functions.Home.bot import bot_list
from dynamic_functions.Home.location import location_list


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
async def game() -> None:
    """Show the welcome modal for the current game session."""
    uid = uuid.uuid4().hex[:8]
    game_name = html_lib.escape(_get_current_game() or "Game")
    html = f"""
<style>
  #game-welcome-{uid} {{
    box-sizing: border-box;
    width: 100%;
    min-width: min(100%, 320px);
    padding: 28px;
    color: #f7f4ea;
    background:
      linear-gradient(135deg, rgba(20, 34, 48, 0.96), rgba(47, 55, 45, 0.96)),
      radial-gradient(circle at 18% 20%, rgba(238, 186, 85, 0.22), transparent 34%);
    border: 1px solid rgba(238, 186, 85, 0.42);
    border-radius: 8px;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #game-welcome-{uid} .game-kicker {{
    color: #eeba55;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
  }}
  #game-welcome-{uid} h2 {{
    margin: 10px 0 8px;
    color: #fffaf0;
    font-size: 30px;
    line-height: 1.1;
  }}
  #game-welcome-{uid} p {{
    max-width: 560px;
    margin: 0 0 22px;
    color: #d8d3c6;
    font-size: 15px;
    line-height: 1.5;
  }}
  #game-welcome-{uid} form {{
    display: grid;
    gap: 12px;
    max-width: 420px;
  }}
  #game-welcome-{uid} label {{
    color: #fffaf0;
    font-size: 13px;
    font-weight: 700;
  }}
  #game-welcome-{uid} input {{
    box-sizing: border-box;
    width: 100%;
    min-height: 42px;
    padding: 0 12px;
    color: #fffaf0;
    background: rgba(7, 15, 22, 0.58);
    border: 1px solid rgba(238, 186, 85, 0.42);
    border-radius: 6px;
    font: inherit;
  }}
  #game-welcome-{uid} input:focus {{
    outline: 2px solid rgba(238, 186, 85, 0.45);
    outline-offset: 2px;
  }}
  #game-welcome-{uid} .game-error {{
    min-height: 18px;
    color: #ffb4a8;
    font-size: 13px;
  }}
  #game-welcome-{uid} button {{
    min-height: 40px;
    padding: 0 16px;
    color: #162230;
    background: #eeba55;
    border: 0;
    border-radius: 6px;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
  }}
  #game-welcome-{uid} button:disabled {{
    cursor: default;
    opacity: 0.65;
  }}
</style>
<section id="game-welcome-{uid}" aria-label="Game welcome">
  <div class="game-kicker">Project Atlantis</div>
  <h2>Welcome to {game_name}</h2>
  <p>Choose the character name you want to use in this session.</p>
  <form id="game-welcome-form-{uid}">
    <label for="game-character-name-{uid}">Character name</label>
    <input id="game-character-name-{uid}" name="character_name" type="text" autocomplete="name" maxlength="80" required autofocus>
    <div id="game-welcome-error-{uid}" class="game-error" aria-live="polite"></div>
    <button id="game-welcome-button-{uid}" type="submit">Enter game</button>
  </form>
</section>
"""
    modal_id = await atlantis.client_modal(html, title="Welcome")
    atlantis.session_shared.set("game_welcome_modal_id", modal_id)

    script = f"""
(function() {{
  function bindWelcomeButton() {{
    var form = document.getElementById("game-welcome-form-{uid}");
    var button = document.getElementById("game-welcome-button-{uid}");
    var input = document.getElementById("game-character-name-{uid}");
    var error = document.getElementById("game-welcome-error-{uid}");
    if (!form || !button || !input) {{
      console.error("[GAME] welcome form controls not found");
      return;
    }}
    function focusCharacterName() {{
      input.focus({{ preventScroll: true }});
      input.select();
    }}
    focusCharacterName();
    setTimeout(focusCharacterName, 120);
    form.addEventListener("submit", async function(event) {{
      event.preventDefault();
      if (!window._accessToken) {{
        console.error("[GAME] window._accessToken is empty or undefined");
        return;
      }}
      var characterName = input.value.trim();
      if (!characterName) {{
        if (error) {{
          error.textContent = "Type a character name to continue.";
        }}
        input.focus();
        return;
      }}
      if (error) {{
        error.textContent = "";
      }}
      button.disabled = true;
      button.textContent = "Entering...";
      await sendChatter(window._accessToken, "$**Home**game_welcome_click", {{
        message: "enter_game",
        character_name: characterName
      }});
    }});
  }}
  requestAnimationFrame(function() {{
    requestAnimationFrame(bindWelcomeButton);
  }});
}})()
"""
    await atlantis.client_script(script)


@visible
async def game_welcome_click(message: str, character_name: str) -> None:
    """Callback target for the welcome modal button."""
    character_name = character_name.strip()
    if not character_name:
        raise ValueError("character_name is required")
    modal_id = atlantis.session_shared.get("game_welcome_modal_id")
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove("game_welcome_modal_id")
    await atlantis.client_command(
        f"@character_self {shlex.quote('Guest')} {shlex.quote(character_name)}"
    )
    #await atlantis.client_log(f"Game welcome button clicked: {message}; character_name={character_name}")


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

    await atlantis.client_log(f"Game locked: game_id {game_id} \u2192 '{name}'")


@visible
async def game_show() -> None:
    """Render a live ER diagram of game state as HTML tables with SVG connectors.

    If no game is set, shows BOT, LOCATION, GAME, and all ROLEs across all games.
    CHARACTER and POSITION require an active game.
    """
    game_name = _get_current_game()
    game_id = atlantis.get_game_id()

    # --- Gather data via list functions ---
    bot_rows = await bot_list()
    loc_rows = await location_list()
    role_rows = role_list()

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
        positions = get_positions()
        pos_rows = [{"sid": sid, "location": loc} for sid, loc in sorted(positions.items())]

    # --- Helper to build an HTML table ---
    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    uid = uuid.uuid4().hex[:8]

    def _table(entity_id, title, headers, rows, disabled=False, variant=None):
        """Return HTML for one entity table."""
        scoped_id = f"{entity_id}-{uid}"
        cls = f"er-entity-{uid}"
        if variant:
            cls += f" er-variant-{variant}-{uid}"
        if disabled:
            cls += f" er-disabled-{uid}"
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
    if game_name:
        tables.append(_table("ent-bot", "BOT", ["sid", "name"],
            [[b["sid"], b["name"]] for b in bot_rows]))
        tables.append(_table("ent-location", "LOCATION", ["name", "description", "connects_to"],
            [[l["name"], l["description"], ", ".join(l["connects_to"])] for l in loc_rows]))
        tables.append(_table("ent-role", "ROLE", ["name", "title"],
            [[r["name"], r["title"]] for r in role_rows]))
    else:
        tables.append(_table("ent-bot", "BOT", ["game", "sid", "name"],
            [[b["game"], b["sid"], b["name"]] for b in bot_rows]))
        tables.append(_table("ent-location", "LOCATION", ["game", "name", "description", "connects_to"],
            [[l["game"], l["name"], l["description"], ", ".join(l["connects_to"])] for l in loc_rows]))
        tables.append(_table("ent-role", "ROLE", ["game", "name", "title"],
            [[r["game"], r["name"], r["title"]] for r in role_rows]))
    no_game = not game_name
    tables.append(_table("ent-character", "CHARACTER", ["sid", "role", "isBot", "humanName"],
        [[c["sid"], c["role"], c["isBot"], c.get("humanName", "")] for c in char_rows],
        disabled=no_game, variant="runtime"))
    tables.append(_table("ent-position", "POSITION", ["sid", "location"],
        [[p["sid"], p["location"]] for p in pos_rows],
        disabled=no_game, variant="runtime"))

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
  #er-wrapper-{uid} .er-variant-runtime-{uid} {{
    background: #1e2e22;
    border-color: #4a7a5a;
  }}
  #er-wrapper-{uid} .er-variant-runtime-{uid} .er-title-{uid} {{
    background: #2f5c42;
    color: #d0ffe0;
  }}
  #er-wrapper-{uid} .er-variant-runtime-{uid} th {{
    background: #223a2c;
    color: #a8c8b0;
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

    if not game_name:
        await atlantis.client_log("Game not yet set")

    await atlantis.client_log("Rendered")
