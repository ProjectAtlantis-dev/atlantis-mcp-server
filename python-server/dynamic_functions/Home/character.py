"""Character tools"""

import atlantis
import json
import logging
import os
import shlex
import uuid
from typing import Any, Dict, List

from dynamic_functions.Home.common import require_game_dir
from dynamic_functions.Home.role import _validate_role

logger = logging.getLogger("mcp_server")


def game_data_dir(game_key: str) -> str:
    """Get the game data directory"""
    return require_game_dir(game_key)


def _characters_path(game_key: str) -> str:
    return os.path.join(game_data_dir(game_key), "characters.json")


def _load_characters(game_key: str) -> List[Dict[str, Any]]:
    path = _characters_path(game_key)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Invalid characters.json: expected a list")
    return data


def _save_characters(game_key: str, characters: List[Dict[str, Any]]) -> None:
    path = _characters_path(game_key)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _find_character(game_key: str, sid: str) -> dict:
    """Find a character by sid"""
    for ch in _load_characters(game_key):
        if ch.get("sid") == sid:
            return ch
    raise ValueError(f"No character found for sid: {sid!r}. Register with character_set() first.")


def is_bot_driven(game_key: str, sid: str) -> bool:
    """A character is bot-driven iff its sid has a Bots/ config AND no live session has claimed its chat slot."""
    from dynamic_functions.Home.common import _load_bot_config
    from dynamic_functions.Home.session import chat_slot_claimed
    if _load_bot_config(sid) is None:
        return False
    return not chat_slot_claimed(sid)


async def character_set(game_key: str, sid: str, role: str, display_name: str = "") -> None:
    """Register a character (or update an existing one). display_name auto-fills from bot config or sid."""
    require_game_dir(game_key)
    _validate_role(role)
    if not sid:
        raise ValueError("sid is required")

    from dynamic_functions.Home.common import _load_bot_config
    if not display_name.strip():
        loaded = _load_bot_config(sid)
        display_name = loaded[0].get("displayName", sid) if loaded else sid
    display_name = display_name.strip()

    characters = _load_characters(game_key)
    record = {"sid": sid, "role": role, "displayName": display_name}
    for ch in characters:
        if ch.get("sid") == sid:
            ch.update(record)
            _save_characters(game_key, characters)
            await atlantis.client_log(f"{display_name} ({sid}) is now roleplaying as {role}")
            return
    characters.append(record)
    _save_characters(game_key, characters)
    await atlantis.client_log(f"{display_name} ({sid}) is now roleplaying as {role}")


@visible
async def prompt_display_name(game_key: str, role: str) -> None:
    """Pop up a modal asking the caller for their display name; on submit, assign the role."""
    require_game_dir(game_key)
    uid = uuid.uuid4().hex[:8]
    game_key_js = json.dumps(game_key)
    role_js = json.dumps(role)
    html = f"""
<style>
  #displayname-{uid} {{
    box-sizing: border-box;
    width: 100%;
    min-width: min(100%, 320px);
    padding: 28px;
    color: #f7f4ea;
    background:
      linear-gradient(to bottom, rgba(20, 34, 48, 0.96), rgba(20, 50, 60, 0.96)),
      radial-gradient(circle at 18% 20%, rgba(20, 255, 208, 0.22), transparent 34%);
    border: 1px solid rgba(20, 255, 208, 0.42);
    border-radius: 8px;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #displayname-{uid} h2 {{
    margin: 10px 0 28px;
    font-size: 30px;
    line-height: 1.1;
    color: #fffaf0;
  }}
  #displayname-{uid} form {{
    display: grid;
    gap: 12px;
    max-width: 420px;
  }}
  #displayname-{uid} label {{
    color: #fffaf0;
    font-size: 13px;
    font-weight: 700;
  }}
  #displayname-{uid} input {{
    box-sizing: border-box;
    width: 100%;
    min-height: 42px;
    padding: 0 12px;
    color: #fffaf0;
    background: rgba(7, 15, 22, 0.58);
    border: 1px solid rgba(20, 255, 208, 0.42);
    border-radius: 6px;
    font: inherit;
  }}
  #displayname-{uid} input:focus {{
    outline: 2px solid rgba(20, 255, 208, 0.45);
    outline-offset: 2px;
  }}
  #displayname-{uid} .err {{
    min-height: 18px;
    color: #ffb4a8;
    font-size: 13px;
  }}
  #displayname-{uid} button {{
    min-height: 40px;
    padding: 0 16px;
    color: #fffaf0;
    background: linear-gradient(to bottom, #1a8a78, #143a52);
    border: 0;
    border-radius: 6px;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
  }}
  #displayname-{uid} button:hover {{
    background: linear-gradient(to bottom, #22b89e, #1a527a);
  }}
  #displayname-{uid} button:disabled {{
    cursor: default;
    opacity: 0.65;
  }}
</style>
<section id="displayname-{uid}" aria-label="Choose display name">
  <h2>Welcome to Atlantis</h2>
  <form id="displayname-form-{uid}">
    <label for="displayname-input-{uid}">Enter your display name</label>
    <input id="displayname-input-{uid}" name="display_name" type="text" autocomplete="name" maxlength="80" required autofocus>
    <div id="displayname-err-{uid}" class="err" aria-live="polite"></div>
    <button id="displayname-btn-{uid}" type="submit">Enter</button>
  </form>
</section>
"""
    modal_id = await atlantis.client_modal(html, title="Welcome")
    atlantis.session_shared.set(f"displayname_modal_id:{game_key}", modal_id)

    script = f"""
(function() {{
  function bind() {{
    var form = document.getElementById("displayname-form-{uid}");
    var button = document.getElementById("displayname-btn-{uid}");
    var input = document.getElementById("displayname-input-{uid}");
    var error = document.getElementById("displayname-err-{uid}");
    if (!form || !button || !input) return;
    function focusInput() {{ input.focus({{ preventScroll: true }}); input.select(); }}
    focusInput();
    setTimeout(focusInput, 120);
    form.addEventListener("submit", async function(event) {{
      event.preventDefault();
      if (!window._accessToken) return;
      var name = input.value.trim();
      if (!name) {{ if (error) error.textContent = "Type a display name to continue."; input.focus(); return; }}
      if (error) error.textContent = "";
      button.disabled = true;
      button.textContent = "Entering...";
      await sendChatter(window._accessToken, "$**Home**prompt_display_name_click", {{
        message: "display_name",
        game_key: {game_key_js},
        role: {role_js},
        display_name: name
      }});
    }});
  }}
  requestAnimationFrame(function() {{ requestAnimationFrame(bind); }});
}})()
"""
    await atlantis.client_script(script)


@visible
async def prompt_display_name_click(message: str, game_key: str, role: str, display_name: str) -> None:
    """Handle the display-name modal submit."""
    require_game_dir(game_key)
    display_name = (display_name or "").strip()
    if not display_name:
        raise ValueError("display_name is required")
    modal_key = f"displayname_modal_id:{game_key}"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove(modal_key)
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    await character_set(game_key, sid, role, display_name)
    await atlantis.client_command(f"@character_move {shlex.quote(game_key)}")


@visible
def character_list(game_key: str) -> List[Dict[str, Any]]:
    """List game characters with their current positions (blank if unplaced)."""
    require_game_dir(game_key)
    from dynamic_functions.Home.location import get_positions
    positions = get_positions(game_key)
    return [
        {**ch, "location": positions.get(ch["sid"], "")}
        for ch in _load_characters(game_key)
    ]
