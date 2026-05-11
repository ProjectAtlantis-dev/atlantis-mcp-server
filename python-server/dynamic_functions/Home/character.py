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


def _find_character(game_key: str, sid: str, is_bot: bool) -> dict:
    """Find a character by sid"""
    for ch in _load_characters(game_key):
        if ch.get("sid") == sid:
            if ch.get("isBot", True) != is_bot:
                kind = "bot" if is_bot else "human"
                raise ValueError(f"Character {sid!r} is not a {kind}")
            return ch
    kind = "character_bot()" if is_bot else "character_self() or character_human()"
    raise ValueError(f"No character found for sid: {sid!r}. Register role with {kind} first.")


async def _upsert_character(game_key: str, sid: str, role: str, is_bot: bool, human_name: str = "") -> None:
    """Create or update a character"""
    _validate_role(role)
    characters = _load_characters(game_key)

    record: Dict[str, Any] = {"isBot": is_bot}
    if is_bot:
        record["sid"] = sid
    else:
        record["sid"] = sid
        record["humanName"] = human_name
    record["role"] = role

    is_self = (not is_bot) and atlantis.get_caller() == sid
    if is_bot:
        subject, verb = f"Bot {sid}", "is"
    elif is_self:
        subject, verb = "You", "are"
    else:
        subject, verb = f"User {sid}", "is"
    suffix = f" named {human_name}" if (human_name and not is_bot) else ""
    message = f"{subject} {verb} now roleplaying as {role}{suffix}"

    for ch in characters:
        if ch.get("sid") == sid:
            ch.update(record)
            _save_characters(game_key, characters)
            await atlantis.client_log(message)
            return

    characters.append(record)
    _save_characters(game_key, characters)
    await atlantis.client_log(message)


@visible
async def character_bot(game_key: str, sid: str, role: str) -> None:
    """Assign a role to a bot"""
    require_game_dir(game_key)
    from dynamic_functions.Home.common import _load_bot_config, _available_bot_sids
    if _load_bot_config(sid) is None:
        raise ValueError(f"Unknown bot sid: {sid!r}. Must match a bot in Bots/ (e.g. {_available_bot_sids()})")
    await _upsert_character(game_key, sid, role, is_bot=True)


@visible
async def character_human(game_key: str, sid: str, role: str, human_name: str) -> None:
    """Assign a role to a human"""
    require_game_dir(game_key)
    if not sid:
        raise ValueError("sid is required for human characters")
    if not human_name or not human_name.strip():
        raise ValueError("human_name is required for human characters")
    await _upsert_character(game_key, sid, role, is_bot=False, human_name=human_name.strip())


@visible
async def character_self(game_key: str, role: str, human_name: str) -> None:
    """Assign a role to the caller"""
    require_game_dir(game_key)
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")
    await character_human(game_key, sid, role, human_name)


@visible
async def human_spawn(game_key: str, role: str) -> None:
    """Assume a role as a human. Prompts for a display name if the caller doesn't have one yet."""
    require_game_dir(game_key)
    _validate_role(role)
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("Unable to determine caller identity")

    existing_name = ""
    for ch in _load_characters(game_key):
        if ch.get("sid") == sid and not ch.get("isBot", True):
            existing_name = (ch.get("humanName") or "").strip()
            break

    if existing_name:
        await character_human(game_key, sid, role, existing_name)
        await atlantis.client_command(f"@go {shlex.quote(game_key)}")
        return

    await _prompt_display_name(game_key, role)


async def _prompt_display_name(game_key: str, role: str) -> None:
    """Show the display-name modal for a human assuming a role."""
    uid = uuid.uuid4().hex[:8]
    game_key_js = json.dumps(game_key)
    role_js = json.dumps(role)
    html = f"""
<style>
  #human-spawn-{uid} {{
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
  #human-spawn-{uid} h2 {{
    margin: 10px 0 28px;
    font-size: 30px;
    line-height: 1.1;
    color: #fffaf0;
  }}
  #human-spawn-{uid} form {{
    display: grid;
    gap: 12px;
    max-width: 420px;
  }}
  #human-spawn-{uid} label {{
    color: #fffaf0;
    font-size: 13px;
    font-weight: 700;
  }}
  #human-spawn-{uid} input {{
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
  #human-spawn-{uid} input:focus {{
    outline: 2px solid rgba(20, 255, 208, 0.45);
    outline-offset: 2px;
  }}
  #human-spawn-{uid} .err {{
    min-height: 18px;
    color: #ffb4a8;
    font-size: 13px;
  }}
  #human-spawn-{uid} button {{
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
  #human-spawn-{uid} button:hover {{
    background: linear-gradient(to bottom, #22b89e, #1a527a);
  }}
  #human-spawn-{uid} button:disabled {{
    cursor: default;
    opacity: 0.65;
  }}
</style>
<section id="human-spawn-{uid}" aria-label="Choose display name">
  <h2>Welcome to Atlantis</h2>
  <form id="human-spawn-form-{uid}">
    <label for="human-spawn-name-{uid}">Enter your display name</label>
    <input id="human-spawn-name-{uid}" name="display_name" type="text" autocomplete="name" maxlength="80" required autofocus>
    <div id="human-spawn-err-{uid}" class="err" aria-live="polite"></div>
    <button id="human-spawn-btn-{uid}" type="submit">Enter</button>
  </form>
</section>
"""
    modal_id = await atlantis.client_modal(html, title="Welcome")
    atlantis.session_shared.set(f"human_spawn_modal_id:{game_key}", modal_id)

    script = f"""
(function() {{
  function bind() {{
    var form = document.getElementById("human-spawn-form-{uid}");
    var button = document.getElementById("human-spawn-btn-{uid}");
    var input = document.getElementById("human-spawn-name-{uid}");
    var error = document.getElementById("human-spawn-err-{uid}");
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
      await sendChatter(window._accessToken, "$**Home**human_spawn_click", {{
        message: "human_spawn",
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
async def human_spawn_click(message: str, game_key: str, role: str, display_name: str) -> None:
    """Handle the human-spawn display-name modal."""
    game_key = str(game_key or "").strip()
    if not game_key:
        raise ValueError("game_key is required")
    require_game_dir(game_key)
    display_name = (display_name or "").strip()
    if not display_name:
        raise ValueError("display_name is required")
    modal_key = f"human_spawn_modal_id:{game_key}"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove(modal_key)
    await atlantis.client_command(
        f"@character_self {shlex.quote(game_key)} {shlex.quote(role)} {shlex.quote(display_name)}"
    )
    await atlantis.client_command(f"@go {shlex.quote(game_key)}")


@visible
def character_list(game_key: str) -> List[Dict[str, Any]]:
    """List game characters"""
    require_game_dir(game_key)
    from dynamic_functions.Home.common import _load_bot_config
    characters = _load_characters(game_key)
    result = []
    for ch in characters:
        entry = dict(ch)
        if ch.get("isBot", True):
            loaded = _load_bot_config(ch["sid"])
            entry["displayName"] = loaded[0].get("displayName", ch["sid"]) if loaded else ch["sid"]
        else:
            entry["displayName"] = ch.get("humanName", ch["sid"])
        result.append(entry)
    return result
