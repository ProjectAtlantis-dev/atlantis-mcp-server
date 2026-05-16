"""Character tools"""

import atlantis
import asyncio
import html as html_lib
import json
import logging
import os
import shlex
import uuid
from typing import Any, Dict, List, Optional

from dynamic_functions.Home.common import require_game_dir
from dynamic_functions.Home.role import _validate_role

logger = logging.getLogger("mcp_server")


def _characters_path(game_key: str) -> str:
    return os.path.join(require_game_dir(game_key), "characters.json")


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

@visible
async def character_assign(game_key: str, sid: str, role: str, display_name: str = "") -> None:
    """Register a character (or update an existing one).

    The (sid x role) specialization prompt lives at Game/Bots/<sid>/<role>.md
    and is loaded at chat time — it's not stored on the character record.
    """
    require_game_dir(game_key)
    _validate_role(role)
    if not sid:
        raise ValueError("sid is required")

    from dynamic_functions.Home.common import _load_bot_config
    loaded = _load_bot_config(sid)
    if loaded is None:
        caller = atlantis.get_caller()
        if sid != caller:
            raise ValueError(f"Invalid sid {sid!r}: must be a bot sid or match the caller")
        if not display_name.strip():
            raise ValueError("display_name is required when sid matches the caller")
    if not display_name.strip():
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
async def prompt_string(prompt_text: str) -> Optional[str]:
    """Pop up a modal asking the caller for a string.

    Returns None if the user closes/cancels the modal without submitting.
    """
    uid = uuid.uuid4().hex[:8]
    prompt_id = f"displayname:{uid}"
    prompt_id_js = json.dumps(prompt_id)
    prompt_text_html = html_lib.escape(prompt_text or "Enter a value")
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    atlantis.session_shared.set(f"{prompt_id}:future", future)
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
    <label for="displayname-input-{uid}">{prompt_text_html}</label>
    <input id="displayname-input-{uid}" name="display_name" type="text" autocomplete="name" maxlength="80" required autofocus>
    <div id="displayname-err-{uid}" class="err" aria-live="polite"></div>
    <button id="displayname-btn-{uid}" type="submit">Enter</button>
  </form>
</section>
"""
    modal_id = await atlantis.client_modal(html, title="Welcome")
    atlantis.session_shared.set(f"{prompt_id}:modal_id", modal_id)

    # Route modal-originated commands (cancel / submit) to this tool call's
    # exec shell so they nest inside the prompt_string subshell rather than
    # polluting the user's parent shell history.
    #
    # WHY THIS IS REQUIRED (do not remove the 4th arg to sendChatter below):
    #   - Every @visible/@chat tool call spawns an isolated callback shell on
    #     the Node side via Session.spawnShell(parent, 'tool', isBackground=True).
    #     That shell gets a FLAT name like "37" (not "2.36.1") precisely so the
    #     tool's internal chatter (modals, scripts, log lines, button click
    #     callbacks) stays out of the user's command history on "2.36".
    #   - The Node engage handler defaults missing shellPath to the websocket's
    #     root working shell (app_server.ts ~line 2131:
    #     `targetShellPath = params.shellPath ?? session.getWorkingPath(rootShellPath)`).
    #     There is no server-side awareness of "the currently active tool
    #     callback shell" - it can't infer "37" from the websocket alone,
    #     since multiple tool calls can be in flight at once.
    #   - Therefore: if `exec_shell_js` is omitted from the sendChatter calls
    #     below, prompt_string_click/_cancel will be routed to the user's main
    #     shell ("2.X") instead of "37.1". Visible symptom: the click event
    #     shows up as a sibling of the prompt_string command in /history.
    #   - The matching Node-side fix that lets the data-callback render skip
    #     auto-display correctly is in Session.ts handleMcpCallback's
    #     `messageType === "data"` branch, which explicitly marks the caller's
    #     shell. See the comment there for the symmetric explanation.
    exec_shell_js = json.dumps(atlantis.get_exec_shell_path())

    script = f"""
(function() {{
  var settled = false;
  var observer = null;
  function cleanup() {{ if (observer) {{ try {{ observer.disconnect(); }} catch (e) {{}} observer = null; }} }}
  async function cancel() {{
    if (settled) return;
    settled = true;
    cleanup();
    if (!window._accessToken) return;
    try {{
      await sendChatter(window._accessToken, "$**Home**prompt_string_cancel", {{
        prompt_id: {prompt_id_js}
      }}, {exec_shell_js});
    }} catch (e) {{}}
  }}
  function bind() {{
    var root = document.getElementById("displayname-{uid}");
    var form = document.getElementById("displayname-form-{uid}");
    var button = document.getElementById("displayname-btn-{uid}");
    var input = document.getElementById("displayname-input-{uid}");
    var error = document.getElementById("displayname-err-{uid}");
    if (!root || !form || !button || !input) return;
    function focusInput() {{ input.focus({{ preventScroll: true }}); input.select(); }}
    focusInput();
    setTimeout(focusInput, 120);
    form.addEventListener("submit", async function(event) {{
      event.preventDefault();
      if (settled) return;
      if (!window._accessToken) return;
      var name = input.value.trim();
      if (!name) {{ if (error) error.textContent = "Type a display name to continue."; input.focus(); return; }}
      if (error) error.textContent = "";
      button.disabled = true;
      button.textContent = "Entering...";
      settled = true;
      cleanup();
      await sendChatter(window._accessToken, "$**Home**prompt_string_click", {{
        prompt_id: {prompt_id_js},
        display_name: name
      }}, {exec_shell_js});
    }});
    observer = new MutationObserver(function() {{
      if (!document.body.contains(root)) {{ cancel(); }}
    }});
    observer.observe(document.body, {{ childList: true, subtree: true }});
  }}
  requestAnimationFrame(function() {{ requestAnimationFrame(bind); }});
}})()
"""
    await atlantis.client_script(script)
    try:
        return await future
    finally:
        atlantis.session_shared.remove(f"{prompt_id}:future")
        atlantis.session_shared.remove(f"{prompt_id}:modal_id")


@visible
async def prompt_string_click(prompt_id: str, display_name: str) -> None:
    """Handle the display-name modal submit."""
    display_name = (display_name or "").strip()
    if not display_name:
        raise ValueError("display_name is required")
    modal_key = f"{prompt_id}:modal_id"
    future_key = f"{prompt_id}:future"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove(modal_key)
    future = atlantis.session_shared.get(future_key)
    if future is None:
        raise ValueError("Display-name prompt is no longer active")
    if not future.done():
        future.set_result(display_name)


@visible
async def prompt_string_cancel(prompt_id: str) -> None:
    """Handle the user closing the prompt modal without submitting."""
    modal_key = f"{prompt_id}:modal_id"
    future_key = f"{prompt_id}:future"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        try:
            await atlantis.client_modal_close(modal_id)
        except Exception:
            pass
        atlantis.session_shared.remove(modal_key)
    future = atlantis.session_shared.get(future_key)
    if future is not None and not future.done():
        future.set_result(None)


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
