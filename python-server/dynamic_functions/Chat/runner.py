"""Game flow orchestration — the @public join/create/resume entry points and
their UI pickers. This layer sits above the game-record foundation (.game) and
the camera/roster views, so it may import all of them without a cycle."""

import atlantis
import asyncio
import html as html_lib
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from .bot import _bot_pick_dialog, bot_image_data
from dynamic_functions.Home.modal import ModalGoBack, _modal_panel_css, modal_confirm, modal_menu, modal_radio, modal_string
from .game import (
    GAME_STATE_STOPPED,
    _caller_is_member,
    _game_create,
    _game_pick_dialog,
    _game_read,
    _game_read_from_dir,
    _game_rows,
    _game_update,
    add_caller_membership,
    game_dir,
    game_find_current,
    game_start,
    require_membership,
)
from .camera import _load_cameras, camera_bind, camera_follow
from .location import _leaf_location_keys, _location_pick_dialog, load_location
from .roster import _roster_row_label, _roster_row_state
from .scene import _scene_pick_dialog
from .common import app_bg_default


async def _warn_empty_roster() -> None:
    await modal_confirm(
        "All roster slots are empty",
        title="Roster",
        cancel_label="",
    )


async def _roster_edit_modal() -> Optional[Dict[str, str]]:

    roster = await atlantis.client_command("@roster_list")
    heading = "Edit Roster"

    uid = uuid.uuid4().hex[:8]
    roster_modal_id = f"roster_edit:{uid}"
    roster_modal_id_js = json.dumps(roster_modal_id)
    heading_block = f"<h2>{html_lib.escape(heading)}</h2>" if heading else ""
    rows_html = []
    state_options = [("human", "Human"), ("ai", "AI"), ("empty", "Empty")]

    for row in roster:
        slot_key = str(row.get("key") or "").strip()
        if not slot_key:
            continue
        state = _roster_row_state(row)
        state_value = state.lower()
        display_name = "" if state == "Empty" else _roster_row_label(row)
        bot_sid = str(row.get("bot_sid") or "").strip()
        bot_image = bot_image_data(bot_sid) if bot_sid else ""
        bot_img_html = (
            f'<img class="roster-bot-thumb" src="{html_lib.escape(bot_image, quote=True)}" alt="{html_lib.escape(bot_sid, quote=True)}">'
            if bot_image else ""
        )
        select_options = "".join(
            f'<option value="{value}"{" selected" if label == state else ""}>{label}</option>'
            for value, label in state_options
        )
        rows_html.append(f"""
      <tr class="roster-row" data-slot-key="{html_lib.escape(slot_key, quote=True)}" data-bot-sid="{html_lib.escape(bot_sid, quote=True)}" data-state="{state_value}" data-name="{html_lib.escape(display_name, quote=True)}">
        <td class="roster-bot-image">{bot_img_html}</td>
        <td>{html_lib.escape(slot_key)}</td>
        <td>{html_lib.escape(bot_sid)}</td>
        <td class="roster-state">
        <select aria-label="{html_lib.escape(slot_key, quote=True)} state">
          {select_options}
        </select>
        </td>
        <td class="roster-name"></td>
      </tr>
""")

    if not rows_html:
        raise RuntimeError("No roster slots found")

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    atlantis.session_shared.set(f"{roster_modal_id}:future", future)
    html = f"""
<style>
{_modal_panel_css(
    f"#roster-edit-panel-{uid}",
    f"#rosteredit-{uid}",
    ready_class="roster-edit-ready",
    padding=22,
    heading_margin="4px 0 16px",
    heading_font_size=24,
    heading_line_height=1.15,
)}
  #rosteredit-{uid} {{
    width: max-content;
    max-width: 100%;
    min-width: 0;
    visibility: hidden;
  }}
  .jsPanel:has(#rosteredit-{uid}) {{
    width: auto !important;
    min-width: 0 !important;
    max-width: calc(100vw - 32px) !important;
    left: 50% !important;
    right: auto !important;
    bottom: auto !important;
    transform: translateX(-50%) !important;
  }}
  #rosteredit-{uid} .roster-table {{
    border-collapse: separate;
    border-spacing: 0 8px;
    table-layout: auto;
    width: auto;
    max-width: 100%;
  }}
  #rosteredit-{uid} th {{
    padding: 0 14px 2px;
    color: rgba(255, 250, 240, 0.68);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0;
    text-align: left;
    text-transform: uppercase;
  }}
  #rosteredit-{uid} td {{
    box-sizing: border-box;
    min-height: 42px;
    padding: 0 14px;
    color: #fffaf0;
    background: rgba(7, 15, 22, 0.58);
    font-size: 18px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: middle;
  }}
  #rosteredit-{uid} td:first-child {{
    border: 1px solid rgba(20, 255, 208, 0.34);
    border-right: 0;
    border-radius: 6px 0 0 6px;
  }}
  #rosteredit-{uid} td:not(:first-child):not(:last-child) {{
    border-top: 1px solid rgba(20, 255, 208, 0.34);
    border-bottom: 1px solid rgba(20, 255, 208, 0.34);
  }}
  #rosteredit-{uid} td:last-child {{
    border: 1px solid rgba(20, 255, 208, 0.34);
    border-left: 0;
    border-radius: 0 6px 6px 0;
  }}
  #rosteredit-{uid} .roster-row {{
    height: 42px;
  }}
  #rosteredit-{uid} .roster-bot-image {{
    width: 48px;
    min-width: 48px;
    padding: 0 8px;
  }}
  #rosteredit-{uid} .roster-bot-thumb {{
    display: block;
    width: 36px;
    height: 36px;
    object-fit: cover;
    border-radius: 4px;
  }}
  #rosteredit-{uid} .roster-empty-name {{
    color: rgba(255, 250, 240, 0.52);
  }}
  #rosteredit-{uid} input,
  #rosteredit-{uid} select {{
    box-sizing: border-box;
    min-height: 30px;
    color: #fffaf0;
    background: rgba(7, 15, 22, 0.58);
    border: 1px solid rgba(20, 255, 208, 0.34);
    border-radius: 6px;
    font: inherit;
    font-size: 16px;
  }}
  #rosteredit-{uid} input {{
    width: 150px;
    padding: 0 10px;
  }}
  #rosteredit-{uid} input::placeholder {{
    color: rgba(255, 250, 240, 0.42);
  }}
  #rosteredit-{uid} select {{
    width: 104px;
    padding: 0 28px 0 10px;
    cursor: pointer;
  }}
  #rosteredit-{uid} input:focus,
  #rosteredit-{uid} select:focus {{
    background: rgba(20, 255, 208, 0.14);
    border-color: rgba(20, 255, 208, 0.62);
    outline: none;
  }}
  #rosteredit-{uid} .roster-error {{
    min-height: 16px;
    margin-top: 2px;
    color: #ffb4a8;
    font-size: 13px;
  }}
  #rosteredit-{uid} .roster-error:empty {{
    display: none;
  }}
  #rosteredit-{uid} .roster-actions {{
    display: flex;
    justify-content: center;
    gap: 10px;
    padding-top: 8px;
  }}
  #rosteredit-{uid} .roster-ok {{
    box-sizing: border-box;
    min-width: 96px;
    min-height: 40px;
    padding: 0 16px;
    color: #fffaf0;
    background: rgba(20, 255, 208, 0.16);
    border: 1px solid rgba(20, 255, 208, 0.62);
    border-radius: 6px;
    font: inherit;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
  }}
  #rosteredit-{uid} .roster-ok:hover,
  #rosteredit-{uid} .roster-ok:focus {{
    background: rgba(20, 255, 208, 0.22);
    border-color: rgba(20, 255, 208, 0.78);
    outline: none;
  }}
</style>
<section id="rosteredit-{uid}" aria-label="Roster editor">
  {heading_block}
  <table class="roster-table">
    <thead>
      <tr>
        <th scope="col"></th>
        <th scope="col">Slot</th>
        <th scope="col">Bot</th>
        <th scope="col">State</th>
        <th scope="col">Name</th>
      </tr>
    </thead>
    <tbody>
    {"".join(rows_html)}
    </tbody>
  </table>
  <div id="roster-error-{uid}" class="roster-error" aria-live="polite"></div>
  <div class="roster-actions">
    <button type="button" class="roster-ok">Continue</button>
  </div>
</section>
"""
    modal_id = await atlantis.client_modal(html, title="Roster")
    atlantis.session_shared.set(f"{roster_modal_id}:modal_id", modal_id)
    exec_shell_js = json.dumps(atlantis.get_exec_shell_path())
    script = f"""
(function() {{
  var settled = false;
  var observer = null;
  function cleanup() {{ if (observer) {{ try {{ observer.disconnect(); }} catch (e) {{}} observer = null; }} }}
  function reveal(root) {{
    root.style.visibility = "visible";
    root.classList.add("roster-edit-ready");
  }}
  function markHost(host) {{
    if (!host) return;
    if (!host.id) host.id = "roster-edit-panel-{uid}";
    host.classList.add("roster-edit-panel");
    host.dataset.modalKind = "roster-edit";
    host.dataset.rosterEditUid = "{uid}";
  }}
  function centerDialog(root, shouldReveal) {{
    var host = null;
    var node = root;
    for (var i = 0; i < 8 && node && node !== document.body; i++) {{
      var style = window.getComputedStyle(node);
      var rect = node.getBoundingClientRect();
      var fillsViewport = rect.width >= window.innerWidth * 0.9 && rect.height >= window.innerHeight * 0.9;
      if ((style.position === "fixed" || style.position === "absolute") && !fillsViewport) {{
        host = node;
        break;
      }}
      node = node.parentElement;
    }}
    if (!host) {{
      if (shouldReveal) reveal(root);
      return;
    }}
    markHost(host);
    host.style.minWidth = "0";
    host.style.width = "auto";
    var preRect = host.getBoundingClientRect();
    var rootRect = root.getBoundingClientRect();
    var hostExtraWidth = Math.max(0, Math.ceil(preRect.width - rootRect.width));
    var targetWidth = Math.ceil(root.scrollWidth + hostExtraWidth);
    var viewportMax = Math.max(0, window.innerWidth - 32);
    host.style.width = Math.min(targetWidth, viewportMax) + "px";
    host.style.maxWidth = "calc(100vw - 32px)";
    host.style.left = "50%";
    host.style.right = "auto";
    host.style.bottom = "auto";
    host.style.margin = "0";
    var adjustedRect = host.getBoundingClientRect();
    var viewportMargin = 16;
    var centeredTop = Math.round((window.innerHeight - adjustedRect.height) / 2);
    var maxTop = Math.max(viewportMargin, window.innerHeight - adjustedRect.height - viewportMargin);
    var clampedTop = Math.min(Math.max(viewportMargin, centeredTop), maxTop);
    host.style.top = clampedTop + "px";
    host.style.transform = "translateX(-50%)";
    if (shouldReveal) reveal(root);
  }}
  function scheduleCenter(root) {{
    centerDialog(root, false);
    requestAnimationFrame(function() {{
      centerDialog(root, false);
      setTimeout(function() {{ centerDialog(root, true); }}, 180);
    }});
  }}
  async function send(action, payload) {{
    if (settled) return;
    if (!window._accessToken) return;
    settled = true;
    cleanup();
    await sendChatter(window._accessToken, action, payload, {exec_shell_js});
  }}
  async function cancel() {{
    try {{
      await send("@roster_edit_modal_cancel", {{ roster_modal_id: {roster_modal_id_js} }});
    }} catch (e) {{}}
  }}
  function error(message) {{
    var node = document.getElementById("roster-error-{uid}");
    if (node) node.textContent = message || "";
  }}
  function rowState(row) {{
    return (row && row.getAttribute("data-state")) || "empty";
  }}
  function rowName(row) {{
    return (row && row.getAttribute("data-name")) || "";
  }}
  function nameCell(row) {{
    return row ? row.querySelector(".roster-name") : null;
  }}
  function selectFor(row) {{
    return row ? row.querySelector("select") : null;
  }}
  function setSelect(row, value) {{
    var select = selectFor(row);
    if (select) select.value = value || rowState(row);
  }}
  function renderName(row, shouldFocus) {{
    var cell = nameCell(row);
    if (!cell) return;
    cell.textContent = "";
    var state = row.getAttribute("data-pending-state") || rowState(row);
    if (state === "human") {{
      var input = document.createElement("input");
      input.type = "text";
      input.value = rowName(row);
      input.placeholder = "Name";
      input.maxLength = 200;
      input.setAttribute("aria-label", (row.getAttribute("data-slot-key") || "Slot") + " name");
      input.addEventListener("keydown", function(event) {{
        if (event.key === "Enter") {{
          event.preventDefault();
          commitHuman(row);
        }} else if (event.key === "Escape") {{
          event.preventDefault();
          error("");
          row.removeAttribute("data-pending-state");
          setSelect(row, rowState(row));
          renderName(row, false);
        }}
      }});
      cell.appendChild(input);
      if (shouldFocus) {{
        input.focus({{ preventScroll: true }});
        input.select();
      }}
      return;
    }}
    var span = document.createElement("span");
    if (state === "empty") {{
      span.className = "roster-empty-name";
      span.textContent = "-";
    }} else {{
      span.textContent = rowName(row) || "-";
    }}
    cell.appendChild(span);
  }}
  function commitHuman(row) {{
    var input = row && row.querySelector(".roster-name input");
    var value = input ? input.value.trim() : "";
    if (!value) {{
      error("Enter a name for the human slot.");
      if (input) input.focus({{ preventScroll: true }});
      return;
    }}
    send("@roster_edit_modal_change", {{
      roster_modal_id: {roster_modal_id_js},
      slot_key: row ? (row.getAttribute("data-slot-key") || "") : "",
      state: "human",
      display_name: value
    }});
  }}
  function commitPendingHuman(root) {{
    var active = document.activeElement;
    var row = active && active.closest ? active.closest(".roster-row") : null;
    if (row && active.matches && active.matches(".roster-name input")) {{
      commitHuman(row);
      return true;
    }}
    row = Array.prototype.slice.call(root.querySelectorAll(".roster-row")).find(function(candidate) {{
      return candidate.getAttribute("data-pending-state") === "human";
    }});
    if (row) {{
      commitHuman(row);
      return true;
    }}
    row = Array.prototype.slice.call(root.querySelectorAll(".roster-row")).find(function(candidate) {{
      var input = candidate.querySelector(".roster-name input");
      return input && input.value.trim() !== rowName(candidate);
    }});
    if (row) {{
      commitHuman(row);
      return true;
    }}
    return false;
  }}
  function bind() {{
    var root = document.getElementById("rosteredit-{uid}");
    if (!root) return;
    scheduleCenter(root);
    Array.prototype.slice.call(root.querySelectorAll(".roster-row")).forEach(function(row) {{
      renderName(row, false);
    }});
    Array.prototype.slice.call(root.querySelectorAll(".roster-row select")).forEach(function(select) {{
      select.addEventListener("change", function() {{
        var row = select.closest(".roster-row");
        error("");
        if (select.value === "human") {{
          if (row) {{
            row.setAttribute("data-pending-state", "human");
            renderName(row, true);
          }}
          return;
        }}
        if (row) row.removeAttribute("data-pending-state");
        send("@roster_edit_modal_change", {{
          roster_modal_id: {roster_modal_id_js},
          slot_key: row ? (row.getAttribute("data-slot-key") || "") : "",
          state: select.value || "",
          bot_sid: row ? (row.getAttribute("data-bot-sid") || "") : "",
          display_name: ""
        }});
      }});
    }});
    var ok = root.querySelector(".roster-ok");
    if (ok) {{
      ok.addEventListener("click", function() {{
        if (commitPendingHuman(root)) return;
        send("@roster_edit_modal_ok", {{ roster_modal_id: {roster_modal_id_js} }});
      }});
    }}
    var firstSelect = root.querySelector(".roster-row select");
    if (firstSelect) firstSelect.focus({{ preventScroll: true }});
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
        atlantis.session_shared.remove(f"{roster_modal_id}:future")
        atlantis.session_shared.remove(f"{roster_modal_id}:modal_id")


async def _settle_roster_edit_modal(roster_modal_id: str, result: Optional[Dict[str, str]]) -> None:
    modal_key = f"{roster_modal_id}:modal_id"
    future_key = f"{roster_modal_id}:future"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        try:
            await atlantis.client_modal_close(modal_id)
        except Exception:
            pass
        atlantis.session_shared.remove(modal_key)
    future = atlantis.session_shared.get(future_key)
    if future is None:
        raise ValueError("Roster editor modal is no longer active")
    if not future.done():
        future.set_result(result)


@public
@visible
async def roster_edit_modal_change(
    roster_modal_id: str,
    slot_key: str,
    state: str,
    display_name: Optional[str] = None,
    bot_sid: Optional[str] = None,
) -> None:
    """Handle a roster editor state dropdown change."""
    await _settle_roster_edit_modal(
        roster_modal_id,
        {
            "action": "state",
            "slot_key": str(slot_key or "").strip(),
            "state": str(state or "").strip().lower(),
            "display_name": str(display_name or "").strip(),
            "bot_sid": str(bot_sid or "").strip(),
        },
    )


@public
@visible
async def roster_edit_modal_ok(roster_modal_id: str) -> None:
    """Handle the roster editor continue button."""
    await _settle_roster_edit_modal(roster_modal_id, {"action": "ok"})


@public
@visible
async def roster_edit_modal_cancel(roster_modal_id: str) -> None:
    """Handle closing the roster editor without an explicit action."""
    await _settle_roster_edit_modal(roster_modal_id, None)


@visible
async def _game_resume(game_key: str) -> str:
    meta = _game_read(game_key)
    members = meta.setdefault("members", {})
    session_key = atlantis.get_session_key()
    if not session_key:
        raise RuntimeError("No session key in this call context")
    if not _caller_is_member(meta, session_key):
        raise PermissionError(f"Session is not a member of game '{game_key}'")
    if session_key not in members:
        add_caller_membership(members)
        _game_update(game_key, meta)
    await atlantis.client_log(f"Existing game found: {game_key}")
    return await _game_enter(game_key)


@visible
async def _game_enter(game_key: str) -> str:
    redirect_url = _game_window_redirect_url(game_key)
    if redirect_url:
        await atlantis.client_log(f"Redirecting to game window: {redirect_url}")
        await atlantis.client_script(f"window.location.assign({redirect_url!r});")
        return game_key
    await atlantis.client_command("/cursor join", {"game_key": game_key})
    await game_init(game_key)
    return game_key


@visible
def _game_window_redirect_url(game_key: str) -> Optional[str]:
    meta = _game_read(game_key)
    target_user_game_id = meta.get("user_game_id")
    if target_user_game_id is None:
        return None

    current_user_game_id = atlantis.get_user_game_id()
    if str(target_user_game_id) == str(current_user_game_id):
        return None

    sid = str(meta.get("owner") or atlantis.get_caller() or "").strip()
    query = {"game": str(target_user_game_id)}
    if sid:
        query["sid"] = sid
    return f"chat.html?{urlencode(query)}"


async def _roster_edit(
    heading: str = "Edit Roster",
) -> bool:
    while True:

        modal_result = await _roster_edit_modal()
        if modal_result is None:
            return False

        if modal_result.get("action") == "ok":
            return True

        slot_key = str(modal_result.get("slot_key") or "").strip()
        if not slot_key:
            return False

        state = str(modal_result.get("state") or "").strip().lower()
        if state not in {"empty", "ai", "human"}:
            continue

        display_name = None
        if state == "human":
            display_name = str(modal_result.get("display_name") or "").strip()
            if not display_name:
                continue
        bot_sid = None
        if state == "ai":
            bot_sid = await _bot_pick_dialog(
                title="Bot",
                heading="Select bot",
                current_bot_sid=str(modal_result.get("bot_sid") or ""),
            )
            if not bot_sid:
                return False

        await atlantis.client_command(
            "@roster_set_slot",
            {
                "slot_key": slot_key,
                "state": state,
                "display_name": display_name,
                "bot_sid": bot_sid,
            },
        )


@visible
async def roster_edit() -> bool:
    return await _roster_edit()


def _camera_current_target(game_key: str) -> Dict[str, str]:
    terminal_key = atlantis.get_terminal_key()
    if not terminal_key:
        return {}
    entry = _load_cameras(game_key).get(terminal_key)
    if not isinstance(entry, dict):
        return {}
    target_type = str(entry.get("target_type") or "").strip()
    if target_type == "location":
        return {
            "target_type": "location",
            "location": str(entry.get("location") or "").strip(),
        }
    if target_type == "slot":
        return {
            "target_type": "slot",
            "slot_key": str(entry.get("slot_key") or "").strip(),
        }
    return {}


def _camera_location_label(location: str) -> str:
    try:
        return load_location(location).get("displayName") or location
    except ValueError:
        return location


def _camera_target_descriptions(current: Dict[str, str]) -> Dict[str, str]:
    location_description = "Watch a fixed place in the scene."
    slot_description = "Follow a roster slot as it moves."
    target_type = current.get("target_type")
    if target_type == "location" and current.get("location"):
        location_description = f"Current: {_camera_location_label(current['location'])}"
    elif target_type == "slot" and current.get("slot_key"):
        slot_description = f"Current: {current['slot_key']}"
    return {
        "location": location_description,
        "slot": slot_description,
    }


async def _camera_mode_dialog(
    current: Dict[str, str],
    *,
    has_locations: bool,
    has_slots: bool,
) -> str:
    uid = uuid.uuid4().hex[:8]
    camera_mode_id = f"camera_mode:{uid}"
    camera_mode_id_js = json.dumps(camera_mode_id)
    target_descriptions = _camera_target_descriptions(current)
    current_target_type = str(current.get("target_type") or "")
    modes = [
        {
            "id": "location",
            "text": "Location",
            "description": target_descriptions["location"] if has_locations else "No locations available.",
            "enabled": has_locations,
        },
        {
            "id": "slot",
            "text": "Roster slot",
            "description": target_descriptions["slot"] if has_slots else "No roster slots available.",
            "enabled": has_slots,
        },
    ]
    enabled_modes = {mode["id"]: mode for mode in modes if mode["enabled"]}
    if not enabled_modes:
        raise RuntimeError("No camera targets found")
    if current_target_type in {"location", "slot"}:
        enabled_modes["__done__"] = {"id": "__done__", "enabled": True}

    cards = []
    for mode in modes:
        mode_id = str(mode["id"])
        is_active = mode_id == current_target_type
        disabled_attr = "" if mode["enabled"] else " disabled aria-disabled=\"true\""
        active_badge = '<span class="camera-mode-badge">Current mode</span>' if is_active else ""
        cards.append(
            f'<button type="button" class="camera-mode-card{" is-active" if is_active else ""}" '
            f'data-mode="{html_lib.escape(mode_id, quote=True)}"{disabled_attr}>'
            '<span class="camera-mode-topline">'
            '<span class="camera-mode-mark" aria-hidden="true"></span>'
            f'<span class="camera-mode-title">{html_lib.escape(str(mode["text"]))}</span>'
            f'{active_badge}'
            '</span>'
            f'<span class="camera-mode-copy">{html_lib.escape(str(mode["description"]))}</span>'
            '</button>'
        )

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    atlantis.session_shared.set(f"{camera_mode_id}:future", future)
    atlantis.session_shared.set(f"{camera_mode_id}:choices", enabled_modes)
    done_button = (
        '<div class="camera-mode-actions">'
        '<button type="button" class="camera-mode-done" data-mode="__done__">Done</button>'
        '</div>'
        if current_target_type in {"location", "slot"} else ""
    )
    html = f"""
<style>
{_modal_panel_css(
    f"#camera-mode-panel-{uid}",
    f"#cameramode-{uid}",
    ready_class="camera-mode-ready",
    padding=24,
    heading_margin="4px 0 16px",
    heading_font_size=24,
    heading_line_height=1.15,
)}
  #cameramode-{uid} {{
    width: 100%;
    min-width: 0;
    visibility: hidden;
  }}
  .jsPanel:has(#cameramode-{uid}) {{
    width: min(680px, calc(100vw - 32px)) !important;
    min-width: 0 !important;
    max-width: calc(100vw - 32px) !important;
    left: 50% !important;
    top: 50% !important;
    right: auto !important;
    bottom: auto !important;
    transform: translate(-50%, -50%) !important;
  }}
  #cameramode-{uid} .camera-mode-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    width: 100%;
  }}
  #cameramode-{uid} .camera-mode-card {{
    display: grid;
    align-content: start;
    gap: 12px;
    box-sizing: border-box;
    width: 100%;
    min-height: 132px;
    padding: 16px;
    color: #fffaf0;
    background-color: rgba(7, 15, 22, 0.88);
    background-image: linear-gradient(to bottom, rgba(7, 15, 22, 0.9), rgba(7, 15, 22, 0.74));
    border: 1px solid rgba(20, 255, 208, 0.34);
    border-radius: 6px;
    font: inherit;
    text-align: left;
    cursor: pointer;
  }}
  #cameramode-{uid} .camera-mode-card.is-active {{
    background-color: rgba(20, 255, 208, 0.16);
    background-image: linear-gradient(to bottom, rgba(20, 255, 208, 0.18), rgba(7, 15, 22, 0.64));
    border-color: rgba(20, 255, 208, 0.72);
  }}
  #cameramode-{uid} .camera-mode-card:hover,
  #cameramode-{uid} .camera-mode-card:focus {{
    background-color: rgba(20, 255, 208, 0.22);
    background-image: linear-gradient(to bottom, rgba(20, 255, 208, 0.22), rgba(7, 15, 22, 0.66));
    border-color: rgba(20, 255, 208, 0.82);
    outline: none;
  }}
  #cameramode-{uid} .camera-mode-card:disabled {{
    color: rgba(255, 250, 240, 0.42);
    background-color: rgba(7, 15, 22, 0.62);
    background-image: none;
    border-color: rgba(255, 250, 240, 0.14);
    cursor: default;
    opacity: 0.62;
  }}
  #cameramode-{uid} .camera-mode-topline {{
    display: grid;
    grid-template-columns: 20px minmax(0, 1fr) auto;
    align-items: flex-start;
    gap: 10px;
    min-width: 0;
  }}
  #cameramode-{uid} .camera-mode-mark {{
    box-sizing: border-box;
    width: 18px;
    height: 18px;
    margin-top: 2px;
    border: 2px solid rgba(255, 250, 240, 0.7);
    border-radius: 50%;
    background: rgba(7, 15, 22, 0.72);
  }}
  #cameramode-{uid} .camera-mode-card.is-active .camera-mode-mark {{
    border-color: #14ffd0;
    box-shadow: inset 0 0 0 4px rgba(7, 15, 22, 0.88);
    background: #14ffd0;
  }}
  #cameramode-{uid} .camera-mode-title {{
    min-width: 0;
    overflow-wrap: anywhere;
    font-size: 21px;
    font-weight: 800;
    line-height: 1.15;
  }}
  #cameramode-{uid} .camera-mode-badge {{
    flex: 0 0 auto;
    max-width: 48%;
    padding: 3px 7px;
    color: #071016;
    background: #14ffd0;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    line-height: 1.15;
    text-transform: uppercase;
  }}
  #cameramode-{uid} .camera-mode-copy {{
    min-width: 0;
    overflow-wrap: anywhere;
    color: rgba(255, 250, 240, 0.72);
    font-size: 14px;
    line-height: 1.35;
  }}
  #cameramode-{uid} .camera-mode-actions {{
    display: flex;
    justify-content: center;
    margin-top: 18px;
  }}
  #cameramode-{uid} .camera-mode-done {{
    box-sizing: border-box;
    min-width: 96px;
    min-height: 40px;
    padding: 0 16px;
    color: #fffaf0;
    background: rgba(20, 255, 208, 0.16);
    border: 1px solid rgba(20, 255, 208, 0.62);
    border-radius: 6px;
    font: inherit;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
  }}
  #cameramode-{uid} .camera-mode-done:hover,
  #cameramode-{uid} .camera-mode-done:focus {{
    background: rgba(20, 255, 208, 0.22);
    border-color: rgba(20, 255, 208, 0.78);
    outline: none;
  }}
  @media (max-width: 560px) {{
    #cameramode-{uid} .camera-mode-grid {{
      grid-template-columns: 1fr;
    }}
    #cameramode-{uid} .camera-mode-card {{
      min-height: 104px;
    }}
  }}
</style>
<section id="cameramode-{uid}" aria-label="Camera mode">
  <h2>Camera</h2>
  <div class="camera-mode-grid">
    {"".join(cards)}
  </div>
  {done_button}
</section>
"""
    modal_id = await atlantis.client_modal(html, title="Camera")
    atlantis.session_shared.set(f"{camera_mode_id}:modal_id", modal_id)
    exec_shell_js = json.dumps(atlantis.get_exec_shell_path())
    script = f"""
(function() {{
  var settled = false;
  var observer = null;
  function cleanup() {{ if (observer) {{ try {{ observer.disconnect(); }} catch (e) {{}} observer = null; }} }}
  function reveal(root) {{
    root.style.visibility = "visible";
    root.classList.add("camera-mode-ready");
  }}
  function cancel() {{
    if (settled) return;
    settled = true;
    cleanup();
    if (!window._accessToken) return;
    sendChatter(window._accessToken, "@camera_mode_cancel", {{
      camera_mode_id: {camera_mode_id_js}
    }}, {exec_shell_js}).catch(function() {{}});
  }}
  function bind() {{
    var root = document.getElementById("cameramode-{uid}");
    if (!root) return;
    reveal(root);
    var buttons = Array.prototype.slice.call(root.querySelectorAll(".camera-mode-card, .camera-mode-done"));
    var enabledButtons = buttons.filter(function(button) {{ return !button.disabled; }});
    if (enabledButtons[0]) enabledButtons[0].focus({{ preventScroll: true }});
    buttons.forEach(function(button) {{
      button.addEventListener("click", function() {{
        if (settled || button.disabled || !window._accessToken) return;
        settled = true;
        cleanup();
        buttons.forEach(function(btn) {{ btn.disabled = true; }});
        sendChatter(window._accessToken, "@camera_mode_select", {{
          camera_mode_id: {camera_mode_id_js},
          mode: button.getAttribute("data-mode") || ""
        }}, {exec_shell_js}).catch(function() {{}});
      }});
      button.addEventListener("keydown", function(event) {{
        if (button.disabled) return;
        var index = enabledButtons.indexOf(button);
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {{
          event.preventDefault();
          enabledButtons[(index + 1) % enabledButtons.length].focus({{ preventScroll: true }});
        }} else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {{
          event.preventDefault();
          enabledButtons[(index + enabledButtons.length - 1) % enabledButtons.length].focus({{ preventScroll: true }});
        }} else if (event.key === "Enter" || event.key === " ") {{
          event.preventDefault();
          button.click();
        }} else if (event.key === "Escape") {{
          event.preventDefault();
          cancel();
        }}
      }});
    }});
    observer = new MutationObserver(function() {{
      if (!document.body.contains(root)) cancel();
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
        atlantis.session_shared.remove(f"{camera_mode_id}:future")
        atlantis.session_shared.remove(f"{camera_mode_id}:modal_id")
        atlantis.session_shared.remove(f"{camera_mode_id}:choices")


def _camera_slot_choices(roster: list, current: Dict[str, str]) -> list[Dict[str, str]]:
    choices = []
    current_slot = current.get("slot_key") if current.get("target_type") == "slot" else ""
    for row in roster:
        slot_key = str(row.get("key") or "").strip()
        if not slot_key:
            continue
        state = _roster_row_state(row)
        name = _roster_row_label(row)
        label = slot_key
        current_marker = "Current" if slot_key == current_slot else ""
        details = [state]
        if name and name != "-":
            details.append(name)
        bot_sid = str(row.get("bot_sid") or "").strip()
        if bot_sid and bot_sid not in details:
            details.append(bot_sid)
        location = str(row.get("location") or "").strip()
        if location:
            details.append(f"at {_camera_location_label(location)}")
        if current_marker:
            details.append(current_marker.lower())
        choices.append({
            "id": slot_key,
            "text": label,
            "description": " - ".join(details),
            "slot_key": slot_key,
        })
    return choices


async def _camera_edit(roster: Optional[list] = None, game_key: Optional[str] = None) -> bool:
    game_key = str(game_key or await game_find_current()).strip()
    roster_rows = roster if roster is not None else await atlantis.client_command("@roster_list")
    current = _camera_current_target(game_key)
    has_locations = bool(_leaf_location_keys())
    slot_choices = _camera_slot_choices(roster_rows, current)
    if not has_locations and not slot_choices:
        raise RuntimeError("No camera targets found")

    while True:
        current = _camera_current_target(game_key)
        terminal_key = atlantis.get_terminal_key() or ""
        cameras = _load_cameras(game_key)
        await atlantis.client_log(
            "camera_edit redraw "
            f"game_key={game_key!r} session_key={atlantis.get_session_key()!r} "
            f"terminal_key={terminal_key!r} camera_keys={sorted(cameras.keys())!r} "
            f"current={current!r}"
        )
        slot_choices = _camera_slot_choices(roster_rows, current)
        try:
            mode = await _camera_mode_dialog(
                current,
                has_locations=has_locations,
                has_slots=bool(slot_choices),
            )
        except ModalGoBack:
            return False
        mode = str(mode or "").strip().lower()
        if mode == "__done__":
            return True

        if mode == "location":
            current_location = ""
            if current.get("target_type") == "location":
                current_location = str(current.get("location") or "")
            location = await _location_pick_dialog(
                title="Location",
                heading="Select location",
                current_location=current_location,
            )
            if not location:
                continue
            await camera_bind(game_key, location)
            continue

        if mode == "slot":
            current_slot = current.get("slot_key") if current.get("target_type") == "slot" else ""
            try:
                choice = await modal_radio(
                    slot_choices,
                    title="Camera",
                    heading="Select roster slot",
                    current_id=str(current_slot or ""),
                )
            except ModalGoBack:
                continue
            slot_key = str(choice.get("slot_key") or choice.get("id") or "").strip()
            if not slot_key:
                continue
            await camera_follow(game_key, slot_key)
            continue
        raise ValueError(f"Unknown camera target mode: {mode!r}")


@visible
async def camera_edit() -> bool:
    return await _camera_edit()


@public
@visible
async def camera_mode_select(camera_mode_id: str, mode: str) -> None:
    mode = str(mode or "").strip()
    modal_key = f"{camera_mode_id}:modal_id"
    future_key = f"{camera_mode_id}:future"
    choices_key = f"{camera_mode_id}:choices"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove(modal_key)
    choices = atlantis.session_shared.get(choices_key) or {}
    if mode not in choices:
        raise ValueError(f"Unknown camera mode: {mode!r}")
    future = atlantis.session_shared.get(future_key)
    if future is None:
        raise ValueError("Camera mode modal is no longer active")
    if not future.done():
        future.set_result(mode)


@public
@visible
async def camera_mode_cancel(camera_mode_id: str) -> None:
    modal_key = f"{camera_mode_id}:modal_id"
    future_key = f"{camera_mode_id}:future"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        try:
            await atlantis.client_modal_close(modal_id)
        except Exception:
            pass
        atlantis.session_shared.remove(modal_key)
    future = atlantis.session_shared.get(future_key)
    if future is not None and not future.done():
        future.set_exception(ModalGoBack("Camera mode selection cancelled"))


@public
async def game_new() -> Dict[str, Any]:
    """Main entry point for creating a new game"""
    for _ in range(10):
        game_key = uuid.uuid4().hex
        data_dir = game_dir(game_key)
        if not os.path.exists(data_dir):
            break
    else:
        raise RuntimeError("Unable to allocate a unique game_key")

    await atlantis.client_log("Creating new game")

    join_password = uuid.uuid4().hex
    _game_create(game_key, {
        'join_password': join_password,
        'owner': atlantis.get_caller() or None,
        'user_game_id': atlantis.get_user_game_id(),
        'state': GAME_STATE_STOPPED,
        'roster_scene': None,
        'roster_created_at': None,
        'members': add_caller_membership({}),
    })

    await atlantis.client_log(f"Game created: {game_key}")
    await app_bg_default()

    return {
        "game_key": game_key,
        "join_password": join_password,
    }


@public
async def game_join(require_other_owner: bool = False) -> Dict[str, Any]:
    """Join existing game"""
    entered_game_key = await modal_string(
        "Enter game key:",
        submit_label="Next",
        title="Game",
        submitting_label="Checking...",
        empty_error="Enter the game key to continue.",
        autocomplete="off",
    )
    if entered_game_key is None:
        return {"cancelled": True}
    game_key = str(entered_game_key or "").strip()
    if not game_key:
        raise ValueError("game_key required")

    meta = _game_read(game_key)
    if require_other_owner and atlantis.get_caller() == meta.get("owner", ""):
        raise PermissionError("Join game requires a game owned by someone else")
    return await _game_join_or_prompt(game_key, meta)


async def _game_join_or_prompt(game_key: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    session_key = atlantis.get_session_key()
    if not session_key:
        raise RuntimeError("No session key in this call context")
    if _caller_is_member(meta, session_key) or atlantis.get_caller() == meta.get("owner", ""):
        return await _game_join_authorized(game_key, meta)

    while True:
        password = await modal_string(
            "Enter game password:",
            submit_label="Join",
            title=f"Game {game_key}",
            submitting_label="Joining...",
            empty_error="Enter the password to continue.",
            input_type="password",
            autocomplete="current-password",
        )
        if password is None:
            return {"cancelled": True}

        if meta.get('join_password') == password:
            return await _game_join_authorized(game_key, meta)

        await _game_password_error(game_key)


@visible
async def _game_password_error(game_key: str) -> None:
    await modal_confirm(
        "Incorrect password",
        title=f"Game {game_key}",
        cancel_label="",
    )


def _game_candidates(games: list, action: str) -> list:
    if action not in {"join", "resume"}:
        raise ValueError(f"Unknown game candidate action: {action!r}")

    session_key = atlantis.get_session_key()
    if not session_key:
        raise RuntimeError("No session key in this call context")
    caller = atlantis.get_caller()
    candidates = []
    for game in games:
        game_key = str(game.get("game_key") or "").strip()
        if not game_key:
            continue
        meta = _game_read(game_key)
        is_member = _caller_is_member(meta, session_key)
        is_owner = bool(caller and meta.get("owner") == caller)
        if action == "join" and not is_member and not is_owner:
            candidates.append(game)
        elif action == "resume" and is_member:
            candidates.append(game)
    return candidates


async def _game_join_authorized(game_key: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    members = meta.setdefault('members', {})
    session_key = atlantis.get_session_key()
    if not session_key:
        raise RuntimeError("No session key in this call context")
    already_member = session_key in members
    if not already_member:
        add_caller_membership(members)
    _game_update(game_key, meta)
    if already_member:
        await atlantis.client_log(f"user already in game: {game_key}")
    else:
        await atlantis.client_log(
            f"✅ {atlantis.get_caller() or atlantis.get_session_key()} joined game {game_key}"
        )
    await _game_enter(game_key)
    return {"game_key": game_key}


@public
async def first_menu() -> str:
    """To the bots"""
    games = _game_rows()
    joinable_games = _game_candidates(games, "join")
    resumable_games = _game_candidates(games, "resume")
    choices: list[Dict[str, Any]] = [{"id": "create", "text": "Create new game"}]
    if resumable_games:
        choices.append({"id": "resume", "text": "Resume existing game"})
    choices.append({
        "id": "join",
        "text": "Join game",
        "disabled": not joinable_games,
    })

    choice = await modal_menu(
        choices,
        title="Game Action",
        heading="What do you want to do?",
    )
    if choice is None:
        raise RuntimeError("Game selection cancelled")

    await atlantis.client_log(f"choice selected: {choice.get('id')!r}")

    if choice.get("id") == "create":
        keys = await game_new()
        game_key = str(keys.get("game_key") or "").strip()
        if not game_key:
            raise RuntimeError("Game create did not return a game_key")
        await atlantis.client_command("/cursor join", keys)
        await game_init(game_key)
        return game_key

    if choice.get("id") == "join":
        game_key = await _game_pick_dialog(
            games=joinable_games,
            heading="Choose a game to join",
        )
        if not game_key:
            raise RuntimeError("Game selection cancelled")
        meta = _game_read(game_key)
        result = await _game_join_or_prompt(game_key, meta)
        if result.get("cancelled"):
            raise RuntimeError("Game selection cancelled")
        game_key = str(result.get("game_key") or "").strip()
        if not game_key:
            raise RuntimeError("Game join did not return a game_key")
        return game_key

    if choice.get("id") == "resume":
        game_key = await _game_pick_dialog(games=resumable_games, heading="Choose a game to resume")
        if not game_key:
            raise RuntimeError("Game selection cancelled")
        return await _game_resume(game_key)

    raise ValueError(f"Unknown game choice: {choice.get('id')!r}")


@public
async def game_init(game_key: str):

    await atlantis.client_log("*** GAME INIT ***")

    """Idempotent game setup"""

    # % atlantis "get_session_key"


    # make sure game exists
    data_dir = require_membership(game_key)
    session_key = atlantis.get_session_key()
    if not session_key:
        raise RuntimeError("No session key in this call context")
    roster_path = os.path.join(data_dir, "roster.json")

    # make sure chat callback is set
    callbacks = await atlantis.client_command("/callback list")
    chat_row = next(row for row in callbacks if row["mode"] == "chat")

    if not chat_row["toolPath"]:
        await atlantis.client_command("callback set preflight auto")
        await atlantis.client_command("callback set chat auto")

        callbacks = await atlantis.client_command("/callback list")
        chat_row = next(row for row in callbacks if row["mode"] == "chat")



    await atlantis.client_log(f"chat callback: toolPath={chat_row['toolPath']!r} filename={chat_row['filename']!r}")

    if os.path.isfile(roster_path):
        roster = await atlantis.client_command("@roster_list")
    else:
        scene = await _scene_pick_dialog()
        if not scene:
            raise RuntimeError("Scene selection cancelled")
        roster = await atlantis.client_command(f"@roster_create {scene}")
        await atlantis.client_log(f"game scene: {scene!r}")

    if not await roster_edit():
        raise RuntimeError("Roster selection cancelled")
    roster = await atlantis.client_command("@roster_list")
    if roster and all(row.get("state") == "Empty" for row in roster):
        await _warn_empty_roster()

    if not await _camera_edit(roster=roster, game_key=game_key):
        raise RuntimeError("Camera selection cancelled")

    meta = _game_read_from_dir(data_dir)
    if (
        atlantis.get_caller() == meta.get("owner")
        and str(meta.get("state") or GAME_STATE_STOPPED).strip().lower() == GAME_STATE_STOPPED
    ):
        choice = await modal_menu(
            [
                {"id": "start", "text": "Start game"},
                {"id": "keep_stopped", "text": "Keep stopped"},
            ],
            title="Game",
            heading="Start the game?",
        )
        if choice is None:
            raise RuntimeError("Game start selection cancelled")
        if choice and choice.get("id") == "start":
            await game_start(game_key)
            meta = _game_read_from_dir(data_dir)
