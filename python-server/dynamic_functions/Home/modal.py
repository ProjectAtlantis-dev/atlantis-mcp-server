"""Modal UI tools"""

import atlantis
import asyncio
import html as html_lib
import json
import uuid
from typing import Any, Dict, List, Optional


class ModalGoBack(RuntimeError):
    """Raised when the user chooses Go back in a modal flow."""


def _modal_shell_css(
    selector: str,
    *,
    padding: int,
    heading_margin: str,
    heading_font_size: int,
    heading_line_height: float,
) -> str:
    return f"""
  {selector} {{
    box-sizing: border-box;
    width: 100%;
    min-width: min(100%, 320px);
    padding: {padding}px;
    color: #f7f4ea;
    background:
      linear-gradient(to bottom, rgba(20, 34, 48, 0.96), rgba(20, 50, 60, 0.96)),
      radial-gradient(circle at 18% 20%, rgba(20, 255, 208, 0.22), transparent 34%);
    border: 1px solid rgba(20, 255, 208, 0.42);
    border-radius: 8px;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  {selector} h2 {{
    margin: {heading_margin};
    font-size: {heading_font_size}px;
    line-height: {heading_line_height};
    color: #fffaf0;
  }}
"""


def _modal_panel_css(
    host_selector: str,
    shell_selector: str,
    *,
    ready_class: str,
    padding: int,
    heading_margin: str,
    heading_font_size: int,
    heading_line_height: float,
) -> str:
    return f"""
  .jsPanel:has({shell_selector}:not(.{ready_class})) {{
    visibility: hidden;
  }}
  {host_selector},
  .jsPanel:has({shell_selector}) {{
    background:
      linear-gradient(to bottom, rgba(20, 34, 48, 0.96), rgba(20, 50, 60, 0.96)),
      radial-gradient(circle at 18% 20%, rgba(20, 255, 208, 0.22), transparent 34%) !important;
    background-color: #142230 !important;
  }}
{_modal_shell_css(shell_selector, padding=padding, heading_margin=heading_margin, heading_font_size=heading_font_size, heading_line_height=heading_line_height)}
"""


def _validated_modal_choices(choices: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a non-empty array")

    choice_by_id: Dict[str, Dict[str, Any]] = {}
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("each choice must be an object")
        choice_id = str(choice.get("id", "")).strip()
        choice_text = str(choice.get("text", "")).strip()
        if not choice_id:
            raise ValueError("each choice requires a non-empty id")
        if not choice_text:
            raise ValueError("each choice requires a non-empty text")
        if choice_id in choice_by_id:
            raise ValueError(f"duplicate choice id: {choice_id!r}")
        choice_by_id[choice_id] = choice
    return choice_by_id


async def _close_modal_if_open(shared_prefix: str) -> None:
    """Best-effort modal close for abnormal exits (errors, cancellation).

    On the normal path the settle handlers close the modal and drop the shared
    modal_id key before the future resolves, so this is a no-op. If the key is
    still present here, the modal is still up and must not be left behind.
    (Backdrop blur is script-owned, not modal-owned — callers manage it with
    their own try/finally, e.g. first_menu.)
    """
    modal_key = f"{shared_prefix}:modal_id"
    modal_id = atlantis.session_shared.get(modal_key)
    if not modal_id:
        return
    try:
        await atlantis.client_modal_close(modal_id)
    except BaseException:
        # Swallow even CancelledError: we're on the unwind path, the original
        # exception resumes propagating once this attempt finishes.
        pass
    atlantis.session_shared.remove(modal_key)


@public
@visible
async def modal_string(
    modal_text: str,
    title: str = "",
    heading: str = "",
    submit_label: str = "Submit",
    submitting_label: str = "Submitting...",
    empty_error: str = "Enter a value to continue.",
    input_type: str = "text",
    autocomplete: str = "off",
) -> Optional[str]:
    """Pop up a modal asking the caller for a string.

    Returns None if the user closes/cancels the modal without submitting.
    """
    uid = uuid.uuid4().hex[:8]
    modal_string_id = f"modal_string:{uid}"
    modal_string_id_js = json.dumps(modal_string_id)
    modal_text_html = html_lib.escape(modal_text or "Enter a value")
    heading_block = f"<h2>{html_lib.escape(heading)}</h2>" if heading else ""
    submit_label_html = html_lib.escape(submit_label)
    submitting_label_js = json.dumps(submitting_label)
    empty_error_js = json.dumps(empty_error)
    input_type_html = html_lib.escape(input_type)
    autocomplete_html = html_lib.escape(autocomplete)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    atlantis.session_shared.set(f"{modal_string_id}:future", future)
    html = f"""
<style>
{_modal_panel_css(
    f"#modal-string-panel-{uid}",
    f"#displayname-{uid}",
    ready_class="modal-string-ready",
    padding=28,
    heading_margin="10px 0 28px",
    heading_font_size=30,
    heading_line_height=1.1,
)}
  #displayname-{uid} {{
    width: 100%;
    visibility: hidden;
  }}
  #displayname-{uid} form {{
    display: grid;
    gap: 12px;
    width: 100%;
  }}
  #displayname-{uid} label {{
    color: #fffaf0;
    font-size: 22px;
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
    font-size: 20px;
  }}
  #displayname-{uid} input:focus {{
    outline: 2px solid rgba(20, 255, 208, 0.45);
    outline-offset: 2px;
  }}
  #displayname-{uid} .err {{
    color: #ffb4a8;
    font-size: 13px;
  }}
  #displayname-{uid} .err:empty {{
    display: none;
  }}
  #displayname-{uid} button {{
    justify-self: center;
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
<section id="displayname-{uid}" aria-label="Input">
  {heading_block}
  <form id="displayname-form-{uid}">
    <label for="displayname-input-{uid}">{modal_text_html}</label>
    <input id="displayname-input-{uid}" name="value" type="{input_type_html}" autocomplete="{autocomplete_html}" maxlength="200" required autofocus>
    <div id="displayname-err-{uid}" class="err" aria-live="polite"></div>
    <button id="displayname-btn-{uid}" type="submit">{submit_label_html}</button>
  </form>
</section>
"""
    modal_id = await atlantis.client_modal(html, title=title or " ")
    atlantis.session_shared.set(f"{modal_string_id}:modal_id", modal_id)

    # Route modal-originated commands (cancel / submit) to this tool call's
    # exec shell so they nest inside the modal_string subshell rather than
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
    #     below, modal_string_click/_cancel will be routed to the user's main
    #     shell ("2.X") instead of "37.1". Visible symptom: the click event
    #     shows up as a sibling of the modal_string command in /history.
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
  function reveal(root) {{
    root.style.visibility = "visible";
    root.classList.add("modal-string-ready");
  }}
  function markHost(host) {{
    if (!host) return;
    if (!host.id) host.id = "modal-string-panel-{uid}";
    host.classList.add("modal-string-panel");
    host.dataset.modalKind = "string";
    host.dataset.modalStringUid = "{uid}";
  }}
  function centerDialog(root) {{
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
      reveal(root);
      return;
    }}
    markHost(host);
    var rect = host.getBoundingClientRect();
    if (!rect.width || !rect.height) {{
      reveal(root);
      return;
    }}
    if (!host.dataset.modalStringOriginalWidth) {{
      host.dataset.modalStringOriginalWidth = String(rect.width);
    }}
    var originalWidth = Number(host.dataset.modalStringOriginalWidth) || rect.width;
    var targetWidth = Math.round(originalWidth * 0.5);
    var viewportMax = Math.max(320, window.innerWidth - 32);
    host.style.width = Math.min(Math.max(320, targetWidth), viewportMax) + "px";
    host.style.maxWidth = "calc(100vw - 32px)";
    host.style.left = "50%";
    host.style.top = "50%";
    host.style.right = "auto";
    host.style.bottom = "auto";
    host.style.transform = "translate(-50%, -50%)";
    host.style.margin = "0";
    reveal(root);
  }}
  function scheduleCenter(root) {{
    centerDialog(root);
    requestAnimationFrame(function() {{
      centerDialog(root);
      setTimeout(function() {{ centerDialog(root); }}, 180);
    }});
  }}
  async function cancel() {{
    if (settled) return;
    settled = true;
    cleanup();
    if (!window._accessToken) return;
    try {{
      await sendChatter(window._accessToken, "@modal_string_cancel", {{
        modal_string_id: {modal_string_id_js}
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
    scheduleCenter(root);
    focusInput();
    setTimeout(function() {{ scheduleCenter(root); focusInput(); }}, 120);
    form.addEventListener("submit", async function(event) {{
      event.preventDefault();
      if (settled) return;
      if (!window._accessToken) return;
      var value = input.value.trim();
      if (!value) {{ if (error) error.textContent = {empty_error_js}; input.focus(); return; }}
      if (error) error.textContent = "";
      button.disabled = true;
      button.textContent = {submitting_label_js};
      settled = true;
      cleanup();
      await sendChatter(window._accessToken, "@modal_string_click", {{
        modal_string_id: {modal_string_id_js},
        display_name: value
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
    try:
        await atlantis.client_script(script)
        return await future
    finally:
        await _close_modal_if_open(modal_string_id)
        atlantis.session_shared.remove(f"{modal_string_id}:future")


@public
@visible
async def modal_confirm(
    message: str,
    title: str = "",
    heading: str = "",
    ok_label: str = "Continue",
    cancel_label: str = "Go back",
) -> bool:
    """Pop up a centered confirmation modal.

    Returns True for Continue, False for Go back or closing the modal.
    """
    uid = uuid.uuid4().hex[:8]
    modal_confirm_id = f"modal_confirm:{uid}"
    modal_confirm_id_js = json.dumps(modal_confirm_id)
    message_html = html_lib.escape(message or "")
    heading_block = f"<h2>{html_lib.escape(heading)}</h2>" if heading else ""
    ok_label_html = html_lib.escape(ok_label or "Continue")
    cancel_label_text = str(cancel_label or "").strip()
    cancel_label_html = html_lib.escape(cancel_label_text)
    cancel_button_html = (
        f'<button type="button" class="confirm-button confirm-cancel">{cancel_label_html}</button>'
        if cancel_label_text else ""
    )
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    atlantis.session_shared.set(f"{modal_confirm_id}:future", future)
    atlantis.session_shared.set(f"{modal_confirm_id}:cancel_raises", bool(cancel_label_text))
    html = f"""
<style>
{_modal_panel_css(
    f"#modal-confirm-panel-{uid}",
    f"#modalconfirm-{uid}",
    ready_class="modal-confirm-ready",
    padding=26,
    heading_margin="4px 0 12px",
    heading_font_size=24,
    heading_line_height=1.15,
)}
  #modalconfirm-{uid} {{
    width: 100%;
    min-width: 0;
    visibility: hidden;
    text-align: center;
  }}
  .jsPanel:has(#modalconfirm-{uid}) {{
    width: min(420px, calc(100vw - 32px)) !important;
    min-width: 0 !important;
    max-width: calc(100vw - 32px) !important;
    left: 50% !important;
    top: 50% !important;
    right: auto !important;
    bottom: auto !important;
    transform: translate(-50%, -50%) !important;
  }}
  #modalconfirm-{uid} .confirm-message {{
    margin: 0;
    color: rgba(255, 250, 240, 0.86);
    font-size: 18px;
    line-height: 1.35;
  }}
  #modalconfirm-{uid} .confirm-actions {{
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 22px;
  }}
  #modalconfirm-{uid} .confirm-button {{
    box-sizing: border-box;
    min-width: 96px;
    min-height: 40px;
    padding: 0 16px;
    color: #fffaf0;
    background: rgba(7, 15, 22, 0.58);
    border: 1px solid rgba(20, 255, 208, 0.34);
    border-radius: 6px;
    font: inherit;
    font-size: 16px;
    font-weight: 700;
    text-align: center;
    cursor: pointer;
  }}
  #modalconfirm-{uid} .confirm-ok {{
    background: rgba(20, 255, 208, 0.16);
    border-color: rgba(20, 255, 208, 0.62);
  }}
  #modalconfirm-{uid} .confirm-button:hover,
  #modalconfirm-{uid} .confirm-button:focus {{
    background: rgba(20, 255, 208, 0.22);
    border-color: rgba(20, 255, 208, 0.78);
    outline: none;
  }}
  #modalconfirm-{uid} .confirm-button:disabled {{
    cursor: default;
    opacity: 0.55;
  }}
</style>
<section id="modalconfirm-{uid}" aria-label="Confirmation">
  {heading_block}
  <p class="confirm-message">{message_html}</p>
  <div class="confirm-actions">
    {cancel_button_html}
    <button type="button" class="confirm-button confirm-ok">{ok_label_html}</button>
  </div>
</section>
"""
    modal_id = await atlantis.client_modal(html, title=title or " ")
    atlantis.session_shared.set(f"{modal_confirm_id}:modal_id", modal_id)
    exec_shell_js = json.dumps(atlantis.get_exec_shell_path())

    script = f"""
(function() {{
  var settled = false;
  var observer = null;
  function cleanup() {{ if (observer) {{ try {{ observer.disconnect(); }} catch (e) {{}} observer = null; }} }}
  function markHost(host) {{
    if (!host) return;
    if (!host.id) host.id = "modal-confirm-panel-{uid}";
    host.classList.add("modal-confirm-panel");
    host.dataset.modalKind = "confirm";
    host.dataset.modalConfirmUid = "{uid}";
  }}
  function center(root) {{
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
    if (host) markHost(host);
    root.style.visibility = "visible";
    root.classList.add("modal-confirm-ready");
  }}
  async function settle(action) {{
    if (settled) return;
    settled = true;
    cleanup();
    if (!window._accessToken) return;
    try {{
      await sendChatter(window._accessToken, action, {{
        modal_confirm_id: {modal_confirm_id_js}
      }}, {exec_shell_js});
    }} catch (e) {{}}
  }}
  function bind() {{
    var root = document.getElementById("modalconfirm-{uid}");
    if (!root) return;
    center(root);
    var ok = root.querySelector(".confirm-ok");
    var cancel = root.querySelector(".confirm-cancel");
    var buttons = Array.prototype.slice.call(root.querySelectorAll(".confirm-button"));
    if (ok) {{
      ok.focus({{ preventScroll: true }});
      ok.addEventListener("click", function() {{
        buttons.forEach(function(button) {{ button.disabled = true; }});
        settle("@modal_confirm_ok");
      }});
    }}
    if (cancel) {{
      cancel.addEventListener("click", function() {{
        buttons.forEach(function(button) {{ button.disabled = true; }});
        settle("@modal_confirm_cancel");
      }});
    }}
    root.addEventListener("keydown", function(event) {{
      if (event.key === "Escape") {{
        event.preventDefault();
        settle("@modal_confirm_cancel");
      }}
    }});
    observer = new MutationObserver(function() {{
      if (!document.body.contains(root)) {{ settle("@modal_confirm_cancel"); }}
    }});
    observer.observe(document.body, {{ childList: true, subtree: true }});
  }}
  requestAnimationFrame(function() {{ requestAnimationFrame(bind); }});
}})()
"""
    try:
        await atlantis.client_script(script)
        return await future
    finally:
        await _close_modal_if_open(modal_confirm_id)
        atlantis.session_shared.remove(f"{modal_confirm_id}:future")
        atlantis.session_shared.remove(f"{modal_confirm_id}:cancel_raises")


@public
@visible
async def modal_radio(
    choices: List[Dict[str, Any]],
    title: str = "",
    heading: str = "",
    current_id: str = "",
    ok_label: str = "Continue",
    cancel_label: str = "Go back",
    require_selection: bool = True,
) -> Dict[str, Any]:
    """Pop up a radio-choice modal and return the selected choice object."""
    choice_by_id = _validated_modal_choices(choices)
    current_id = str(current_id or "").strip()
    if current_id not in choice_by_id and require_selection:
        current_id = next(
            (
                str(choice.get("id") or "").strip()
                for choice in choices
                if not choice.get("disabled")
            ),
            "",
        )
    elif current_id not in choice_by_id:
        current_id = ""
    if require_selection and not current_id:
        raise ValueError("radio choices must include at least one enabled choice")

    uid = uuid.uuid4().hex[:8]
    modal_radio_id = f"modal_radio:{uid}"
    modal_radio_id_js = json.dumps(modal_radio_id)
    heading_block = f"<h2>{html_lib.escape(heading)}</h2>" if heading else ""
    ok_label_html = html_lib.escape(ok_label or "Continue")
    cancel_label_text = str(cancel_label or "").strip()
    cancel_label_html = html_lib.escape(cancel_label_text)
    cancel_button_html = (
        f'<button type="button" class="radio-button radio-cancel">{cancel_label_html}</button>'
        if cancel_label_text else ""
    )
    radio_items = []
    for choice in choices:
        choice_id = str(choice.get("id") or "").strip()
        choice_text = str(choice.get("text") or "").strip()
        disabled = bool(choice.get("disabled"))
        checked_attr = " checked" if choice_id == current_id else ""
        disabled_attr = " disabled aria-disabled=\"true\"" if disabled else ""
        description = str(choice.get("description") or "").strip()
        description_html = (
            f'<span class="radio-description">{html_lib.escape(description)}</span>'
            if description else ""
        )
        radio_items.append(
            f'<label class="radio-choice{" is-disabled" if disabled else ""}">'
            f'<input type="radio" name="modal-radio-{uid}" value="{html_lib.escape(choice_id, quote=True)}"{checked_attr}{disabled_attr}>'
            '<span class="radio-mark" aria-hidden="true"></span>'
            '<span class="radio-copy">'
            f'<span class="radio-text">{html_lib.escape(choice_text)}</span>'
            f'{description_html}'
            '</span>'
            '</label>'
        )

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    atlantis.session_shared.set(f"{modal_radio_id}:future", future)
    atlantis.session_shared.set(f"{modal_radio_id}:choices", choice_by_id)
    html = f"""
<style>
{_modal_panel_css(
    f"#modal-radio-panel-{uid}",
    f"#modalradio-{uid}",
    ready_class="modal-radio-ready",
    padding=24,
    heading_margin="4px 0 16px",
    heading_font_size=24,
    heading_line_height=1.15,
)}
  #modalradio-{uid} {{
    width: 100%;
    min-width: 0;
    visibility: hidden;
  }}
  .jsPanel:has(#modalradio-{uid}) {{
    width: min(460px, calc(100vw - 32px)) !important;
    min-width: 0 !important;
    max-width: calc(100vw - 32px) !important;
    left: 50% !important;
    top: 50% !important;
    right: auto !important;
    bottom: auto !important;
    transform: translate(-50%, -50%) !important;
  }}
  #modalradio-{uid} .radio-list {{
    display: grid;
    gap: 8px;
    width: 100%;
  }}
  #modalradio-{uid} .radio-choice {{
    display: grid;
    grid-template-columns: 22px minmax(0, 1fr);
    gap: 12px;
    align-items: center;
    box-sizing: border-box;
    width: 100%;
    min-height: 44px;
    padding: 10px 12px;
    color: #fffaf0;
    background: rgba(7, 15, 22, 0.48);
    border: 1px solid rgba(20, 255, 208, 0.28);
    border-radius: 6px;
    cursor: pointer;
  }}
  #modalradio-{uid} .radio-choice:has(input:checked) {{
    background: rgba(20, 255, 208, 0.16);
    border-color: rgba(20, 255, 208, 0.7);
  }}
  #modalradio-{uid} .radio-choice:focus-within,
  #modalradio-{uid} .radio-choice:hover {{
    border-color: rgba(20, 255, 208, 0.72);
    outline: none;
  }}
  #modalradio-{uid} .radio-choice.is-disabled {{
    cursor: default;
    opacity: 0.5;
  }}
  #modalradio-{uid} input[type="radio"] {{
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }}
  #modalradio-{uid} .radio-mark {{
    box-sizing: border-box;
    width: 18px;
    height: 18px;
    border: 2px solid rgba(255, 250, 240, 0.7);
    border-radius: 50%;
    background: rgba(7, 15, 22, 0.42);
  }}
  #modalradio-{uid} .radio-choice:has(input:checked) .radio-mark {{
    border-color: #14ffd0;
    box-shadow: inset 0 0 0 4px rgba(7, 15, 22, 0.88);
    background: #14ffd0;
  }}
  #modalradio-{uid} .radio-copy {{
    display: grid;
    gap: 3px;
    min-width: 0;
  }}
  #modalradio-{uid} .radio-text {{
    min-width: 0;
    overflow-wrap: anywhere;
    font-size: 18px;
    font-weight: 700;
    line-height: 1.2;
  }}
  #modalradio-{uid} .radio-description {{
    min-width: 0;
    overflow-wrap: anywhere;
    color: rgba(255, 250, 240, 0.68);
    font-size: 13px;
    line-height: 1.25;
  }}
  #modalradio-{uid} .radio-actions {{
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 20px;
  }}
  #modalradio-{uid} .radio-button {{
    box-sizing: border-box;
    min-width: 96px;
    min-height: 40px;
    padding: 0 16px;
    color: #fffaf0;
    background: rgba(7, 15, 22, 0.58);
    border: 1px solid rgba(20, 255, 208, 0.34);
    border-radius: 6px;
    font: inherit;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
  }}
  #modalradio-{uid} .radio-ok {{
    background: rgba(20, 255, 208, 0.16);
    border-color: rgba(20, 255, 208, 0.62);
  }}
  #modalradio-{uid} .radio-button:hover,
  #modalradio-{uid} .radio-button:focus {{
    background: rgba(20, 255, 208, 0.22);
    border-color: rgba(20, 255, 208, 0.78);
    outline: none;
  }}
</style>
<section id="modalradio-{uid}" aria-label="Choose an option">
  {heading_block}
  <div class="radio-list" role="radiogroup">
    {"".join(radio_items)}
  </div>
  <div class="radio-actions">
    {cancel_button_html}
    <button type="button" class="radio-button radio-ok">{ok_label_html}</button>
  </div>
</section>
"""
    modal_id = await atlantis.client_modal(html, title=title or " ")
    atlantis.session_shared.set(f"{modal_radio_id}:modal_id", modal_id)
    exec_shell_js = json.dumps(atlantis.get_exec_shell_path())
    require_choice_before_submit_js = json.dumps(True)

    script = f"""
(function() {{
  var settled = false;
  var observer = null;
  function cleanup() {{ if (observer) {{ try {{ observer.disconnect(); }} catch (e) {{}} observer = null; }} }}
  function reveal(root) {{
    root.style.visibility = "visible";
    root.classList.add("modal-radio-ready");
  }}
  function settle(action, choiceId) {{
    if (settled) return;
    settled = true;
    cleanup();
    if (!window._accessToken) return;
    sendChatter(window._accessToken, action, {{
      modal_radio_id: {modal_radio_id_js},
      choice_id: choiceId || ""
    }}, {exec_shell_js}).catch(function() {{}});
  }}
  function selectedValue(root) {{
    var selected = root.querySelector('input[type="radio"]:checked');
    return selected ? selected.value : "";
  }}
  function syncOk(root, ok) {{
    if (!ok) return;
    ok.disabled = {require_choice_before_submit_js} && !selectedValue(root);
  }}
  function bind() {{
    var root = document.getElementById("modalradio-{uid}");
    if (!root) return;
    reveal(root);
    var ok = root.querySelector(".radio-ok");
    var cancel = root.querySelector(".radio-cancel");
    var selected = root.querySelector('input[type="radio"]:checked');
    if (selected) selected.focus({{ preventScroll: true }});
    syncOk(root, ok);
    Array.prototype.slice.call(root.querySelectorAll('input[type="radio"]')).forEach(function(input) {{
      input.addEventListener("change", function() {{ syncOk(root, ok); }});
    }});
    if (ok) ok.addEventListener("click", function() {{
      if (ok.disabled) return;
      settle("@modal_radio_select", selectedValue(root));
    }});
    if (cancel) cancel.addEventListener("click", function() {{ settle("@modal_radio_cancel", ""); }});
    root.addEventListener("keydown", function(event) {{
      if (event.key === "Escape") {{
        event.preventDefault();
        settle("@modal_radio_cancel", "");
      }}
    }});
    observer = new MutationObserver(function() {{
      if (!document.body.contains(root)) {{ settle("@modal_radio_cancel", ""); }}
    }});
    observer.observe(document.body, {{ childList: true, subtree: true }});
  }}
  requestAnimationFrame(function() {{ requestAnimationFrame(bind); }});
}})()
"""
    try:
        await atlantis.client_script(script)
        return await future
    finally:
        await _close_modal_if_open(modal_radio_id)
        atlantis.session_shared.remove(f"{modal_radio_id}:future")
        atlantis.session_shared.remove(f"{modal_radio_id}:choices")


@public
@visible
async def modal_menu(
    choices: List[Dict[str, Any]],
    title: str = "",
    heading: str = "",
) -> Optional[Dict[str, Any]]:
    """Pop up a modal menu and return the selected choice object.

    Returns None if the user closes/cancels the modal without selecting.
    """
    choice_by_id = _validated_modal_choices(choices)
    choice_buttons = []
    for choice in choices:
        choice_id = str(choice.get("id", "")).strip()
        choice_text = str(choice.get("text", "")).strip()
        disabled_attr = " disabled aria-disabled=\"true\"" if choice.get("disabled") else ""
        columns = choice.get("columns")
        if isinstance(columns, list) and columns:
            column_html = []
            for column in columns:
                if isinstance(column, dict) and column.get("type") == "image":
                    src = str(column.get("src") or "").strip()
                    alt = str(column.get("alt") or "")
                    if src:
                        column_html.append(
                            '<span class="menu-choice-cell menu-choice-image-cell">'
                            f'<img class="menu-choice-thumb" src="{html_lib.escape(src, quote=True)}" alt="{html_lib.escape(alt, quote=True)}">'
                            '</span>'
                        )
                    else:
                        column_html.append('<span class="menu-choice-cell menu-choice-image-cell"></span>')
                    continue
                column_html.append(
                    f'<span class="menu-choice-cell">{html_lib.escape(str(column or ""))}</span>'
                )
            button_content = (
                '<span class="menu-choice-grid">'
                + "".join(column_html)
                + "</span>"
            )
        else:
            button_content = html_lib.escape(choice_text)
        choice_buttons.append(
            '<button type="button" class="menu-choice" role="menuitem" '
            f'data-choice-id="{html_lib.escape(choice_id, quote=True)}"{disabled_attr}>'
            f"{button_content}</button>"
        )

    uid = uuid.uuid4().hex[:8]
    modal_menu_id = f"modal_menu:{uid}"
    modal_menu_id_js = json.dumps(modal_menu_id)
    heading_block = f"<h2>{html_lib.escape(heading)}</h2>" if heading else ""
    table_headers = None
    for choice in choices:
        headers = choice.get("column_headers")
        if isinstance(headers, list) and headers:
            table_headers = headers
            break
    header_block = ""
    if table_headers:
        header_block = (
            '<div class="menu-header" aria-hidden="true">'
            + "".join(
                f'<span class="menu-header-cell">{html_lib.escape(str(header or ""))}</span>'
                for header in table_headers
            )
            + "</div>"
        )
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    atlantis.session_shared.set(f"{modal_menu_id}:future", future)
    atlantis.session_shared.set(f"{modal_menu_id}:choices", choice_by_id)
    html = f"""
<style>
{_modal_panel_css(
    f"#modal-menu-panel-{uid}",
    f"#modalmenu-{uid}",
    ready_class="modal-menu-ready",
    padding=22,
    heading_margin="4px 0 16px",
    heading_font_size=24,
    heading_line_height=1.15,
)}
  #modalmenu-{uid} {{
    width: 100%;
    min-width: 0;
    visibility: hidden;
  }}
  /* Layout contract:
     - This CSS is only a fallback for Atlantis/jsPanel wrappers that start at
       browser width. It may constrain horizontal size and left edge.
     - Do not set top, height, overflow, or vertical transforms here. The script
       below measures the actual rendered panel and clamps vertical placement.
  */
  .jsPanel:has(#modalmenu-{uid}) {{
    width: auto !important;
    min-width: 0 !important;
    max-width: calc(100vw - 32px) !important;
    left: 50% !important;
    right: auto !important;
    bottom: auto !important;
    transform: translateX(-50%) !important;
  }}
  #modalmenu-{uid} .menu-list {{
    display: grid;
    gap: 8px;
    width: 100%;
  }}
  #modalmenu-{uid} .menu-header {{
    display: grid;
    grid-template-columns: minmax(118px, 1.1fr) minmax(74px, 0.75fr) minmax(74px, 0.75fr) minmax(58px, 0.55fr);
    gap: 10px;
    padding: 0 14px 2px;
    color: rgba(255, 250, 240, 0.68);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
  }}
  #modalmenu-{uid} .menu-header-cell {{
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  #modalmenu-{uid} .menu-choice {{
    box-sizing: border-box;
    width: 100%;
    min-height: 42px;
    padding: 0 14px;
    color: #fffaf0;
    background: rgba(7, 15, 22, 0.58);
    border: 1px solid rgba(20, 255, 208, 0.34);
    border-radius: 6px;
    font: inherit;
    font-size: 18px;
    font-weight: 400;
    text-align: left;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    cursor: pointer;
  }}
  #modalmenu-{uid} .menu-choice-grid {{
    display: grid;
    grid-template-columns: minmax(118px, 1.1fr) minmax(74px, 0.75fr) minmax(74px, 0.75fr) minmax(58px, 0.55fr);
    gap: 10px;
    align-items: center;
  }}
  #modalmenu-{uid} .menu-choice-cell {{
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  #modalmenu-{uid} .menu-choice-image-cell {{
    width: 44px;
    min-width: 44px;
    height: 32px;
    overflow: hidden;
  }}
  #modalmenu-{uid} .menu-choice-thumb {{
    display: block;
    width: 44px;
    height: 32px;
    object-fit: cover;
    border-radius: 4px;
  }}
  #modalmenu-{uid} .menu-choice:hover,
  #modalmenu-{uid} .menu-choice:focus {{
    background: rgba(20, 255, 208, 0.14);
    border-color: rgba(20, 255, 208, 0.62);
    outline: none;
  }}
  #modalmenu-{uid} .menu-choice:disabled {{
    color: rgba(255, 250, 240, 0.36);
    background: rgba(7, 15, 22, 0.28);
    border-color: rgba(255, 250, 240, 0.14);
    cursor: default;
    opacity: 0.55;
  }}
</style>
<section id="modalmenu-{uid}" aria-label="Menu">
  {heading_block}
  <div class="menu-list" role="menu">
    {header_block}
    {"".join(choice_buttons)}
  </div>
</section>
"""
    modal_id = await atlantis.client_modal(html, title=title or " ")
    atlantis.session_shared.set(f"{modal_menu_id}:modal_id", modal_id)
    exec_shell_js = json.dumps(atlantis.get_exec_shell_path())

    script = f"""
(function() {{
  var settled = false;
  var observer = null;
  function cleanup() {{ if (observer) {{ try {{ observer.disconnect(); }} catch (e) {{}} observer = null; }} }}
  function reveal(root) {{
    root.style.visibility = "visible";
    root.classList.add("modal-menu-ready");
  }}
  function markHost(host) {{
    if (!host) return;
    if (!host.id) host.id = "modal-menu-panel-{uid}";
    host.classList.add("modal-menu-panel");
    host.dataset.modalKind = "menu";
    host.dataset.modalMenuUid = "{uid}";
  }}
  function px(value) {{
    var number = parseFloat(value);
    return Number.isFinite(number) ? number : 0;
  }}
  function horizontalBox(style) {{
    return px(style.paddingLeft) + px(style.paddingRight) + px(style.borderLeftWidth) + px(style.borderRightWidth);
  }}
  function horizontalBorder(style) {{
    return px(style.borderLeftWidth) + px(style.borderRightWidth);
  }}
  function textWidth(element) {{
    var text = (element.textContent || "").trim();
    if (!text) return 0;
    var style = window.getComputedStyle(element);
    var canvas = textWidth.canvas || (textWidth.canvas = document.createElement("canvas"));
    var context = canvas.getContext("2d");
    context.font = style.font || [style.fontStyle, style.fontVariant, style.fontWeight, style.fontSize, style.fontFamily].join(" ");
    return Math.ceil(context.measureText(text).width);
  }}
  function collectGridWidths(root) {{
    var widths = [];
    Array.prototype.slice.call(root.querySelectorAll(".menu-header, .menu-choice-grid")).forEach(function(grid) {{
      Array.prototype.slice.call(grid.children || []).forEach(function(cell, index) {{
        widths[index] = Math.max(widths[index] || 0, Math.ceil(cell.scrollWidth), textWidth(cell) + 2);
      }});
    }});
    return widths;
  }}
  function gridGap(root) {{
    var grid = root.querySelector(".menu-header, .menu-choice-grid");
    if (!grid) return 0;
    var style = window.getComputedStyle(grid);
    return px(style.columnGap || style.gap);
  }}
  function gridWidth(widths, gap) {{
    if (!widths.length) return 0;
    return widths.reduce(function(total, width) {{
      return total + width;
    }}, 0) + gap * Math.max(0, widths.length - 1);
  }}
  function applyGridWidths(root, widths, enabled) {{
    var template = enabled && widths.length ? widths.map(function(width) {{ return width + "px"; }}).join(" ") : "";
    Array.prototype.slice.call(root.querySelectorAll(".menu-header, .menu-choice-grid")).forEach(function(grid) {{
      grid.style.gridTemplateColumns = template;
    }});
  }}
  function elementGridWidth(grid) {{
    var cells = Array.prototype.slice.call(grid.children || []);
    if (!cells.length) return 0;
    var style = window.getComputedStyle(grid);
    var gap = px(style.columnGap || style.gap);
    return cells.reduce(function(total, cell) {{
      return total + Math.ceil(cell.scrollWidth);
    }}, 0) + gap * Math.max(0, cells.length - 1);
  }}
  function menuMetrics(root) {{
    var rootStyle = window.getComputedStyle(root);
    var rootBox = horizontalBox(rootStyle);
    var columnWidths = collectGridWidths(root);
    var gap = gridGap(root);
    var fullGridWidth = gridWidth(columnWidths, gap);
    var needed = 0;
    var header = root.querySelector(".menu-header");
    if (header) {{
      needed = Math.max(needed, (fullGridWidth || elementGridWidth(header)) + horizontalBox(window.getComputedStyle(header)) + rootBox);
    }}
    Array.prototype.slice.call(root.querySelectorAll(".menu-choice")).forEach(function(button) {{
      var buttonStyle = window.getComputedStyle(button);
      var grid = button.querySelector(".menu-choice-grid");
      var buttonWidth = grid
        ? (fullGridWidth || elementGridWidth(grid)) + horizontalBox(buttonStyle)
        : textWidth(button) + horizontalBox(buttonStyle);
      needed = Math.max(needed, buttonWidth + rootBox);
    }});
    return {{ width: needed, columns: columnWidths }};
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
      reveal(root);
      return;
    }}
    markHost(host);
    host.style.minWidth = "0";
    var rect = host.getBoundingClientRect();
    var rootRect = root.getBoundingClientRect();
    if (!rect.width || !rect.height) {{
      if (shouldReveal) reveal(root);
      return;
    }}
    var metrics = menuMetrics(root);
    var hostExtraWidth = Math.max(0, Math.ceil(rect.width - rootRect.width));
    var targetWidth = Math.ceil(metrics.width + hostExtraWidth + 24);
    var viewportMax = Math.max(0, window.innerWidth - 32);
    var finalWidth = Math.min(targetWidth, viewportMax);
    applyGridWidths(root, metrics.columns, finalWidth - hostExtraWidth >= metrics.width);
    host.style.width = finalWidth + "px";
    host.style.maxWidth = "calc(100vw - 32px)";
    host.style.left = "50%";
    host.style.top = "50%";
    host.style.right = "auto";
    host.style.bottom = "auto";
    host.style.transform = "translate(-50%, -50%)";
    host.style.margin = "0";
    // Keep vertical geometry measured here. CSS cannot know the final panel
    // height after menu content, jsPanel chrome, and dynamic grid columns settle.
    // The 16px viewport margin mirrors the horizontal max-width margin above.
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
  async function cancel() {{
    if (settled) return;
    settled = true;
    cleanup();
    if (!window._accessToken) return;
    try {{
      await sendChatter(window._accessToken, "@modal_menu_cancel", {{
        modal_menu_id: {modal_menu_id_js}
      }}, {exec_shell_js});
    }} catch (e) {{}}
  }}
  function bind() {{
    var root = document.getElementById("modalmenu-{uid}");
    if (!root) return;
    var buttons = Array.prototype.slice.call(root.querySelectorAll(".menu-choice"));
    var enabledButtons = buttons.filter(function(button) {{ return !button.disabled; }});
    if (!buttons.length || !enabledButtons.length) return;
    scheduleCenter(root);
    enabledButtons[0].focus({{ preventScroll: true }});
    buttons.forEach(function(button) {{
      button.addEventListener("click", async function() {{
        if (settled) return;
        if (button.disabled) return;
        if (!window._accessToken) return;
        var choiceId = button.getAttribute("data-choice-id") || "";
        buttons.forEach(function(btn) {{ btn.disabled = true; }});
        settled = true;
        cleanup();
        await sendChatter(window._accessToken, "@modal_menu_select", {{
          modal_menu_id: {modal_menu_id_js},
          choice_id: choiceId
        }}, {exec_shell_js});
      }});
      button.addEventListener("keydown", function(event) {{
        if (button.disabled) return;
        var index = enabledButtons.indexOf(button);
        if (event.key === "ArrowDown") {{
          event.preventDefault();
          enabledButtons[(index + 1) % enabledButtons.length].focus({{ preventScroll: true }});
        }} else if (event.key === "ArrowUp") {{
          event.preventDefault();
          enabledButtons[(index + enabledButtons.length - 1) % enabledButtons.length].focus({{ preventScroll: true }});
        }} else if (event.key === "Enter" || event.key === " ") {{
          event.preventDefault();
          button.click();
        }}
      }});
    }});
    observer = new MutationObserver(function() {{
      if (!document.body.contains(root)) {{ cancel(); }}
    }});
    observer.observe(document.body, {{ childList: true, subtree: true }});
  }}
  requestAnimationFrame(function() {{ requestAnimationFrame(bind); }});
}})()
"""
    try:
        await atlantis.client_script(script)
        return await future
    finally:
        await _close_modal_if_open(modal_menu_id)
        atlantis.session_shared.remove(f"{modal_menu_id}:future")
        atlantis.session_shared.remove(f"{modal_menu_id}:choices")


@public
@visible
async def modal_string_click(modal_string_id: str, display_name: str) -> None:
    """Handle the display-name modal submit."""
    display_name = (display_name or "").strip()
    if not display_name:
        raise ValueError("display_name is required")
    modal_key = f"{modal_string_id}:modal_id"
    future_key = f"{modal_string_id}:future"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove(modal_key)
    future = atlantis.session_shared.get(future_key)
    if future is None:
        raise ValueError("Display-name modal is no longer active")
    if not future.done():
        future.set_result(display_name)


@public
@visible
async def modal_string_cancel(modal_string_id: str) -> None:
    """Handle the user closing the modal without submitting."""
    modal_key = f"{modal_string_id}:modal_id"
    future_key = f"{modal_string_id}:future"
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


async def _settle_modal_confirm(modal_confirm_id: str, result: bool) -> None:
    modal_key = f"{modal_confirm_id}:modal_id"
    future_key = f"{modal_confirm_id}:future"
    cancel_raises_key = f"{modal_confirm_id}:cancel_raises"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        try:
            await atlantis.client_modal_close(modal_id)
        except Exception:
            pass
        atlantis.session_shared.remove(modal_key)
    future = atlantis.session_shared.get(future_key)
    if future is not None and not future.done():
        if result is False and atlantis.session_shared.get(cancel_raises_key):
            future.set_exception(ModalGoBack("Modal flow cancelled by Go back"))
        else:
            future.set_result(result)


@public
@visible
async def modal_confirm_ok(modal_confirm_id: str) -> None:
    """Handle a continue click in a confirmation modal."""
    await _settle_modal_confirm(modal_confirm_id, True)


@public
@visible
async def modal_confirm_cancel(modal_confirm_id: str) -> None:
    """Handle cancel or close in a confirmation modal."""
    await _settle_modal_confirm(modal_confirm_id, False)


@public
@visible
async def modal_radio_select(modal_radio_id: str, choice_id: str) -> None:
    """Handle a radio modal continue click."""
    choice_id = (choice_id or "").strip()
    if not choice_id:
        raise ValueError("choice_id is required")
    modal_key = f"{modal_radio_id}:modal_id"
    future_key = f"{modal_radio_id}:future"
    choices_key = f"{modal_radio_id}:choices"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove(modal_key)
    choices = atlantis.session_shared.get(choices_key) or {}
    choice = choices.get(choice_id)
    if choice is None:
        raise ValueError(f"Unknown radio choice: {choice_id!r}")
    future = atlantis.session_shared.get(future_key)
    if future is None:
        raise ValueError("Radio modal is no longer active")
    if not future.done():
        future.set_result(choice)


@public
@visible
async def modal_radio_cancel(modal_radio_id: str, choice_id: str = "") -> None:
    """Handle cancel or close in a radio modal."""
    modal_key = f"{modal_radio_id}:modal_id"
    future_key = f"{modal_radio_id}:future"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        try:
            await atlantis.client_modal_close(modal_id)
        except Exception:
            pass
        atlantis.session_shared.remove(modal_key)
    future = atlantis.session_shared.get(future_key)
    if future is not None and not future.done():
        future.set_exception(ModalGoBack("Modal flow cancelled by Go back"))


@public
@visible
async def modal_menu_select(modal_menu_id: str, choice_id: str) -> None:
    """Handle a modal menu selection."""
    choice_id = (choice_id or "").strip()
    if not choice_id:
        raise ValueError("choice_id is required")
    modal_key = f"{modal_menu_id}:modal_id"
    future_key = f"{modal_menu_id}:future"
    choices_key = f"{modal_menu_id}:choices"
    modal_id = atlantis.session_shared.get(modal_key)
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove(modal_key)
    choices = atlantis.session_shared.get(choices_key) or {}
    choice = choices.get(choice_id)
    if choice is None:
        raise ValueError(f"Unknown menu choice: {choice_id!r}")
    future = atlantis.session_shared.get(future_key)
    if future is None:
        raise ValueError("Menu modal is no longer active")
    if not future.done():
        future.set_result(choice)


@public
@visible
async def modal_menu_cancel(modal_menu_id: str) -> None:
    """Handle the user closing a modal menu without selecting."""
    modal_key = f"{modal_menu_id}:modal_id"
    future_key = f"{modal_menu_id}:future"
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
