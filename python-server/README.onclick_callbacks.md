# Onclick Callbacks: Browser → Dynamic Function

This document explains how to wire a UI element in a dynamic function so that a user action in the browser (button click, drop, etc.) calls back into a Python function on the MCP server.

The mechanism is `sendChatter`. It's how browser-side JavaScript (rendered by a dynamic function) invokes another Python function in the same dynamic-functions tree, with arguments.

A complete working example lives in `dynamic_functions/InWork/create_video_with_image.py` — this README walks through that example.

---

## The big picture

```
┌──────────────┐   render HTML+JS    ┌────────────────────┐
│ Dynamic fn A │ ──────────────────► │ Browser            │
│ (renders UI) │                     │ (button, dropzone) │
└──────────────┘                     └─────────┬──────────┘
                                               │ user clicks
                                               ▼
                                     sendChatter(token, target, data)
                                               │
                                               ▼
┌──────────────┐    nodejs routes    ┌────────────────────┐
│ Dynamic fn B │ ◄────────────────── │ atlantis_nodejs    │
│ (callback)   │                     │ app_server         │
└──────────────┘                     └────────────────────┘
```

A and B can be the same module. In the video-upload example, the rendering function and the callback function live in one file.

---

## The four moving pieces

### 1. The HTML element with a stable id

```python
button_html = '''
<button id="sendButton_{UPLOAD_ID}" disabled>Generate Video</button>
'''
```

`{UPLOAD_ID}` is a per-render unique token so multiple instances of the UI don't collide.

### 2. The browser-side click handler

In the `miniscript` (the JS embedded in the dynamic function), bind an event listener that gathers the data and calls `sendChatter`:

```javascript
sendButton.addEventListener('click', async function() {
    const file = window.atlantis_file_selected_{UPLOAD_ID}.file;
    const editedPrompt = document.getElementById('promptInput_{UPLOAD_ID}').value;

    const reader = new FileReader();
    reader.onload = async function(e) {
        const base64Content = e.target.result;

        const data = {
            // these attrs must match the function param names
            base64Content: base64Content,
            filename: file.name,
            filetype: file.type,
            prompt: editedPrompt
        };

        await sendChatter(
            window._accessToken,
            '$**InWork**process_video_upload',
            data
        );
    };
    reader.readAsDataURL(file);
});
```

### 3. `sendChatter(accessToken, target, data)`

A globally-available browser function provided by the atlantis runtime. Its three arguments:

| Arg | Meaning |
|-----|---------|
| `accessToken` | `window._accessToken` — a per-session UUID minted by the nodejs server when the websocket connects, stashed on `window` for the browser to send back as auth. |
| `target` | A routing string of the form `'$**<Subdir>**<function_name>'`. The `$` prefix starts lookup at the root of the current MCP server, which is usually what browser callbacks want. The `**...**` segments name the subdirectory under `dynamic_functions/`; the trailing segment is the function. Example: `'$**InWork**process_video_upload'` → `dynamic_functions/InWork/process_video_upload`. |
| `data` | A plain JS object. Each key becomes a kwarg passed to the Python function. **Key names must match the Python parameter names exactly.** |

### 4. The Python callback function

```python
async def process_video_upload(
    base64Content: str,
    filename: str,
    filetype: str,
    prompt: str
):
    username = atlantis.get_caller() or "unknown_user"
    await atlantis.client_log("📥 Processing uploaded image...")
    # ... do the work, log progress back to the user via atlantis.client_log
```

The dispatcher unpacks `data` as keyword arguments, so the parameter list mirrors the JS payload one-for-one. The function can be `async` and use the full `atlantis.*` runtime API to log progress, return media, etc.

---

## Naming contract — easy to break, easy to debug

The single most common mistake is a name mismatch between the JS payload keys and the Python parameter names. The example file calls this out with an inline comment:

```javascript
const data = {
    // these attrs must match the function param names
    base64Content: base64Content,
    ...
};
```

If you rename a parameter on the Python side, update every `data` key on every JS caller. There is no implicit coercion (e.g. camelCase ↔ snake_case).

---

## Access token: where it comes from

You don't manage `window._accessToken` from your dynamic function — the nodejs server handles the whole lifecycle:

| Step | Where |
|------|-------|
| Mint UUID per websocket | `atlantis_nodejs/src/app_server.ts` — `websocket._accessToken = uuid.v4()` |
| Push to the browser | `app_server.ts` — sent as `accessToken` in an outgoing message |
| Browser stores it | `atlantis_nodejs/src/browser/App.ts` — `window._accessToken = params.accessToken` |
| Server validates incoming chatter | `app_server.ts` — `if (params.accessToken !== websocket._accessToken) { reject }` |

So as a dynamic-function author you just read `window._accessToken` in your JS handler and pass it to `sendChatter`. Defensive code is wise:

```javascript
if (!window._accessToken) {
    thread.console.softError('[send] window._accessToken is empty or undefined!');
}
```

---

## UX patterns from the example

- **Disable the trigger after firing.** Once `sendChatter` is sent, set `sendButton.disabled = true` and update its label (`'Processing...'`). This prevents double-submits on long-running jobs.
- **Yield to the event loop after `client_log`.** In the Python callback, `await asyncio.sleep(0)` after a `client_log` call helps ensure the "Starting..." message reaches the user before the next blocking step begins.
- **Log to two channels.** Use `atlantis.client_log(...)` for messages the user should see, and `atlantis.owner_log(...)` for diagnostic detail tied to a `job_id`.

---

## Modal callbacks

The same `sendChatter` pattern works inside modal HTML rendered with `atlantis.client_modal(...)` or `atlantis.client_html(..., modal=True)`.

`client_modal()` returns the client `modalId`. Store it somewhere request/session scoped if the callback should close the modal later:

```python
modal_id = await atlantis.client_modal(html, title="Welcome")
atlantis.session_shared.set("welcome_modal_id", modal_id)
await atlantis.client_script(script)
```

Then the callback can close the modal after it handles the event:

```python
async def handle_welcome_click(character_name: str):
    modal_id = atlantis.session_shared.get("welcome_modal_id")
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove("welcome_modal_id")
    await atlantis.client_log(f"entered as {character_name}")
```

Two details matter:

- Render the modal first, then send a script that binds event listeners to elements inside it.
- Use a per-render id suffix in modal DOM ids, because a user can open more than one modal or re-render the same tool.

See `dynamic_functions/Home/game.py` for the current welcome-modal pattern.

---

## Minimum viable example

```python
import atlantis

@visible
async def render_button():
    UPLOAD_ID = "abc123"  # in practice, generate per-render
    html = f'<button id="btn_{UPLOAD_ID}">Click me</button>'
    await atlantis.client_html(html)

    script = f'''
    document.getElementById('btn_{UPLOAD_ID}').addEventListener('click', async () => {{
        await sendChatter(window._accessToken, '$**MySubdir**handle_click', {{
            message: 'hello from the browser'
        }});
    }});
    '''
    await atlantis.client_script(script)


@visible
async def handle_click(message: str):
    await atlantis.client_log(f"got: {message}")
```

Drop both functions in `dynamic_functions/MySubdir/` and the routing string `'$**MySubdir**handle_click'` will resolve to `handle_click` from the current MCP server root.

---

## Reference: the canonical example

| File | Role |
|------|------|
| `python-server/dynamic_functions/InWork/create_video_with_image.py` | renders the upload UI and defines `process_video_upload` as the callback target |
| `python-server/dynamic_functions/InWork/wan_i2v_workflow.py` | shared workflow data used by the callback |

Look at `create_video_with_image.py` line 537 (the click handler) and line 45 (the Python callback) side-by-side to see the full round-trip.
