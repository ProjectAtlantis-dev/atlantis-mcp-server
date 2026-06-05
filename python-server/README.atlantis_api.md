# `atlantis` — Dynamic Function API

`atlantis.py` is the bridge a **dynamic function** uses to talk to the calling
client and to read the context of the current call. Inside a dynamic function
you simply:

```python
import atlantis

async def my_function():
    caller = atlantis.get_caller()
    await atlantis.client_log(f"Hi {caller}!")
```

Everything below is part of the public surface. Functions and names prefixed
with `_` (e.g. `_client_command`, `_get_client_id`) are internal plumbing and
are **not** meant to be called from dynamic functions.

Most messaging calls are `async` — `await` them.

---

## Call context — who/where am I?

These read the `CallContext` that the server establishes for the active call.
They take no arguments and return `None` when there is no active context.

| Function | Returns | Notes |
| --- | --- | --- |
| `get_context()` | `CallContext` | The whole context object, if you need raw fields. |
| `get_caller()` | `str` | The **sid** (username) of whoever invoked this function. |
| `get_request_id()` | `str` | Correlation id for this request — used to route messages back. |
| `get_entry_point_name()` | `str` | Name of the originally invoked (entry point) function. |
| `get_session_key()` | `str` | Stable per-session identifier. `None` if any component is missing. |
| `get_terminal_key()` | `str` | Session narrowed to one terminal: `session_key` + the user's root shell. |
| `get_caller_shell_path()` | `str` | The user's **root** shell path (used for attribution). |
| `get_exec_shell_path()` | `str` | The shell where this call's work actually runs. Outbound callbacks are tagged with this; falls back to the caller shell path. |
| `get_user_game_id()` | `int` | The integer `user_game_id` for this call. |

### Owner / permission checks

| Function | Returns | Notes |
| --- | --- | --- |
| `get_default_owner()` | `str` | Default owner username of this server instance. |
| `get_owner_usernames()` | `List[str]` | All owner usernames. |
| `is_owner(username)` | `bool` | Whether `username` is an owner. |

### Context setters

| Function | Notes |
| --- | --- |
| `set_exec_shell_path(path)` | Override the exec shell path for the current context (e.g. lobster socket tasks where the lobster's single shell *is* the exec shell). |
| `set_context(ctx)` / `reset_context()` | Set/clear the active `CallContext`. Primarily for the function manager — rarely needed in user code. |

---

## Logging & messages

| Function | Notes |
| --- | --- |
| `client_log(message, level="INFO", message_type="text", is_private=True, location=None)` | The core "send something back to the client" call. Auto-captures sequence number, caller name, and entry point. `message_type` can be `"text"`, `"json"`, or an image mime (`"image/png"`, …). `is_private=False` adds a cloud-side routing hint for broadcast. |
| `client_description(message, ...)` | `client_log` with `message_type="description"`. |
| `client_warning(message, ...)` | `client_log` with `message_type="warning"`. |
| `owner_log(message)` | Appends a structured entry to `log/owner_log.json` (timestamp, tool name, caller) and echoes to the server console. |
| `gather_logs()` | `await` to block until all pending `client_log` tasks have actually been delivered. Returns `True` if there were tasks to wait on. |

---

## Rendering rich content

All of these send content to the client and `await` its acknowledgment.

| Function | Sends |
| --- | --- |
| `client_markdown(content)` | Markdown to render. |
| `client_html(content, modal=False, title=None)` | HTML. With `modal=True` it renders in a modal and the ack must carry a `modalId`. |
| `client_modal(content, title=None)` | HTML in a modal; **returns the modal UUID**. |
| `client_modal_close(modal_id)` | Closes a previously opened modal by id. |
| `client_data(description, data, column_formatter=None)` | A JSON-serializable object for styled rendering. Arrays of objects auto-display as a table. `column_formatter` maps column names to display options. Raises `TypeError` if `data` isn't JSON-serializable. |
| `client_image(image_path, image_format=None)` | An image file (base64-encoded). Mime auto-detected from extension if omitted. |
| `client_video(video_path, video_format=None)` | A video file (base64-encoded). Mime auto-detected if omitted. |
| `client_script(content, is_private=True)` | JavaScript that runs **once** (deduped by event completion). |
| `client_terminal_script(content, is_private=True)` | JavaScript that **re-runs on every render** — use for cosmetic DOM effects that must survive a page reload. |

### Backgrounds

| Function | Notes |
| --- | --- |
| `set_background(image_path, image_format=None, vertical_align="bottom")` | Set a background image. |
| `set_background_video(video_source, ...)` | Set a background video. `video_source` may be a URL, data URL, or local path. Knobs: `vertical_align`, `playback_rate`, `brightness`, `loop`, `muted`, `autoplay`, `plays_inline`, `remove_on_ended`, `toggle_audio`, `replay`. |
| `set_background_player(video_source, ...)` | Like `set_background_video` but a controllable player. Adds a `controls` flag; no `toggle_audio`. |

---

## Streaming

For incremental output (e.g. token-by-token). Open a stream, push snippets,
then close it.

| Function | Notes |
| --- | --- |
| `stream_start(sid, who)` | Opens a stream; **returns a `stream_id`** to use for the following calls. |
| `stream(message, stream_id_param)` | Sends one snippet on the given stream. |
| `stream_end(stream_id_param)` | Closes the stream. |

```python
sid = atlantis.get_caller()
stream_id = await atlantis.stream_start(sid, who="my_function")
for chunk in chunks:
    await atlantis.stream(chunk, stream_id)
await atlantis.stream_end(stream_id)
```

Ack behavior is governed by the module-level flags `AWAIT_STREAM_START_ACK`,
`AWAIT_STREAM_MSG_ACK`, `AWAIT_STREAM_END_ACK` (all `True` by default — wait for
each ack).

---

## Commands & interaction

| Function | Notes |
| --- | --- |
| `client_command(command, data=None, message_type="command", is_private=True)` | Sends a command to the client and **waits for its result**. The general-purpose request/response primitive that most of the rendering helpers above are built on. |
| `tool_result(name, result)` | Pushes a tool-call result into the transcript so the LLM sees it on the next turn. |
| `client_onclick(key, callback)` | Registers an async `callback` to fire when the client reports a click for `key`. |
| `client_upload(key, callback)` | Registers an async `callback` to fire when an upload occurs for `key`. |

---

## Browser → Dynamic Function callbacks

How to wire a UI element rendered by a dynamic function so that a user action in
the browser (button click, drop, upload) calls back into a Python function on
the MCP server.

The mechanism is `sendChatter` — a browser-side function (provided by the
atlantis runtime) that invokes a Python function anywhere in the current
dynamic-functions tree, with arguments.

A complete working example lives in
`dynamic_functions/InWork/create_video_with_image.py`.

### The big picture

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

A and B can be the same module.

### The four moving pieces

**1. An HTML element with a stable id.** Use a per-render unique token so
multiple instances of the UI don't collide:

```python
button_html = '<button id="sendButton_{UPLOAD_ID}" disabled>Generate Video</button>'
```

**2. A browser-side handler** that gathers data and calls `sendChatter`:

```javascript
sendButton.addEventListener('click', async function() {
    const data = {
        // these keys must match the Python function param names
        base64Content: base64Content,
        filename: file.name,
        prompt: editedPrompt
    };
    await sendChatter(window._accessToken, '$**InWork**process_video_upload', data);
});
```

**3. `sendChatter(accessToken, target, data)`** — globally available in the browser:

| Arg | Meaning |
|-----|---------|
| `accessToken` | `window._accessToken` — a per-session UUID minted by the nodejs server on websocket connect, sent back as auth. |
| `target` | Routing string `'$**<Subdir>**<function_name>'`. The `$` prefix starts lookup at the current MCP server root; `**...**` segments name the subdirectory under `dynamic_functions/`; the trailing segment is the function. E.g. `'$**InWork**process_video_upload'` → `dynamic_functions/InWork/process_video_upload`. |
| `data` | A plain JS object. Each key becomes a kwarg to the Python function. **Key names must match the Python parameter names exactly** — there is no camelCase ↔ snake_case coercion. |

**4. The Python callback** — params mirror the JS payload one-for-one:

```python
async def process_video_upload(base64Content: str, filename: str, prompt: str):
    username = atlantis.get_caller() or "unknown_user"
    await atlantis.client_log("📥 Processing uploaded image...")
    # ... do the work, log progress back via atlantis.client_log
```

### Access token

You don't manage `window._accessToken` — the nodejs server owns its lifecycle
(mints a UUID per websocket in `app_server.ts`, pushes it to the browser where
`App.ts` stores it on `window`, and validates it on every incoming chatter).
As a dynamic-function author you just read it in your JS handler and pass it to
`sendChatter`. Guarding against an empty token is wise:

```javascript
if (!window._accessToken) {
    thread.console.softError('[send] window._accessToken is empty or undefined!');
}
```

### Modal callbacks

The same `sendChatter` pattern works inside modal HTML rendered with
`client_modal(...)` or `client_html(..., modal=True)`. `client_modal()` returns
the `modalId`; stash it (request/session scoped) if the callback should close
the modal later:

```python
modal_id = await atlantis.client_modal(html, title="Welcome")
atlantis.session_shared.set("welcome_modal_id", modal_id)
await atlantis.client_script(script)   # binds listeners to elements in the modal

async def handle_welcome_click(character_name: str):
    modal_id = atlantis.session_shared.get("welcome_modal_id")
    if modal_id:
        await atlantis.client_modal_close(modal_id)
        atlantis.session_shared.remove("welcome_modal_id")
    await atlantis.client_log(f"entered as {character_name}")
```

Render the modal *first*, then send the script that binds listeners. Use a
per-render id suffix on modal DOM ids — a user can open more than one modal or
re-render the same tool. See `dynamic_functions/Home/game.py`.

### `client_onclick` / `client_upload`

`client_onclick(key, callback)` and `client_upload(key, callback)` are the
**key-registration** variant: instead of routing by `$**Subdir**fn`, you
register an async `callback` against a `key` and the client fires it back by
that key. Use these when the handler is set up imperatively rather than via a
`sendChatter` routing string.

### UX patterns

- **Disable the trigger after firing** (`sendButton.disabled = true`, label →
  `'Processing...'`) to prevent double-submits on long jobs.
- **Yield after `client_log`.** `await asyncio.sleep(0)` after a `client_log`
  helps the message reach the user before the next blocking step.
- **Log to two channels.** `atlantis.client_log(...)` for the user;
  `atlantis.owner_log(...)` for diagnostics tied to a `job_id`.

### Minimum viable example

```python
import atlantis

@visible
async def render_button():
    UPLOAD_ID = "abc123"  # in practice, generate per-render
    await atlantis.client_html(f'<button id="btn_{UPLOAD_ID}">Click me</button>')
    await atlantis.client_script(f'''
    document.getElementById('btn_{UPLOAD_ID}').addEventListener('click', async () => {{
        await sendChatter(window._accessToken, '$**MySubdir**handle_click', {{
            message: 'hello from the browser'
        }});
    }});
    ''')

@visible
async def handle_click(message: str):
    await atlantis.client_log(f"got: {message}")
```

Drop both in `dynamic_functions/MySubdir/` and `'$**MySubdir**handle_click'`
resolves to `handle_click` from the current MCP server root.

---

## Shared state across reloads

Two containers survive dynamic-function reloads. Both expose
`get(key, default=None)`, `set(key, value)`, `remove(key)`, and `keys()`.

| Object | Scope |
| --- | --- |
| `server_shared` | Global, server-wide (DB connections, busy tracking, etc.). |
| `session_shared` | Auto-namespaced by session id — one session cannot see another's data. Requires an active session context. |

```python
conn = atlantis.server_shared.get("db")
atlantis.session_shared.set("last_query", q)
```

---

## Utilities

| Function | Notes |
| --- | --- |
| `image_to_base64(image_path)` | Read a file and return a base64 string. Raises `FileNotFoundError` / `IOError`. |
| `video_to_base64(video_path)` | Same for video. |
</content>
</invoke>
