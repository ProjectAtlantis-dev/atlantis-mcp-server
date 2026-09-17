# `atlantis` — Dynamic Function API

`atlantis.py` is the bridge a dynamic function uses to talk to the calling
client and read the context of the current call. Signatures and parameters are
in `atlantis.py` itself; this page covers how the pieces fit together.

```python
import atlantis

async def my_function():
    caller = atlantis.get_caller()
    await atlantis.client_log(f"Hi {caller}!")
```

Names starting with `_` are internal plumbing, so don't call them from dynamic
functions. Most messaging calls are `async`.

Prefer running cloud commands directly (`client_command`) over adding wrappers
to `atlantis.py` that do the same thing.

## Areas

- **Call context:** the `get_*` readers (caller, request, session, terminal,
  shell paths, game id) return `None` outside an active call.
  `get_owner_usernames`, `get_default_owner` and `is_owner` answer permission
  questions.
- **Output:** `client_log` and its variants, `client_markdown`, `client_html`,
  `client_modal`, `client_data`, `client_image`, `client_video`, `client_widget`,
  the background setters, and `stream_start` / `stream` / `stream_end` for
  incremental output.
- **Scripts:** `client_script` runs once. `client_terminal_script` re-runs on
  every render, so use it for DOM effects that must survive a reload.
- **Commands:** `client_command` runs a cloud command and waits for its result.
  `tool_result` pushes a result into the transcript for the LLM's next turn.
- **Diagnostics:** `owner_log` writes to the owner log instead of the user.

## Shell routing

Helpers that take `shell` accept:

- a tab name such as `"d_map"` or `"t_main"`
- a shell type (`"display"`, `"terminal"`, `"user"`), which picks the first open tab of that type
- `"exec"`, the isolated tool-execution shell
- `"caller"`, the originating terminal

`"display"` output is live-only, while `"user"` output replays after a
reconnect. If the target has no open tab, output falls back to the caller's
terminal. No shell is ever created.

## Shared state

`server_shared` and `session_shared` survive dynamic-function reloads.
`server_shared` is server-wide (DB connections and so on). `session_shared` is
namespaced per session and raises if there is no session context.

## Browser callbacks

Rendered HTML calls back into Python with `sendChatter`, a browser global:

```javascript
await sendChatter(window._accessToken, '$**MyApp**handle_click', { message: 'hello' }, execShellPath);
```

- **Token:** `window._accessToken` is owned by the Node server. Just pass it through.
- **Target:** `$` starts at the current remote's root, `**MyApp**` names the
  app folder, and the last segment is the function.
- **Payload:** keys become keyword arguments. They must match the Python
  parameter names exactly, with no camelCase/snake_case conversion.
- **Exec shell (4th argument):** pass `atlantis.get_exec_shell_path()` from the
  rendering call. Without it, the click lands in the user's main shell history
  instead of nesting under the tool call. See the comment in
  `dynamic_functions/Home/modal.py`.

For modals, render the modal first, then send the script that binds the
listeners. Suffix DOM ids per render, since the same UI can be open more than
once. Keep the id returned by `client_modal` in `session_shared` if the
callback needs to close it. `Home/modal.py` is the working example.

`client_onclick(key, callback)` and `client_upload(key, callback)` are the
alternative: they register a Python callable under a key, which the built-in
`_public_click` / `_public_upload` tools fire. Use them when there is no named
dynamic function to route to.
