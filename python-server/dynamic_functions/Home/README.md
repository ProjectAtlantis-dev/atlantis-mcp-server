# Home

A small app that comes with the platform. It sets up the default homepage and terminal, gives Lobster and Multix their readme entry points, and holds the file callback.

- **`main.py`**: the `README` and `README_LOBSTER` tools. Both return [MULTIX.md](MULTIX.md), the Lobster MCP tool and Multix shell guide.
- **`homepage.py`**: the `@homepage` startup script. It sets the working folder and path, turns on the file callback, and starts the terminal.
- **`file.py`**: the file callback (see below).

## Default Path

The default shell path should include the Home folder, so its tools resolve from anywhere with `@name` (for example `@README`). `homepage.py` handles this at startup with `/path push <Home folder>` followed by `/env save`. If you replace the homepage or reset the path, push the Home folder back on.

## File Callback

`file.py` defines `file_callback`, which is marked with `@file`. The cloud calls it to read, write and list text files. Right now only **Excalidraw** uses it, to load and save `.excalidraw` scenes (see `foo.json` for a sample scene).

To turn it on:

```
/callback set file Home/file_callback
```

or `/callback set file auto`, which is what `homepage.py` runs at startup.

Operations:

| Operation | Arguments | Result |
| --- | --- | --- |
| `get` | `filename` | The UTF-8 contents. Raises `FileNotFoundError` if the file doesn't exist. |
| `set` | `filename`, `content` | Creates or overwrites the file and returns `"Saved <filename>"`. `content` is required. |
| `list` | `suffix`, passed either as the second argument or by name (not both) | `{"name", "suffix"}` rows for files with that extension, e.g. `excalidraw`, written without the dot |

Rules:

- Files live in the `Home/` folder itself. Only bare filenames are accepted. Paths, `.`/`..`, drive letters and symlinks raise `ValueError`.
- For `get`/`set`, if you pass a `suffix`, a `filename` without an extension gets `.<suffix>` added (`foo` becomes `foo.json`). A filename that already has an extension must match the suffix.
- Unknown operations raise `ValueError`.

Tests are in `python-server/test_file_callback.py`.
