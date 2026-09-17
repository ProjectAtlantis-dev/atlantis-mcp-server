# Home

A small app that comes with the platform. It sets up the default homepage and terminal, gives Lobster and Multix their readme entry points, and holds the file callback.

- **`main.py`**: the `README` and `README_LOBSTER` tools. Both return [MULTIX.md](MULTIX.md), the Lobster MCP tool and Multix shell guide.
- **`homepage.py`**: the `@homepage` startup script. It sets the working folder and path, turns on the file callback, and starts the terminal.
- **`file.py`**: the file callback (see below).

## Default Path

The default shell path should include the Home folder, so its tools resolve from anywhere with `@name` (for example `@README`). `homepage.py` handles this at startup with `/path push <Home folder>` followed by `/env save`. If you replace the homepage or reset the path, push the Home folder back on.

## File Callback

`file.py` is the `@file` callback the cloud uses to read, write and list text files in this folder. Right now only **Excalidraw** uses it, for `.excalidraw` scenes. See `homepage.py` for how it is turned on at startup.

It only accepts bare filenames, and it raises instead of guessing. The docstrings cover the operations, and `python-server/test_file_callback.py` covers the behavior.
