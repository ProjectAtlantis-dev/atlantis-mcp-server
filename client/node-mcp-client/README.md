# Lobster MCP client 🦞

See the [project README](../../README.md) for Python server setup. Start that server before connecting Lobster.

Use the port configured by `--port` in the Python server's launch command (for example, your `runServer` script). Replace `YOUR_SERVER_PORT` in this template with that number. `8000` is only the default when no port override is configured.

```json
{
    "mcpServers": {
        "atlantis_lobster": {
            "command": "npx",
            "args": [
               "atlantis-mcp",
               "--port",
               "YOUR_SERVER_PORT"
               ]
         }
    }
  }
```

To add Atlantis to Claude Code, replace the same placeholder in:

```bash
claude mcp add atlantis_lobster -- npx atlantis-mcp --port YOUR_SERVER_PORT
```

If you change the server port, update the MCP entry's `--port` argument too. The client connects to `127.0.0.1` by default; add `--host` with the server's hostname if it runs elsewhere.
