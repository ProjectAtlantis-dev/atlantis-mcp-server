# MCP Python Bridge 🐱

A simple bridge client for connecting MCP hosts to our Python dynamic functions server.

## How it Works

This package acts as a bridge between an MCP host (like Claude) and our Python WebSocket server. It:

1. Connects to our already-running Python server
2. Forwards messages between the MCP host and the Python server
3. Handles proper shutdown when terminated
4. Automatically checks for server PID only when connecting to localhost

## Installation

```bash
# Install globally
npm install -g mcp-python-bridge

# Or use directly with npx/uvx
npx mcp-python-bridge
```

## Usage

First, ensure your Python MCP server is running:

```bash
python server.py  # This starts the Python server on 127.0.0.1:8000
```

Then, in a different terminal or from an MCP host, run the bridge:

```bash
mcp-python-bridge --port 8000
```

## MCP Configuration

To use this with an MCP host like Claude, use the following configuration:

```json
{
  "mcpServers": {
    "python-dynamic-functions": {
      "command": "npx",
      "args": ["mcp-python-bridge", "--port", "8000"]
    }
  }
}
```

Or, if using locally without publishing:

```json
{
  "mcpServers": {
    "python-dynamic-functions": {
      "command": "node",
      "args": ["/path/to/node-mcp-client/index.js", "--port", "8000"]
    }
  }
}
```

## Options

- `--host`, `-h`: Host of the Python MCP server (default: '127.0.0.1')
- `--port`, `-p`: Port of the Python MCP server (default: 8000)
- `--path`: WebSocket path (default: '/mcp')

## Behavior Notes

- When connecting to localhost (127.0.0.1), the client will automatically check for a server PID file to verify the server is running
- When connecting to any other host, PID checking is automatically skipped
- All configuration is done via command line arguments only
