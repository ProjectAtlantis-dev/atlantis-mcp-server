# MCP Python Bridge 🐱

See the other README instead

Your MCP config should look something like this, assuming that the MCP server is already running on port 8000
{
    "mcpServers": {
        "atlantis": {
            "command": "npx",
            "args": [
               "atlantis-mcp",
               "--port",
               "8000"
               ]
         }
    }
  }


To add Atlantis to Claude Code, use the command:

claude mcp add atlantis -- npx atlantis-mcp --port 8000

