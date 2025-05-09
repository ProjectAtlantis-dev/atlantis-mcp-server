![happy](/happy.png)

# Project Atlantis

The somewhat hand-wavey goal of Project Atlantis is simulating a future buildout of rugged, remote Greenland using emerging autonomous technologies. However, new technologies require testing so Project Atlantis is a giant sandbox of sorts

## MCP Remote Server

This project folder provides a flexible and generic Model Context Protocol (MCP) server written in Python (which we call a 'remote') to play with and collaborate

### Key Components

1. **Python MCP Server** (`python-server/`)
   - Our 'remote'. Runs locally but can be controlled remotely via the Atlantis cloud, which may be handy if trying to control servers across multiple machines

2. **MCP Client** (`client/`)
   - Useful for AI that wants to control the remote as another ordinary MCP tool
   - Written using npx
   - No cloud needed although it might produce annoying errors
   - Can only see tools on the local MCP server (at least right now), although tools can call back to the cloud

## Quick Start

1. Edit the runServer script in the `python-server` folder

`python server.py --email=your@gmail.com --api-key=foobar --host=localhost --port=8000 --cloud-host=https://www.projectatlantis.ai --cloud-port=3010 --service-name=home`

2. Sign up at https://www.projectatlantis.ai. Your default API key will let the remot connect

## Python MCP Server Features

#### Dynamic Functions

- Gives users the ability to create and maintain custom functions-as-tools in the `dynamic_functions/` folder
- Functions are loaded dynamically and automatically reloaded when modified
- You can either edit functions locally and the server will automatically detect changes, or edit remotely in the Atlantis cloud
- Dynamic functions can import each other and the server should correctly handle hot-loaded dependency changes, within the constraints of the Python VM
- Every dynamic function has access to a generic `atlantis` utility module:
  ```python
  import atlantis
  atlantis.client_log("This message will appear in the Atlantis cloud console!")
  ```
-  The MCP spec is in flux and so the protocol between our MCP server and the cloud is a superset (we rely heavily on annotations)
- A lot of this stuff including below may end up getting lumped under MCP "Resources"

#### Dynamic MCP Servers

- Gives users the ability to install and manage third-party MCP server tools in the `dynamic_servers/` folder
- This lets you host the growing ecosystem other MCP tools as if they were part of your own server
- Using the npx client, you host other MCP servers directly or use the Atlantis cloud
- Each server config follows the usual JSON structure that contains an 'mcpServers' element
- For example, this installs our openweather MCP server:

   ```json
   {
      "mcpServers": {
         "openweather": {
            "command": "uvx",
            "args": [
            "--from",
            "atlantis-open-weather-mcp",
            "start-weather-server",
            "--api-key",
            "<your openweather api key>"
            ]
         }
      }
   }
   ```

See [here](https://github.com/ProjectAtlantis-dev/atlantis-open-weather-mcp)
