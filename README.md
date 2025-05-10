![happy](/happy.png)

# Project Atlantis

The somewhat hand-wavey goal of Project Atlantis is simulating a future buildout of rugged, remote Greenland using emerging autonomous technologies. However, new technologies require testing so Project Atlantis is a giant bot sandbox of sorts

## MCP Python Remote Server

I wrote this trying to get past the hype and learn what an MCP (Model Context Protocol) server was, as well as explore potential future MCP directions (keep in mind that Silicon Valley has its own plans)

I think most confusion with MCP is because the BYOC architecture is inverted to traditional cloud, but this gives the user much more control over compute and AI privacy

The main piece of this project is just a hotloadable Python server (which I call a 'remote') for curious people to play with and collaborate. I was building a Node counterpart but shelved it for now because the Node VM hotloader is not nearly as easy to work with

### Architecture

Caveat: MCP terminology is already terrible and I just made it worse. I'm not an MCP expert and so naively I called everything a 'server' and it quickly became a mess. I've tried to rename things retroactively but you may still see some inconsistencies

Pieces of the system:

- **Cloud**: the Atlantis cloud server (free since most compute runs on your local box anyway); mostly a chat UX and backend database
- **Remote**: a Python p2p MCP server running on each box (you can have >1 just specify different service name)
- **Dynamic Function**: a custom Python function that can act as a tool, can be reloaded on the fly, see below
- **Dynamic MCP Server**: any 3rd party MCP, what gets stored is just a JSON config file, see below
#### Terminology Notes
The MCP spec seems to suggest that MCP stuff connects from an MCP 'host' (think Claude Desktop or Cursor or Windsurf) but since I usually access these hosts remotely from the cloud, I ended up calling them "remotes" instead. While a remote can host 3rd party MCP tools (which have tools ofc), it can also host Python functions

![design](/design.png)

Why the cloud? MCP auth and security are still being worked out it's easier to have a trusted host for now. Our intention for Greenland is for each town, settlement work site etc. to have at least one remote

### Key Components

1. **Python Remote (MCP P2P server)** (`python-server/`)
   - Our 'remote'. Runs locally but can be controlled remotely via the Atlantis cloud, which may be handy if trying to control servers across multiple machines

2. **MCP Client** (`client/`)
   - Useful if you want to treat your local remote as an ordinary MCP tool
   - Written using npx
   - No cloud needed although it might produce annoying errors
   - Capabilities limited to tools/list
   - Can only see tools on the local box (at least right now), although tools can call back to the cloud

## Quick Start

1. Prerequisites - need to install Python for the server and Node for the MCP client; you should also install uvx and npx

2. All this may need to be run in a fairly modern 3.13 Python venv to ensure everything works or it will fallback to whatever basic Python you have which could be quite old (claude windsurf have the same issue btw so if it seems like no MCP stuff is working, it's almost certainly the Python environment)

3. Edit the runServer script in the `python-server` folder and set the email and service name:

```bash
python server.py
  --email=your@gmail.com
  --api-key=foobar   // you should change this
  --host=localhost
  --port=8000
  --cloud-host=https://www.projectatlantis.ai --cloud-port=3010
  --service-name=home
```

4. Sign up at https://www.projectatlantis.ai under the same email

5. Your remote(s) should autoconnect using email and default api key = 'foobar' (which you should change via '\user api_key' command). The first server to connect will be assigned your 'default'

6. If you run more than once remote, names must be unique

7. Initially the functions and servers folders will be empty

## Features

#### Dynamic Functions

- gives users the ability to create and maintain custom functions-as-tools, which are kept in the `dynamic_functions/` folder
- functions are loaded on start and should be automatically reloaded when modified
- you can either edit functions locally and the server will automatically detect changes, or edit remotely in the Atlantis cloud
- the first comment found in the function is used as the tool description
- dynamic functions can import each other and the server should correctly handle hot-loaded dependency changes, within the constraints of the Python VM
- every dynamic function has access to a generic `atlantis` utility module:

  ```python
  import atlantis

  ...

  atlantis.client_log("This message will appear in the Atlantis cloud console!")
  ```
-  the MCP spec is in flux and so the protocol between our MCP server and the cloud is a superset (we rely heavily on annotations)
- a lot of this stuff here may end up getting lumped under MCP "Resources" or something

#### Dynamic MCP Servers

- gives users the ability to install and manage third-party MCP server tools; JSON config files are kept in the `dynamic_servers/` folder
- each MCP server will need to be 'started' first to fetch the list of tools
- each server config follows the usual JSON structure that contains an 'mcpServers' element; for example, this installs an openweather MCP server:

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

The weather MCP service is just an existing one I ported to uvx. See [here](https://github.com/ProjectAtlantis-dev/atlantis-open-weather-mcp)


## Cloud

