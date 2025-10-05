![happy](/happy.png)

# Project Atlantis
Meow! Ideally, you may want to create an account first at www.projectatlantis.ai and then have the bot walk you through setup (assuming everything works okay)

I wrote this to get past the hype and learn what an MCP (Model Context Protocol) server was, as well as explore potential future directions. I'm not really a Python programmer so this repo suffers from a bit vibe coding, which I've been trying to clean up

## MCP Python Remote Server

What is MCP (Model Context Protocol) and why all the fuss? Well, MCP gives bots all sorts of agentic capabilities but LangChain was doing that a year ago. APIs are not new either. What is new about MCP is that it inverts the traditional cloud architecture and lets you run more stuff locally and control what is going on. The downside is more setup headache

The centerpiece this project is a Python MCP host (which I call a 'remote') that lets you install functions and 3rd party MCP tools on the fly

## Quick Start

1. Prerequisites - need to install Python for the server and Node for the MCP client; you should also install uv/uvx and node/npx since it seems that MCP needs both


2. Python 3.12 seems to be most stable right now, 3.13 is iffy

3. Edit the runServer script in the `python-server` folder and set the email and service name (it's actually best practice to create a copy "runServerFoo" that you can replace the runServer file with when we do updates):

```bash
python server.py  \
  --email=youremail@gmail.com  \             # email you use for project atlantis
  --api-key=foobar \                         # should change online
  --host=localhost \                         # npx MCP will be looking here to connect to remote (assumes there is at least one running locally)
  --port=8000  \
  --cloud-host=wss://projectatlantis.ai  \   # points to cloud
  --cloud-port=443  \
  --service-name=home                        # remote name, can be anything but must be unique across all machines
```
4. To use this as a regular standalone MCP server, add the following config to Windsurf or Cursor or whatever:

```json
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
```

To add Atlantis to Claude Code, this should work:

```claude mcp add atlantis -- npx atlantis-mcp --port 8000```

5. To connect to Atlantis, sign into https://www.projectatlantis.ai under the same email

6. Your remote(s) should autoconnect using email and default api key = 'foobar' (see '\user api_key' command to change). The first server to connect will be assigned your 'default' unless you manually change it later

7. Initially the functions and servers folders will be empty except for some examples

8. You can run this standalone MCP or accessed from the cloud or both

### Architecture

Caveat: MCP terminology is already terrible and calling things 'servers' or 'hosts' just makes it more confusing because MCP is inherently p2p

Pieces of the system:

- **Cloud**: our experimental Atlantis cloud server; mostly a place to share tools and let users bang on them
- **Remote**: the Python server process found in this repo, which I think is officially called an MCP 'host' (you can run >1 either on same box or on different one, just specify different service names)
- **Dynamic Function**: a simple Python function that you write, acts as a tool
- **Dynamic MCP Server**: any 3rd party MCP, stored as a JSON config file

![design](/design.png)

Note that MCP auth and security are still being worked out so using the cloud for auth is easier right now

### Directories

1. **Python Remote (MCP P2P server)** (`python-server/`)
   - Location of our 'remote'. Runs locally but can be controlled remotely

2. **MCP Client** (`client/`)
   - lets you treat the remote like any another MCP
   - uses npx (easy to install into claude or cursor)
   - cloud connection not needed - although it may complain
   - only supports a subset of the spec
   - can only see tools on the local box (at least right now) or shared
     tools set to 'public'


## Features

#### Dynamic Functions

Dynamic functions give users the ability to create and maintain custom functions-as-tools, which are kept in the `dynamic_functions/` folder. Functions are loaded on start and automatically reloaded when modified.

For detailed information about creating and using dynamic functions, see the [README in the python-server folder](python-server/README.dynamic_functions.md).

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

