![terminal](/terminal.png)

# Project Atlantis
Meow! Ideally, you may want to create an account first at www.projectatlantis.ai and then have the bot walk you through setup (assuming everything works okay)

Basically we have a distributed linux-style system that provides tool infra for bots. Tools are arranged in folders for easy management across functions and teams. Teams can call each other's functions directly or of course the bots can just do things themselves. Under the covers is an MCP-compliant system but we support hotloading etc. without some of the clunky overhead of constantly updating MCP tools.

To get started, clone the repo, do the Python env stuff, set up your OPENROUTER API KEY (or whatever) in Home/kitty.py and connect this local Python server to the main server (see runServer). We give you all the source code to build your own tool-calling chatbot just like Claude or whatever

*note that Home/game.py is run whenever a new chat is created and will set the default chat tool to Home/kitty.py


## Project Atlantis Network

Each MCP server is part of a collaborative network of AI agents and developers. Using the Model Context Protocol, the platform creates an ecosystem where agents can discover and use each other's capabilities across the network. Tools and functions can be shared, discovered, and coordinated between agents—whether for robot-driven frontier development, automation tasks, or any other application. The network architecture enables agents to find and leverage tools from other users, creating a decentralized ecosystem of shared capabilities.

The centerpiece of this project is a Python MCP host (referred to as a 'remote') that lets you install functions and 3rd party MCP tools on the fly

## Quick Start

1. Prerequisites - need to install Python for the server and Node for the MCP client; you should also install uv/uvx and node/npx since it seems that MCP needs both


2. Python 3.13 seems to be most stable right now because of async support

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

To add Atlantis Open Weather for testing:

```claude mcp add --transport stdio weather_forecast --env OPENWEATHER_API_KEY=mykey123 -- uvx --from atlantis-open-weather-mcp start-weather-server```

5. To connect to Atlantis, sign into https://www.projectatlantis.ai under the same email

6. Your remote(s) should autoconnect using email and default api key = 'foobar' (see 'api' command to generate a new key later). The first server to connect will be assigned your 'default' unless you manually change it later

7. Initially the functions and servers folders will be empty except for some examples

8. You can run this standalone MCP or accessed from the cloud or both

### Architecture

Caveat: MCP terminology is already terrible and calling things 'servers' or 'hosts' just makes it more confusing because MCP is inherently p2p

Pieces of the system:

- **Cloud**: our experimental Atlantis cloud server; mostly a place to share tools and let users bang on them
- **Remote**: the Python server process found in this repo, officially referred to as an MCP 'host' (you can run >1 either on same box or on different one, just specify different service names)
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

For detailed information about creating and using dynamic functions, see the [Dynamic Functions Documentation](python-server/README.dynamic_functions.md).

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

The cloud service at https://www.projectatlantis.ai provides a centralized hub for managing your remote servers and sharing tools across machines.

### App Organization

Dynamic functions are organized into apps using **folder structure**. Simply place your `.py` files in subdirectories:

```
dynamic_functions/
├── Home/                    # App: "Home"
│   └── kitty.py
├── Accounting/              # App: "Accounting"
│   ├── accounting.py
│   └── foo.py
└── FilmFromImage/          # App: "FilmFromImage"
    └── qwen_image_edit_local.py
```

**The folder name IS the app name.** Functions in `Home` folder are assigned accordingly.

#### Nested Apps (Subfolders)

Create nested app structures using subfolders:

```
dynamic_functions/
└── MyApp/
    └── SubModule/
        └── Feature/
            └── my_function.py
```

This creates the app name: `MyApp/SubModule/Feature`

**Best Practices:**
- Keep it simple - one level of folders is usually enough
- Use descriptive folder names (e.g., `Chat`, `Admin`, `Tools`)
- Group related functions together in the same folder
- The folder structure keeps your code organized and clear

### Tool Calling with Search Terms

When calling tools, you can use **compound tool names** to disambiguate functions. **Only include as much of the path as needed to uniquely identify the function.**

**Format:** `remote_owner*remote_name*app*location*function`

**Key Principle: Use the simplest form that resolves uniquely**

```python
# If you have these functions:
# - dynamic_functions/Chat/send_message.py
# - dynamic_functions/Email/send_message.py
# - dynamic_functions/SMS/send_message.py

send_message              ❌ Ambiguous! Which one?
**Chat**send_message      ✅ Clear! The one in Chat
**Email**send_message     ✅ Clear! The one in Email
```

**Examples:**

```
update_image                          → Simple call (only works if unique)
**MyApp**update_image                 → Specify app to disambiguate
**MyApp/SubModule**process_data       → Nested app path
alice*prod*Admin**restart             → Full routing: owner + remote + app + function
***office*print                       → Just location context
```

**How it works:**
- Fields: `remote_owner*remote_name*app*location*function`
- Separate fields with `*` (asterisk)
- **Omit fields you don't need** (use empty strings: `**App**func`)
- The app field supports slash notation for nested apps (`MyApp/SubModule`)
- The last field is always the function name
- No asterisks = treat entire name as function name

**When to use compound names:**
- **Name conflicts**: Multiple apps have functions with the same name
- **Remote targeting**: Call functions on specific remotes from the cloud
- **Location routing**: Target functions at specific physical locations
- **Multi-user setups**: Specify owner and remote in shared environments

**Best practice:** Start simple (`update_image`) and add context only when needed to resolve ambiguity (`**ImageTools**update_image`).

**Example:**
```python
# File: dynamic_functions/ImageTools/process.py
@visible
async def update_image(image_path: str):
    """Update an image."""
    return "updated"

# If this is the ONLY update_image:
update_image                          ✅ Works fine!

# If Chat app ALSO has update_image:
**ImageTools**update_image            ✅ Now we need to specify the app
```

## Our Greenland Terrain Server
![happy](/happy.png)

The goal is to use this system as the main bot infrastructure (tool etc.) for our [Greenland terrain server](https://github.com/ProjectAtlantis-dev/atlantis-terrain-server)