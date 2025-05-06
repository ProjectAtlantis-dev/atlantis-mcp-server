![happy](/happy.png)

# Project Atlantis MCP

The somewhat hand-wavey goal of Project Atlantis is simulating a future buildout of rugged, remote Greenland using emerging autonomous technologies such as robots and drones (both flying and sea-going) and eVTOLs that make such an effort even possible without exposing humans to undue risk

We also assume portable SMR nuclear and various other new technologies will also play key enabling roles. Most notably we want to experiment with AI governance to avoid the usual traps of human governance


## Motivation

Struggle is central to human progress, and the struggle of conquest through empire expansion has always been a driving force of civilizational growth

However, expansion along a frontier is fundamentally different from an adversarial, zero-sum empire conquest mindset. A frontier fundamentally represents a collaborative struggle against non-human elements. While the Roman Empire certainly had a notion of frontier, we would argue that the New World buildout fostered more innovation because of this collaboration

Not having to fight massive armies, subjugate large existing populations, or fight entrenched bureaucracies, the New World was a unique opportunity for humans to experiment with greenfielding civilization itself

Innovations required for the successful buildout of the New World demanded the establishment of meritocratic systems and freedoms that went beyond Roman military logistics and infrastructure

Unfortunately, once most buildouts were done, most of the Americas reverted back to adversarial mindset. The United States is the obvious outlier but did not avoid civil war either, and is now struggling with the paradox of overly centralized control

Meanwhile, the rise of China has broken every rule in the book. Against all odds, China overhauled an existing and very crowded nation in a very short time, largely because its ruling party is still haunted by memories of Communist rule. However, the China buildout is largely winding down; the glittering cities can only get so glittery, and China risks devolving back into a fixed-pie scarcity mindset trap unless they find a new frontier

This, of course, puts China potentially at odds with the United States, unless China and the US can develop new frontiers independently with minimal contention for shared resources


## The Robot Age

Ironically, the same autonomous AI technoligies that can be used for frontier can also be used for war. That is, while Greenland may seem like irrelevant Arctic folly today it may become a necessary refuge from robot and drone wars tomorrow, particularly for Europe

# MCP Server Reference Implementation

This project folder provides a flexible Model Context Protocol (MCP) implementation for creating and managing tools used by Atlantis bots

### Key Components

1. **Python MCP Server** (`python-server/`)
   - Runs locally but can be controlled remotely via an Atlantis cloud account
   - Provides a bridge between local resources and remote Atlantis services
   - Manages server-to-server communication in the MCP ecosystem

2. **MCP Client** (`client/`)
   - Local npx-based client for interacting with the Python MCP server
   - Useful for testing and controlling the server locally without requiring cloud access

### Main Features

#### Dynamic Functions

- Gives users the ability to create and maintain custom functions-as-tools in the `dynamic_functions/` folder
- Functions are loaded dynamically and automatically reloaded when modified
- You can either edit functions locally and the server will automatically detect changes, or edit remotely in the Atlantis cloud
- Functions can import each other and the server should correctly handle hot-loaded dependency changes, within the many constraints of the Python VM
- Every dynamic function has access to a generic `atlantis` utility module:
  ```python
  import atlantis
  atlantis.client_log("This message will appear in the Atlantis cloud console!")
  ```

#### Dynamic MCP Servers

- Gives users the ability to install and manage third-party MCP server tools in the `dynamic_servers/` folder
- This lets you host the growing ecosystem other MCP tools as if they were part of your own server
- Using the npx client, you host other MCP servers directly or use the Atlantis cloud
- each server config follows the usual JSON structure that contains an 'mcpServers' element
- for example, this installs our openweather MCP server:

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