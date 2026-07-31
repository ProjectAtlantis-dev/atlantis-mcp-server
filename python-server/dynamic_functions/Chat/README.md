# Chat

A multi-user human and bot chat system. An MCP "game" is the resulting scenario
and should match 1:1 with the game.

## Concepts

- **Bot**: A character, persona, and engine unit. The bot primary key is its
  `sid`.
- **Slot**: A game roster slot. A slot can be assigned to either a human or a
  bot, so any game can mix humans and bots.
- **Camera**: A terminal/browser viewport bound to a location. Cameras can also
  follow roster slots.

## Bot Responses

Bot responses can be triggered by chat or by tick.

The chat callback must:

1. Pull the transcript.
2. Determine where the chat happened.
3. Determine who spoke.
4. Determine who was listening, if anyone.
5. Decide who should respond.
6. If a bot should respond, send the package to OpenRouter or the configured
   model provider.

## Runtime

These files are dynamically loaded by the Atlantis MCP server that hosts them.
The loader publishes functions according to their decorators. These functions
can interact with the Atlantis cloud system through the `atlantis.*` library,
usually by running commands directly rather than adding duplicate wrapper
methods to `atlantis.py`.

Server log: `python-server/runServer.log`

## Layout

- `Game/` holds static content: bots, locations, and scenes. Tracked in git.
- `Data/` holds live per-game state, keyed by `game_key`. Not tracked.
- `Data/games/<game_key>/tools/<bot_sid>.json` is that bot's authoritative
  per-game tool-name inventory.

Both resolve through `common.home_path()`, rooted at this folder.
