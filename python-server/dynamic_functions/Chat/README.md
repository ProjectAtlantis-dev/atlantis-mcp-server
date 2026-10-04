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

## Game identity

Node sends `game_uuid` on remote tool calls. It is the sole lookup key for
`Data/games/<game_uuid>/game.json`; Python does not allocate another game UUID.
Preflight and chat resolve this record directly. If it is absent, only a caller
recognized by `atlantis.is_owner()` may create it. `game_new()` writes the
record with the caller as its initial member, joins the cursor, and runs the
existing `game_init()` scene, roster, and camera dialogs. Completing setup
starts the game. Calling `game_new()` for an existing record without a roster
also runs setup; configured records are reused. Once a roster is assigned, a chat
message from the game owner resumes a stopped game and is processed normally.
The speaker comes from the transcript: the callback caller alone does not prove
who spoke. Visitor messages in stopped games raise a permission error back to
Node. Resuming this way
preserves the transcript. Membership checks for game operations remain separate
from lookup.

Session keys are `<caller_sid>:<game_uuid>`. The numeric `user_game_id` remains
available as diagnostic data and for the existing Node browser-window URLs;
it does not select a local game. Missing UUIDs are errors, with no numeric-ID
fallback. Older directories named with Python-generated keys are not migrated
automatically. Node and Python need this protocol update together.

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

- `Game/` holds static content: locations and scenes. Tracked in git.
- Bots live in the sibling `Bot/` app, possibly on another machine. Chat
  knows a bot only once it joins via `roster_join_bot`, which caches the full
  bot config at `Data/games/<game_key>/bots/<sid>.json`.
- `Data/` holds live per-game state, keyed by `game_key`. Not tracked.
- `Data/games/<game_key>/tools/<bot_sid>.json` is that bot's authoritative
  per-game tool-name inventory.

Both resolve through `common.home_path()`, rooted at this folder.
