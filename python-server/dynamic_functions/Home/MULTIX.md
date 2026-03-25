# Atlantis MCP tools

## Overview

You should have an MCP called Atlantis, with the following ways to access the wonderful Project Atlantis system, which helps us manage automation across the island of Greenland:

- **readme** - this file, see bottom for secret word of the day
- **command** - use this to enter an Atlantis command
- **chat** - use this to just talk into the chat

## Commands

`command` lets you send commands to Atlantis. All commands should start with `/`. They kinda follow a linux style shell approach. Folders contain tools instead of files. Tools are usually Python or functions hot-loaded in the dynamic_functions folder. Note that functions are contained in Python container files such as main.py and more than one function can be in the same file. Generally, you should not have to care about the containing file.

Some interesting commands to get you started:

- `help` - shows all the keywords (**warning:** calling with no arguments dumps a LOT of output; prefer `help <topic>` instead)
- `help <topic>` - does fuzzy search of both keywords and tools
- `ls` - list contents of current folder
- `dir` - list contents from root (ignores current folder)
- `tree` - list all contents from current location
- `pwd` - show current directory (if reconnecting)
- `search` - search functions by description
- `history` - show shell history
- `whoami` - show your username
- `cd` - change into a folder, some special notes:
  - `cd /` - go to root, root is arranged by user
  - `cd ~` - go to your home, arranged by connected servers
  - `cd ..` - go back up one folder
- `add` - add a function in the current location
- `edit` - DO NOT USE, use `set` instead (see below)
- `rls` - list the connected remotes

## Tools

You will start at the root of the Atlantis virtual filesystem arranged by usernames, and then you go into a user's home folder, and then connected MCP server for that user. You cannot go into disconnected servers.

To run a tool in the current folder, you can simply use `@name` plus any params, much like a javascript function e.g. `@foo` or `@foo(3,100)`

- `cat` - retrieves the text of a function
- `set` - sets the content of a function

## Tool Prefixes

- `@foo` - run in current folder, assuming you navigated your way down
- `%*foo` - run from global root; needs wildcard search term (e.g. `%*coffee` finds the first `coffee` across all users)
- `~*foo` - run from user home folder; the path after `~` starts at remote level (`remote*app**function`), so `~coffee` looks for a *remote* called `coffee`. Use `~*coffee` to wildcard the remote, or be explicit: `~terrain*Tools**coffee`

Note: `%` and `~` require search term syntax (with `*` wildcards) — a bare name like `%foo` or `~foo` will try to match a remote/user, not a function. See Search Terms below.

## Search Terms

Search terms allow you to specify a function using wildcards without having to cd to that folder first, assuming it resolves uniquely e.g. `brickhouse*terrain*InWork**foo` could also be called via `*Inwork*foo`

If you just say `foo` from the top-level it could be ambiguous which one you mean.

## Named Function Parameters

While purely positional parameters usually work, it is better to use explicitly named JSON arguments (parenthesis are optional) to avoid escaping issues:

- `foo { x: 3, name: "chicago" }`
- `set { searchTerm: "bar", contents: "async def bar(): ... rest of code here ..." }`

The `help` command also provides parameter info.

## How the Description Field Is Populated

The first comment in a Python function is the description displayed in `search` and other various commands.

## SQL Select

When a command or tool returns tabular data, that is saved into a pseudo-table called `prior` and you can run `/select <cols> from prior where ...` if you want to narrow down prior results.

## MCP Servers

Be aware you can list, start, and stop classic MCP servers as well (which are more like plug-ins).
