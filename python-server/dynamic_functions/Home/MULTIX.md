# Atlantis MCP tools

## Overview

Each Atlantis MCP acts as a filesystem node for Multix, our nix-like 'operating system' for future Greenland. However, since bots rely on tools, each folder in Multix contains functions instead of files. We feel this approach is closer to the original 1960s vision for UNIX, namely Multics.

- **readme** - this file
- **command** - use this to enter an Atlantis command
- **chat** - use this to just talk into the chat

## Commands

`command` lets humans or bots send commands to Multix. All commands should start with `/`. They kinda follow a Linux style shell approach. In fact, you can enable terminal mode to enter the Multix terminal directly and avoid having to prefix everything with slashes. The main difference is that each MCP exposes a virtual filesystem of sorts but of functions instead of files. The file containers are essentially unwrapped and then hotloaded so they are call ready. Note that the default container is usually main.py and more than one function can be in the same file. Generally, you should not have to care about the containing file except for versioning.

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
  - `cd H*me` - globs work within a segment (`*` never crosses `/`)
- `add` - add a function in the current location
- `edit` - DO NOT USE, use `codeset` instead (see below)
- `rls` - list the connected remotes

## Tools

You will start at the root of the Atlantis virtual filesystem arranged by usernames, and then you go into a user's home folder, and then connected MCP server for that user. You cannot go into disconnected servers.

To run a tool in the current folder, you can simply use `@name` plus any params, much like a JavaScript function e.g. `@foo` or `@foo(3,100)`

- `cat` - retrieves the text of a function
- `codeset` - sets the content of a function

## Paths and Globs

Paths follow Linux conventions. `/` is the ONLY path separator; dots are ordinary name characters.

- `*` and `?` glob WITHIN a single path segment — `*` never crosses a `/`
- `**` as a whole segment spans any number of folders (globstar)
- `.` and `..` resolve against your current directory
- Matching prefers the exact case first, then falls back to case-insensitive

Anchors (where a path starts):

- no prefix - relative to your current folder
- `/` or `%` - global root (root is arranged by user, so the first segment is a username)
- `~` - your home; `~name` is user *name*'s home (like Linux)
- `$` - root of the remote you are currently inside

## Tool Prefixes

- `@foo` - run `foo` from the current folder (searches PATH folders, nearest match wins)
- `@App/foo` - run `foo` in the App subfolder of the current folder
- `%user/remote/App/foo` or `/user/remote/App/foo` - fully qualified call
- `~/**/coffee` - your own `coffee`, anywhere under your home
- `%**/coffee` - the first `coffee` across all users (anywhere in the tree)
- `%*foo` - note: a single `*` is one segment, so this matches *users* ending in `foo`, not functions
- `$Tools/coffee` - `coffee` inside the `Tools` folder at the current remote's root

## Search Terms

Search terms are just glob paths. You can call a deep function without cd-ing to it first, as long as it resolves uniquely, e.g. `/brickhouse/terrain/InWork/foo` or `**/InWork/foo`.

If you just say `foo` from the top level it could be ambiguous which one you mean; the shell will show the candidates so you can pick a fuller path.

## Named Function Parameters

While purely positional parameters usually work, it is better to use explicitly named JSON arguments (parentheses are optional) to avoid escaping issues:

- `foo { x: 3, name: "chicago" }`
- `codeset { searchTerm: "bar", contents: "async def bar(): ... rest of code here ..." }`

The `help` command also provides parameter info.

## How the Description Field Is Populated

The first comment in a Python function is the description displayed in `search` and other various commands.

## SQL Select

When a command or tool returns tabular data, that is saved into a pseudo-table called `prior` and you can run `/select <cols> from prior where ...` if you want to narrow down prior results.

## MCP Servers

Be aware you can list, start, and stop classic MCP servers as well (which are more like plug-ins).

## Cursors

In practical programming terms, a cursor does what a call stack what normally do - it's simply a place to hold function parameters so you don't have to constantly type them in.  That is if you have a function `foo(x,y)` and the cursor already hold `x=3` then you only need to supply `y` on the command line.

Why cursors? Well although tools are functions, we don't have an interactive way to compose functions in a categories (think category theory) since there's no functional programming (FP) here and everyone hates FP anyway. So cursors play the role of monads ie the 'glue' btw functions. If you are familiar with Scala ZIO, think ZLayers; if you are familiar with Unison, think ability stack. Setting x=3 and then x=4 pushes two structures onto the Multix cursor stack and you can pop later.
