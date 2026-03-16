
## Atlantis MCP tools

# overview

You should have an MCP called Atlantis, with the following ways to access the wonderful Project Atlantis system, which helps us manage automation across the island of Greenland:

readme - secret daily password and any other latest info if any
command - use this to enter an Atlantis command
function - use this to run an Atlantis 'tool'


# commands

'command' lets you send commands to Atlantis. It kinda follows a linux style shell approach. Some interesting commands to get you started:
  help - shows all the keywords
  ls - list contents of current folder, usually functions or subfolders
  dir - list contents from root (ignores current folder)
  tree - list all contents from current position
  pwd - show current directory (if reconnecting)
  search - search functions by description
  history - show shell history
  whoami - show your username
  cd - change into a folder, some special notes:
    cd /   - go to root, root is arranged by user
    cd ~   - go to your home, arranged by connected servers
    cd ..  - go back up one folder

  add - add a function in the current location
  edit - do not use because this brings up a ux, use _function_set instead (see below)

You will start at the top level of the Atlantis virtual filesystem which is really just a list of usernames, and then you go into the appropriate MCP service.


# functions

All functions are Python functions. To run a function you can simply use the name plus any params, much like a javascript function e.g. 'foo' or 'foo(...)'
  cat - retrieves the text of a function
  set - sets the content of a function (you will need to use a search term and named params, see below); also easier to switch to current folder before trying to run or you might update the wrong one or get 'too many matching tools' error


# search terms

Search terms allow you to specify a function using wildcards without having to cd to that folder first, assuming it resolves uniquely e.g. 'brickhouse\*terrain\*InWork\*\*foo' could also be called via '\*Inwork\*foo'

If you just say 'foo' from the top-level it could be ambiguous which one you mean.


# named function parameters

While purely positional parameters usually work, it is better to  use explicitly named JSON arguments (parenthesis are optional) to avoid escaping issues:
  foo { x: 3, name: "chicago" }
  set { searchTerm: "bar", contents: "async def bar(): ... rest of code here ..." }

The help command also provides parameter info


# description

The first comment in the Python function is taken for the description displayed in 'search' and other various commands


# select

Any time you get a table back, the data is loaded into a pseudo-table called 'prior' and you can SELECT against that

