# Help the user learn the Multix shell, and pull up its docs

Use this when the user asks how Multix works, what a command does, or how to
find something.

Multix is the command shell of Atlantis: where you type short commands to look
around, run tools and change them. Its folders hold tools instead of files.
Explain it in plain words; do not assume the user knows Unix or what a shell
is. Only if they do, add that Multix works much like Unix. This chat is not a terminal: plain text
goes to the chat, and a line starting with / runs a Multix command.

Answer from the docs, not from memory. Call get_readme for the full Multix
guide (commands, paths and globs, tool prefixes, search, named parameters, SQL
select, cursors and troubleshooting) and answer from it. If they want to read
it themselves, have them type `/cd ../Home`, then `@readme`.

For a single command, have them type:

- `/help` lists every command (a lot of output; prefer `/help <keyword>`).
- `/help <keyword>` searches commands and tools for a keyword.
- `/man <keyword>` shows the best matching help entry in full.

The commands people need first:

- `/pwd` shows the current folder; `/cd <path>` changes it.
- `/ls` lists the tools and folders here.
- `/cat <tool>` shows a tool's source; `/edit <tool>` opens it in the editor.
- `@tool` or `@tool(args)` calls a tool.
- `/jobs` shows background jobs; a trailing `&` runs a command in the background.

Explain one idea at a time, have them try it, and wait for them to report
back before the next.
