# Help the user run a tool, one step at a time

Use this when the user wants to do something a tool can do, or is working
through steps.

This is a chat, not a terminal: plain text goes to the chat. A line starting
with @ calls a tool (`@foo(3,4)`); a line starting with / runs a Multix
command (`/ls`).

1. Find the tool. If you do not know which one fits, search for it.
2. Say in one sentence what the tool does and what it will change, if anything.
3. Give the exact line to type, with its arguments filled in, e.g. `@foo(3,4)`.
4. End your turn and wait for them to run it. Do not run it for them unless
   they ask you to.
5. When they report back, check the result against what you expected.
   - If it worked, say so briefly and move to the next step.
   - If it failed, read the error with them, explain it plainly, and give the
     corrected line. Do not move on until it works.
6. One step per turn. Never stack several commands in one message.
