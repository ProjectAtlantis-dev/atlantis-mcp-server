import json
import logging
import posixpath
from datetime import datetime
from pathlib import Path

import atlantis

logger = logging.getLogger("dynamic_function")

# Robot/ holds what Chat and the bot use for the app folder it sits in: the
# hooks and status Chat drives, and the bot's own tools (the walkthrough). That app folder's name is the Chat game name.
APP_NAME = Path(__file__).absolute().parents[1].name
# Data for the Chat game this app joins (Chat/<APP_NAME>/Data)
DATA_PATH = Path(__file__).absolute().parents[2] / "Chat" / APP_NAME / "Data"
# In each game's folder, per user: when they confirmed each walkthrough step.
# Progress lives with the game, so a new game starts the walkthrough over.
PROGRESS_FILE = "progress.json"
FOO_PATH = Path(__file__).absolute().parents[1] / "foo.py"
# In each game's folder: who this game's bot has met, {username: {lastSeen}}.
# Bot memory lives with the game, so a new game starts with a bot that has met nobody.
VISITORS_FILE = "visitors.json"


@visible
@index
async def index():
    """Robot info for Taffy"""
    return "Demo robot"


@visible
@enter
async def on_enter():
    """
    Runs when entering chat
    """
    logger.info(f"Executing Demo on_enter...")

    await atlantis.set_background(str(Path(__file__).absolute().parents[1] / "lobby.jpeg"), vertical_align="center")

    # The app folder name is the Chat game name; preflight and chat read it from the cursor
    await atlantis.client_command("/cursor join", {"game_name": APP_NAME})

    # Point the calling terminal at the app folder's chat shell (upserted by Node, keyed by folder)
    script_folder = atlantis.get_script_folder()
    if not script_folder:
        raise RuntimeError("No script folder; on_enter must run as a dynamic function")
    app_folder = posixpath.dirname(script_folder)
    await atlantis.client_command("/chat bind " + json.dumps(app_folder), shell="caller")

    return "Demo entered"


@visible
@exit
async def on_exit():
    """
    Runs when leaving chat
    """
    logger.info(f"Executing Demo on_exit...")

    # Walk out of the room while this folder's chat is still bound
    from dynamic_functions.Chat.roster import roster_leave
    await roster_leave(APP_NAME)

    await atlantis.clear_background()

    # Drop the game cursor joined in on_enter
    await atlantis.client_command("/cursor clear")

    # Detach the calling terminal from the app folder's chat shell; the chat shell stays
    await atlantis.client_command("/chat unbind", shell="caller")

    return "Demo exited"


@visible
async def get_status():
    """
    Demo status for the current user: who they are, whether you have met them
    in this game, every walkthrough step with whether it is done, and exactly
    what to do next. Chat runs this before every bot round, so it only reads.
    """
    logger.info(f"Executing get_status...")

    username = _caller()
    prefix, typing = await _typing_rules()
    steps = _check_steps(username, prefix)

    if username in _read_json(_game_path(VISITORS_FILE)):
        seen = f"You have already met {username} in this game; carry on from where you are."
    else:
        seen = (f"You have not met {username}. Call remember_visitor with the name {username!r}, "
                "then introduce yourself and start the walkthrough.")

    current = next((number for number, step in enumerate(steps, 1) if not step["done"]), None)
    lines = [f"The user is {username}. {seen}", typing, f"The Demo walkthrough has {len(DEMO_STEPS)} steps:"]
    for number, ((name, _, _), step) in enumerate(zip(DEMO_STEPS, steps), 1):
        if step["done"]:
            label = "done"
        elif number == current:
            label = "NEXT"
        else:
            label = f"waiting on step {step['waitsOn']}"
        lines.append(f"{number}. [{label}] {name}: {step['state']}")
    if current is None:
        lines.append("Next: the walkthrough is complete. Offer to find them any other tool: search "
                     "covers every folder on their search path, "
                     f"which they can see with {prefix}path show.")
    else:
        lines.append(f"Next (step {current}): {steps[current - 1]['fix']}")
    lines.append("If the user would rather explore on their own, call list_demo_tools.")
    return "\n".join(lines)


@visible
async def record_visit():
    """
    Record that the bot has met the current user in this game. Chat runs this
    once a bot's reply has gone out.
    """
    username = _caller()
    visitors_path = _game_path(VISITORS_FILE)
    visitors = _read_json(visitors_path)
    visitors[username] = {"lastSeen": _now()}
    _write_json(visitors_path, visitors)
    return f"Recorded {username}'s visit"


async def _typing_rules() -> tuple[str, str]:
    """(command prefix, how the user types and where they are) from the
    caller's shell env: ui.appMode is set when they type in the chat window,
    where a Multix command needs a leading /; a terminal takes it bare.
    """
    shell = atlantis.get_caller_shell_path()
    if not shell:
        raise RuntimeError("No caller shell; get_status must run from the user's terminal")
    ui = await atlantis.client_command(f"/env peek {shell} ui")
    path = await atlantis.client_command(f"/env peek {shell} path")
    folder = "/" + "/".join(path["current"])
    script_folder = atlantis.get_script_folder()
    if not script_folder:
        raise RuntimeError("No script folder; get_status must run as a dynamic function")
    app_folder = posixpath.dirname(script_folder)

    if ui.get("appMode") is True:
        prefix = "/"
        rules = ("The user types in the chat window: plain text goes to the chat, a line starting "
                 "with / runs a Multix command and a line starting with @ calls a tool, so every "
                 "command below starts with /.")
    else:
        prefix = ""
        rules = ("The user types in a Multix terminal, not the chat window: a line runs as a "
                 "Multix command exactly as typed and a line starting with @ calls a tool. Never "
                 "tell them to put / in front of a command.")
    if folder == app_folder:
        where = f"Their current folder is the Demo folder, {folder}."
    else:
        where = (f"Their current folder is {folder}, not the Demo folder; before anything else, "
                 f"have them type {prefix}cd {app_folder} and wait.")
    return prefix, f"{rules} {where}"


def _check_steps(username: str, prefix: str) -> list[dict]:
    """One {done, state, fix, waitsOn} per DEMO_STEPS entry, in order, with
    each fix's commands written with the user's command prefix"""
    confirmed = _read_json(_game_path(PROGRESS_FILE)).get(username, {})
    steps: list[dict] = []
    for number, (name, key, fix) in enumerate(DEMO_STEPS, 1):
        # The walkthrough is a sequence: each step needs the one before it
        if number > 1 and not steps[-1]["done"]:
            steps.append({"done": False, "state": f"needs step {number - 1} first.",
                          "fix": None, "waitsOn": number - 1})
        elif key in confirmed:
            steps.append({"done": True, "state": f"the user finished it on {confirmed[key]}.",
                          "fix": None, "waitsOn": None})
        else:
            steps.append({"done": False, "state": "not done yet.",
                          "fix": fix.replace("{sh}", prefix), "waitsOn": None})
    return steps


def _confirm(key: str) -> str:
    return f"Only after they confirm, call confirm_demo_step with step \"{key}\"."


# (name, confirm_demo_step key, what to do next). Ported from the parked MCP
# onboarding (atlantis_nodejs resources/mcp/onboarding-demo.md). {sh} marks where
# a Multix command needs its prefix: "/" when the user types in the chat window,
# nothing in a terminal (see _typing_rules).
DEMO_STEPS = [
    ("list the Demo tools", "ls", (
        "have them type {sh}ls, which lists the functions in the Demo folder. Ask whether they see the list. "
        "End your turn and wait. " + _confirm("ls"))),
    ("run foo", "foo", (
        "ask whether they would like a quick introduction to the Multix shell, and wait. If yes, "
        "explain in a few plain sentences: Multix is where you type commands to explore and run "
        "tools, and its folders hold functions rather than ordinary files; {sh}pwd shows the current "
        "folder and {sh}ls lists its functions. Then have them type @foo(3,4), which calls the foo "
        "function and should return 7. Ask them to confirm they got 7, and wait; resolve any error "
        "first. " + _confirm("foo"))),
    ("read foo's source", "source", (
        "have them type {sh}cat foo, which shows the source code of the foo function they just ran. "
        "Ask whether they see it, and wait. Then explain this is the same code as the file "
        f"{FOO_PATH}, which they can open and edit in VS Code. Point out the @public decorator on "
        "foo: it means anyone can run this function, which lets you share things with other teams "
        "very quickly, though others cannot see or change its code without more permissions. Then "
        "point out that the function's docstring is the description {sh}ls shows for foo, \"Add two "
        "integers\". " + _confirm("source"))),
    ("edit foo", "edit", (
        f"ask them to edit foo in {FOO_PATH} themselves so it returns x + y + 100, save the file, "
        "and type @foo(3,4) again. The server reloads functions when a file is saved, so no "
        "restart is needed, and it should now return 107. Do not make the edit for them. End your "
        "turn and wait for them to confirm 107; resolve any error first. " + _confirm("edit"))),
    ("see images and video", "media", (
        "have them type @myImage, which shows how functions return images; ask whether they see "
        "the image, and wait. Then have them type @myVideo, which shows how functions return "
        "video; ask whether they see it play, and wait. " + _confirm("media"))),
    ("see tables", "table", (
        "have them type @myTable: it returns a list of anime heroines, and Atlantis shows "
        "tabular data like this as a table. Ask whether they see the table, and wait. Then have "
        "them type {sh}cat myTable to see its source: it just returns a list of rows, one dictionary "
        "per row with the same keys, and Atlantis renders that as a table. Ask whether they see "
        "it, and wait. Then have them type @myTableWithImages: the same heroines with their "
        "creators and a thumbnail of each, because a column holding an image data URI shows as a "
        "picture. Ask whether they see the thumbnails, and wait. Then tell them any table can be "
        "queried with SQL: wrapping a command in square brackets turns its output into a table to "
        "select from. Have them type {sh}select * from [myTable] where name like 'M%', which shows "
        "only the heroines whose names start with M. Ask whether they see the filtered table, and "
        "wait. Then have them type {sh}select * from [myTableWithImages] where name like 'M%' order "
        "by name desc, which runs the same filter on the table with thumbnails, sorted by name in "
        "reverse. Ask whether they see the sorted table with images, and wait. Then have them pull "
        "out just the first row by wrapping that query in parentheses and indexing it: "
        "{sh}(select * from [myTableWithImages] where name like 'M%' order by name desc)[0]. A single "
        "record shows as JSON instead of a table, and its image field is the raw base64 data URI "
        "rather than a picture. Ask whether they see the JSON, and wait. Then "
        "have them pull just that row's image by adding .image: "
        "{sh}(select * from [myTableWithImages] where name like 'M%' order by name desc)[0].image. "
        "Ask whether they see the image, and wait. Then show that tables can be joined, like in "
        "any database: naming each bracketed command with AS lets the query match rows across "
        "them. Have them type {sh}select t.name, t.series, i.creator from [myTable] AS t join "
        "[myTableWithImages] AS i on t.name = i.name, which puts each heroine's series from the "
        "first table next to her creator from the second. Ask whether they see the joined table, "
        "and wait. " + _confirm("table"))),
    ("say hello", "hello", (
        "tell them they are past the halfway point of the demo. Then have them type @hello, which "
        "greets them by their Atlantis username. Ask whether they see the greeting, and wait. "
        "Then point out that since a function knows exactly who is calling it, it can also "
        "control access: decide who may run it, or show each caller only their own data. "
        + _confirm("hello"))),
    ("open the code editor", "editor", (
        "have them type {sh}edit hello, which opens the built-in Multix code editor on the hello "
        "function. Unlike {sh}cat hello, which only shows the source, edit lets them change the "
        "function and save it. Ask whether they see the editor, and wait. Point out that hello uses "
        "the atlantis package, which provides many commands such as get_caller(). Then ask them to "
        "close the editor; closing does not save. Wait for them to confirm. " + _confirm("editor"))),
    ("run the editor in the background", "background", (
        "explain that while the editor was open, the terminal was waiting on it. Have them type "
        "{sh}edit hello & instead: like Unix, a trailing & runs the command in the background, so the "
        "editor opens and the terminal stays free. Ask whether the editor opened while the terminal "
        "is still usable, and wait. " + _confirm("background"))),
    ("run commands from the editor", "command", (
        "in the editor, have them make a fresh blank line above @visible and press Cmd+Right Arrow "
        "(Ctrl+Right Arrow on Windows or Linux) on it, which inserts a # % command comment; hello.py "
        "may already have example # % comments, which run the same way. Have them type ls after the "
        "% and press Cmd+Right Arrow again to run it from the editor: the terminal lists the Demo "
        "functions with their descriptions. Ask whether they see the list, and wait. Then have them "
        "change the command to hello and run it; ask whether they see the greeting, and wait. Then "
        "have them change the docstring under def hello() to something new and save with Cmd+S "
        "(Ctrl+S on Windows or Linux): the docstring is the description ls shows. Wait. Then have "
        "them change the command to ls hello and run it; ask whether they see the new description, "
        "and wait. Then have them close the editor and type {sh}clear to clear the terminal, and wait. "
        "Tell them the Demo walkthrough is done. Only after they confirm the list, the greeting, the "
        "new description and a cleared terminal, call confirm_demo_step with step \"command\".")),
]
DEMO_STEP_KEYS = [key for _, key, _ in DEMO_STEPS]


def _caller() -> str:
    username = atlantis.get_caller()
    if not username:
        raise RuntimeError("Demo has no caller identity")
    return username


def _game_path(filename: str) -> Path:
    """A file in this game's folder under DATA_PATH/games.

    Walkthrough state lives with the game, so a new game starts from scratch.
    The game folder must already exist (Chat creates it on join).
    """
    game_uuid = atlantis.get_game_uuid()
    if not game_uuid:
        raise RuntimeError("No game_uuid; this must be called from the Demo chat")
    game_dir = DATA_PATH / "games" / game_uuid
    if not game_dir.is_dir():
        raise FileNotFoundError(f"No game folder for game {game_uuid!r} at {game_dir}")
    return game_dir / filename


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
