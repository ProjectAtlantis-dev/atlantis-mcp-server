import json
import logging
from datetime import datetime
from pathlib import Path

import atlantis

logger = logging.getLogger("dynamic_function")

# Data for the Chat game this app joins (Chat/Demo/Data)
DATA_PATH = Path(__file__).absolute().parents[1] / "Chat" / Path(__file__).parent.name / "Data"
# In each game's folder, per user: when they confirmed each walkthrough step.
# Progress lives with the game, so a new game starts the walkthrough over.
PROGRESS_FILE = "progress.json"
FOO_PATH = Path(__file__).absolute().parent / "foo.py"


@index
@public
async def index():
    """Demo app"""
    return "Welcome to the Atlantis demo app!"


@visible
async def confirm_demo_step(step: str):
    """
    Record that the user finished a step of the Demo walkthrough: "ls", "foo",
    "source", "edit", "media", "table", "hello", "editor", "background" or "command".
    Call it only after the user has actually said so.
    """
    logger.info(f"Executing confirm_demo_step({step!r})...")

    if step not in DEMO_STEP_KEYS:
        raise ValueError(f"step must be one of {', '.join(DEMO_STEP_KEYS)}, not {step!r}")
    username = _caller()
    steps = _check_steps(username)
    number = DEMO_STEP_KEYS.index(step) + 1
    if steps[number - 1]["waitsOn"]:
        waits_on = steps[number - 1]["waitsOn"]
        raise RuntimeError(f"Cannot confirm {step!r}: step {waits_on} ({DEMO_STEPS[waits_on - 1][0]}) comes first")
    if step == "edit":
        # The edit is the one step the server can see for itself
        result = await atlantis.client_command("@$Demo/foo", {"x": 3, "y": 4})
        if result != 107:
            raise RuntimeError(f"Cannot confirm 'edit': foo(3, 4) returns {result!r}, not 107")

    progress_path = _game_path(PROGRESS_FILE)
    progress = _read_json(progress_path)
    progress.setdefault(username, {})[step] = _now()
    _write_json(progress_path, progress)
    return f"Recorded that {username} finished {step!r}."


def _check_steps(username: str) -> list[dict]:
    """One {done, state, fix, waitsOn} per DEMO_STEPS entry, in order"""
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
            steps.append({"done": False, "state": "not done yet.", "fix": fix, "waitsOn": None})
    return steps


def _confirm(key: str) -> str:
    return f"Only after they confirm, call confirm_demo_step with step \"{key}\"."


# (name, confirm_demo_step key, what to do next). Ported from the parked MCP
# onboarding (atlantis_nodejs resources/mcp/onboarding-demo.md). This chat runs
# in the user's terminal: lines starting with / are shell commands and lines
# starting with @ call tools.
DEMO_STEPS = [
    ("list the Demo tools", "ls", (
        "the user is already in a terminal, in the Demo folder: this chat runs in it. Have them "
        "type /ls, which lists the functions in the Demo folder. Ask whether they see the list. "
        "End your turn and wait. " + _confirm("ls"))),
    ("run foo", "foo", (
        "ask whether they would like a quick introduction to the Multix shell, and wait. If yes, "
        "explain in a few plain sentences: Multix is where you type commands to explore and run "
        "tools, and its folders hold functions rather than ordinary files; /pwd shows the current "
        "folder and /ls lists its functions. Then have them type @foo(3,4), which calls the foo "
        "function and should return 7. Ask them to confirm they got 7, and wait; resolve any error "
        "first. " + _confirm("foo"))),
    ("read foo's source", "source", (
        "have them type /cat foo, which shows the source code of the foo function they just ran. "
        "Ask whether they see it, and wait. Then explain this is the same code as the file "
        f"{FOO_PATH}, which they can open and edit in VS Code. Point out the @public decorator on "
        "foo: it means anyone can run this function, which lets you share things with other teams "
        "very quickly, though others cannot see or change its code without more permissions. Then "
        "point out that the function's docstring is the description /ls shows for foo, \"Add two "
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
        "them type /cat myTable to see its source: it just returns a list of rows, one dictionary "
        "per row with the same keys, and Atlantis renders that as a table. Ask whether they see "
        "it, and wait. Then have them type @myTableWithImages: the same heroines with their "
        "creators and a thumbnail of each, because a column holding an image data URI shows as a "
        "picture. Ask whether they see the thumbnails, and wait. Then tell them any table can be "
        "queried with SQL: wrapping a command in square brackets turns its output into a table to "
        "select from. Have them type /select * from [myTable] where name like 'M%', which shows "
        "only the heroines whose names start with M. Ask whether they see the filtered table, and "
        "wait. Then have them type /select * from [myTableWithImages] where name like 'M%' order "
        "by name desc, which runs the same filter on the table with thumbnails, sorted by name in "
        "reverse. Ask whether they see the sorted table with images, and wait. Then have them pull "
        "out just the first row by wrapping that query in parentheses and indexing it: "
        "/(select * from [myTableWithImages] where name like 'M%' order by name desc)[0]. A single "
        "record shows as JSON instead of a table, and its image field is the raw base64 data URI "
        "rather than a picture. Ask whether they see the JSON, and wait. Then "
        "have them pull just that row's image by adding .image: "
        "/(select * from [myTableWithImages] where name like 'M%' order by name desc)[0].image. "
        "Ask whether they see the image, and wait. " + _confirm("table"))),
    ("say hello", "hello", (
        "tell them they are past the halfway point of the demo. Then have them type @hello, which "
        "greets them by their Atlantis username. Ask whether they see the greeting, and wait. "
        "Then point out that since a function knows exactly who is calling it, it can also "
        "control access: decide who may run it, or show each caller only their own data. "
        + _confirm("hello"))),
    ("open the code editor", "editor", (
        "have them type /edit hello, which opens the built-in Multix code editor on the hello "
        "function. Unlike /cat hello, which only shows the source, edit lets them change the "
        "function and save it. Ask whether they see the editor, and wait. Point out that hello uses "
        "the atlantis package, which provides many commands such as get_caller(). Then ask them to "
        "close the editor; closing does not save. Wait for them to confirm. " + _confirm("editor"))),
    ("run the editor in the background", "background", (
        "explain that while the editor was open, the terminal was waiting on it. Have them type "
        "/edit hello & instead: like Unix, a trailing & runs the command in the background, so the "
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
        "and wait. Then have them close the editor and type /clear to clear the terminal, and wait. "
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
