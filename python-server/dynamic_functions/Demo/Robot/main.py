import json
import logging
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import atlantis

logger = logging.getLogger("dynamic_function")

# Robot/ holds what Chat and the bot use for the app folder it sits in: the
# hooks and status Chat drives, and the bot's own tools (the walkthrough). That
# app folder's name is the Chat game name. Chat itself keeps who the bot has met
# and tells it how the user types (see Chat/turn.py, _robot_status).
APP_NAME = Path(__file__).absolute().parents[1].name
# Data for the Chat game this app joins (Chat/<APP_NAME>/Data)
DATA_PATH = Path(__file__).absolute().parents[2] / "Chat" / APP_NAME / "Data"
# In each game's folder, per user, their walkthrough:
#   actions: {action key: {event, at}}, the command event that finished each
#   after:   the last transcript event id when they reset it; only commands
#            after it count (absent until a reset)
#   ended:   when the bot was first told it is complete, set by the reply that
#            told them (absent until then)
# Progress lives with the game, so a new game starts the walkthrough over.
PROGRESS_FILE = "progress.json"
FOO_PATH = Path(__file__).absolute().parents[1] / "foo.py"
HELLO_PATH = Path(__file__).absolute().parents[1] / "hello.py"


def _caller_terminal() -> str:
    """The terminal that called this tool, whose transcript proves what they ran"""
    shell = atlantis.get_caller_shell_path()
    if not shell:
        raise RuntimeError("No caller shell; this must run from the user's terminal")
    return shell


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
    from dynamic_functions.Chat.zone import zone_enter
    return await zone_enter(Path(__file__).absolute().parents[1] / "lobby.jpeg")


@visible
@exit
async def on_exit():
    """
    Runs when leaving chat
    """
    from dynamic_functions.Chat.zone import zone_exit
    return await zone_exit()


@visible
async def get_status(command_prefix: str, current_folder: str):
    """
    Demo walkthrough status for the current user: every step with how far
    they are, what the last thing they ran should show, and exactly what to
    do next. Chat runs this before every bot round, passing the prefix a
    Multix command takes where they type and the folder they are in. It only
    reads; Chat tells the bot who the user is and how they type.
    """
    logger.info(f"Executing get_status...")
    if command_prefix not in ("", "/"):
        raise ValueError(f"command_prefix must be '' or '/', got {command_prefix!r}")

    walk = await _walk(_caller())
    lines: list[str] = []
    app_folder = _app_folder()
    if current_folder != app_folder:
        lines.append(f"They are not in the Demo folder; before anything else, have them type "
                     f"{command_prefix}cd {app_folder} and wait.")
    lines += _walk_lines(walk, command_prefix)
    lines.append("If the user would rather explore on their own, call list_demo_tools. If they ask "
                 "to start the walkthrough over, call reset_demo_walkthrough.")
    return "\n".join(lines)


@visible
async def record_turn():
    """
    Record every walkthrough action the server saw the current user finish
    (the terminal transcript only reaches back to when the bot arrived, so
    progress must be kept). Chat runs this once a bot's reply has gone out,
    after it records the visit itself.
    """
    username = _caller()
    walk = await _walk(username)
    entry: dict[str, Any] = {"actions": walk.proofs}
    if walk.after is not None:
        entry["after"] = walk.after
    # The status this reply was written from said the walkthrough is complete
    # unless the final command landed while the bot was writing it; then this
    # reply did not say so, and the next status still says it is finished.
    if walk.ended is not None:
        entry["ended"] = walk.ended
    elif walk.current is None:
        entry["ended"] = _now()
    progress_path = _game_path(PROGRESS_FILE)
    progress = _read_json(progress_path)
    progress[username] = entry
    _write_json(progress_path, progress)
    return f"Recorded {len(walk.proofs)} finished walkthrough action(s) for {username}"


@visible
async def reset_walkthrough():
    """
    Start the current user's Demo walkthrough over from step 1: forget what
    they finished, and count only commands they type from now on.
    """
    username = _caller()
    transcript = await _transcript()
    progress_path = _game_path(PROGRESS_FILE)
    progress = _read_json(progress_path)
    progress[username] = {"actions": {}, "after": max((event["id"] for event in transcript), default=0)}
    _write_json(progress_path, progress)
    return f"Reset {username}'s Demo walkthrough to step 1"


def _app_folder() -> str:
    """The Demo folder as a /user/remote/... path, as the user's shell names it"""
    script_folder = atlantis.get_script_folder()
    if not script_folder:
        raise RuntimeError("No script folder; this must run as a dynamic function")
    return posixpath.dirname(script_folder)


# ---------------------------------------------------------------------------
# The walkthrough
# ---------------------------------------------------------------------------
#
# Every action is one command the user types in their terminal, so the server
# sees for itself when it ran: Node's /transcript get holds each command typed
# there and names, on every error, the typedCommandId it came from. Nothing
# waits on the bot to report progress. What a command should show is told to
# the bot only once it has run, so it cannot describe a result ahead of time.

@dataclass(frozen=True)
class Action:
    key: str
    # The forms that count, {sh} marking where a Multix command takes its
    # prefix; the first is the one the bot shows
    commands: tuple[str, ...]
    # What the bot tells them before they type it
    intro: str
    # What it should show, told only once it has run (one or more sentences)
    shows: str
    # A file they must save after the action before this one, then run the
    # command after saving
    saved: Optional[Path] = None
    # (tool, args, value): the server runs the tool and it must return value
    returns: Optional[tuple[str, dict, Any]] = None


@dataclass(frozen=True)
class Step:
    name: str
    actions: tuple[Action, ...]


# Ported from the parked MCP onboarding (atlantis_nodejs resources/mcp/onboarding-demo.md).
_SELECT_SORTED = "select * from [@myTableWithImages] where name like 'M%' order by name desc"
DEMO_STEPS = (
    Step("list the Demo tools", (
        Action("ls", ("{sh}ls",),
               "Tell them {sh}ls lists the functions in the folder they are in, the Demo folder.",
               "the Demo functions, each with its description."),
    )),
    Step("run foo", (
        Action("foo", ("@foo(3,4)",),
               "Ask whether they would like a quick introduction to the Multix shell, and wait. If "
               "yes, explain in a few plain sentences: Multix is where you type commands to explore "
               "and run tools, and its folders hold functions rather than ordinary files; {sh}pwd "
               "shows the current folder and {sh}ls lists its functions. Then tell them @foo(3,4) "
               "calls the foo function with 3 and 4.",
               "7: foo adds its two numbers."),
    )),
    Step("read foo's source", (
        Action("source", ("{sh}cat foo",),
               "Tell them {sh}cat shows the source code of a function, here foo, which they just ran.",
               f"the source code of foo. Explain that this is the same code as the file {FOO_PATH}, "
               "which they can open and edit in VS Code, and that the function's docstring is the "
               "description {sh}ls shows for foo, \"Add two integers\"."),
    )),
    Step("edit foo", (
        Action("edit", ("@foo(3,4)",),
               f"Ask them to edit foo in {FOO_PATH} themselves so it returns x + y + 100, and save "
               "the file. The server reloads functions when a file is saved, so no restart is "
               "needed. Do not make the edit for them. Then have them run foo again.",
               "107: their edit is live without a restart.",
               saved=FOO_PATH, returns=("@$Demo/foo", {"x": 3, "y": 4}, 107)),
    )),
    Step("see images and video", (
        Action("image", ("@myImage",),
               "Tell them myImage shows how functions return images.",
               "an image."),
        Action("video", ("@myVideo",),
               "Tell them myVideo shows how functions return video.",
               "a video playing."),
    )),
    # SQL over embedded commands: a bracket must hold a tool call ([@myTable]); a
    # bare [myTable] fails with "Command or tool not recognized: mytable". Tested
    # 2026-11-23 through Lobster: where/like, order by, (select ...)[0] and a join
    # of two brackets all work. Untested: brackets holding tools with arguments,
    # Multix commands, or nested selects; check those by hand before adding them.
    Step("see tables", (
        Action("table", ("@myTable",),
               "Tell them myTable returns a list of anime heroines.",
               "a table of anime heroines: Atlantis shows tabular data like this as a table."),
        Action("table_source", ("{sh}cat myTable",),
               "Have them look at the source of myTable.",
               "that myTable just returns a list of rows, one dictionary per row with the same keys, "
               "and Atlantis renders that as a table."),
        Action("table_images", ("@myTableWithImages",),
               "Tell them myTableWithImages returns the same heroines with their creators and a "
               "thumbnail of each.",
               "a thumbnail picture in each row: a column holding an image data URI shows as a picture."),
        Action("select", ("{sh}select * from [@myTable] where name like 'M%'",),
               "Tell them any table can be queried with SQL: wrapping a command in square brackets "
               "turns its output into a table to select from. This query keeps only the heroines "
               "whose names start with M.",
               "only the heroines whose names start with M."),
        Action("select_sorted", ("{sh}" + _SELECT_SORTED,),
               "Tell them this runs the same filter on the table with thumbnails, sorted by name in "
               "reverse.",
               "the heroines whose names start with M, with thumbnails, sorted by name in reverse."),
        Action("first_row", ("{sh}(" + _SELECT_SORTED + ")[0]",),
               "Tell them wrapping that query in parentheses and indexing it pulls out just the "
               "first row.",
               "a single record, which shows as JSON instead of a table, so its image field is the "
               "raw base64 data URI rather than a picture."),
        Action("first_image", ("{sh}(" + _SELECT_SORTED + ")[0].image",),
               "Tell them adding .image pulls just that row's image.",
               "the image itself."),
        Action("join", ("{sh}select t.name, t.series, i.creator from [@myTable] AS t join "
                        "[@myTableWithImages] AS i on t.name = i.name",),
               "Tell them tables can be joined, like in any database: naming each bracketed command "
               "with AS lets the query match rows across them.",
               "each heroine's series from the first table next to her creator from the second."),
    )),
    Step("say hello", (
        Action("hello", ("@hello",),
               "Tell them they are past the halfway point of the demo, and that hello greets whoever "
               "calls it.",
               "a greeting with their Atlantis username. Point out that a function knows exactly "
               "who is calling it."),
    )),
    Step("open the code editor", (
        Action("editor", ("{sh}edit hello",),
               "Tell them {sh}edit opens the built-in Multix code editor on a function: unlike "
               "{sh}cat, which only shows the source, it lets them change the function and save it.",
               "the code editor open on hello. Point out that hello uses the atlantis package, "
               "which provides many commands such as get_caller()."),
    )),
    Step("run the editor in the background", (
        Action("background", ("{sh}edit hello &",),
               "Have them close the editor; closing does not save. Explain that while the editor "
               "was open, the terminal was waiting on it; like Unix, a trailing & runs a command in "
               "the background, so the editor opens and the terminal stays free.",
               "the editor open while the terminal is still free to use."),
    )),
    Step("run commands from the editor", (
        Action("editor_ls", ("{sh}ls",),
               "In the editor, have them make a fresh blank line above @visible and press "
               "Cmd+Right Arrow (Ctrl+Right Arrow on Windows or Linux) on it, which inserts a # % "
               "command comment; hello.py may already have example # % comments, which run the same "
               "way. They type the command after the % and press Cmd+Right Arrow again to run it "
               "from the editor.",
               "the Demo functions with their descriptions, in the terminal."),
        Action("editor_hello", ("hello", "@hello"),
               "Have them change the command in that # % comment and run it from the editor the "
               "same way.",
               "the greeting, in the terminal."),
        Action("docstring", ("{sh}ls hello",),
               "Have them change the docstring under def hello() to something new and save with "
               "Cmd+S (Ctrl+S on Windows or Linux): the docstring is the description ls shows. Then "
               "have them change the # % command and run it from the editor.",
               "hello with the new description they wrote.",
               saved=HELLO_PATH),
        Action("clear", ("{sh}clear",),
               "Have them close the editor.",
               "a cleared terminal."),
    )),
)
_ACTION_KEYS = [action.key for step in DEMO_STEPS for action in step.actions]
if len(set(_ACTION_KEYS)) != len(_ACTION_KEYS):
    raise ValueError("DEMO_STEPS action keys must be unique")


@dataclass(frozen=True)
class Walk:
    # Proof of every action finished, by key: the command event that did it
    proofs: dict[str, dict]
    # (step index, action index) of the first unfinished action; None when done
    current: Optional[tuple[int, int]]
    # The last action finished, None before the first
    last: Optional[Action]
    # Why what they ran for the current action does not count yet, with
    # {command} where its command goes
    problem: Optional[str]
    # Only commands after this transcript event id count (set by a reset)
    after: Optional[int]
    # When the bot was first told the walkthrough is complete
    ended: Optional[str]


async def _walk(username: str) -> Walk:
    """Where the user is in the walkthrough: the actions recorded in this game,
    then whatever the commands in their terminal prove since. Each action must
    come after the one before it."""
    entry = _read_json(_game_path(PROGRESS_FILE)).get(username, {"actions": {}})
    if not isinstance(entry, dict) or "actions" not in entry or not set(entry) <= {"actions", "after", "ended"}:
        raise ValueError(f"{PROGRESS_FILE} holds {entry!r} for {username}, which is not a walkthrough; "
                         "it predates the current walkthrough, so start a new game")
    stored = entry["actions"]
    for key, proof in stored.items():
        if key not in _ACTION_KEYS or not isinstance(proof, dict) or set(proof) != {"event", "at"}:
            raise ValueError(f"{PROGRESS_FILE} holds {key!r}: {proof!r}, which is not a walkthrough "
                             "action proof; it predates the current walkthrough, so start a new game")
    after = entry.get("after")
    ended = entry.get("ended")
    ran = [event for event in await _ran_commands() if after is None or event["id"] > after]

    proofs: dict[str, dict] = {}
    before: Optional[dict] = None
    last: Optional[Action] = None
    for step_index, step in enumerate(DEMO_STEPS):
        for action_index, action in enumerate(step.actions):
            proof = stored.get(action.key)
            problem = None
            if proof is None:
                used = {proof["event"] for proof in proofs.values()}
                proof, problem = await _attempt(action, before, [event for event in ran if event["id"] not in used])
            if proof is None:
                return Walk(proofs, (step_index, action_index), last, problem, after, ended)
            proofs[action.key] = proof
            before, last = proof, action
    return Walk(proofs, None, last, None, after, ended)


async def _transcript() -> list[dict]:
    """Every event in the caller's terminal"""
    transcript = await atlantis.client_command(f"/transcript get {json.dumps(_caller_terminal())}")
    if not isinstance(transcript, list):
        raise TypeError(f"/transcript get returned {type(transcript).__name__}, expected a list")
    return transcript


async def _ran_commands() -> list[dict]:
    """Commands typed in the caller's terminal that raised no error, oldest
    first. Node names on each error the typedCommandId it came from."""
    transcript = await _transcript()
    failed = {event["typedCommandId"] for event in transcript
              if event.get("type") == "error" and "typedCommandId" in event}
    return sorted((event for event in transcript
                   if event.get("type") == "command" and event["id"] not in failed),
                  key=lambda event: event["id"])


async def _attempt(action: Action, before: Optional[dict], ran: list[dict]) -> tuple[Optional[dict], Optional[str]]:
    """(proof, None) when a command they ran after the action before this one
    finishes it, else (None, why not yet), the reason None when they have not
    run it at all. `ran` holds no command that already proved another action."""
    forms = {_command_key(command.replace("{sh}", "")) for command in action.commands}
    matching = [event for event in ran if _command_key(event["content"]) in forms]
    typed = [event for event in matching if before is None or event["id"] > before["event"]]
    if not typed:
        if matching:
            return None, ("They ran {command} earlier, before the step before it was done, so it "
                          "does not count: have them run it again now.")
        return None, None

    if action.saved:
        saved_at = datetime.fromtimestamp(action.saved.stat().st_mtime, timezone.utc)
        if before is not None and saved_at <= datetime.fromisoformat(before["at"]):
            return None, (f"They ran {{command}}, but have not saved {action.saved.name} since the "
                          "step before, so the change is not made yet.")
        typed = [event for event in typed if datetime.fromisoformat(event["created_at"]) > saved_at]
        if not typed:
            return None, f"They saved {action.saved.name}, but have not run {{command}} since."

    if action.returns:
        tool, args, value = action.returns
        result = await atlantis.client_command(tool, args)
        if result != value:
            return None, (f"They ran {{command}}, but it returns {result!r} right now, not {value!r}: "
                          f"the change in {action.saved.name if action.saved else tool} is not right yet.")

    return {"event": typed[0]["id"], "at": typed[0]["created_at"]}, None


def _command_key(text: str) -> str:
    """How typed commands compare: without the / the chat window needs, in any
    case or spacing, and without () on a tool call with no arguments"""
    key = re.sub(r"\s+", "", text).lower().removeprefix("/")
    return key.removesuffix("()") if key.startswith("@") else key


def _walk_lines(walk: Walk, prefix: str) -> list[str]:
    """The walkthrough part of the status"""
    def shown(action: Action) -> str:
        return f"`{action.commands[0].replace('{sh}', prefix)}`"

    lines = [f"The Demo walkthrough has {len(DEMO_STEPS)} steps:"]
    for number, step in enumerate(DEMO_STEPS, 1):
        done = sum(action.key in walk.proofs for action in step.actions)
        if done == len(step.actions):
            label = "done"
        elif walk.current is not None and walk.current[0] == number - 1:
            label = "NEXT" if len(step.actions) == 1 else f"NEXT, {done} of {len(step.actions)} done"
        else:
            label = "not started"
        lines.append(f"{number}. [{label}] {step.name}")

    other_tools = ("Offer to find them any other tool: search covers every folder on their search "
                   f"path, which they can see with {prefix}path show.")
    if walk.ended is not None:
        lines.append(f"They finished the walkthrough at {walk.ended}, and you have told them so. "
                     f"Help them with whatever they ask. {other_tools}")
        return lines
    if walk.current is None:
        if walk.last is None:
            raise RuntimeError("The walkthrough is complete but no action was finished; DEMO_STEPS is empty")
        lines.append(f"They just ran {shown(walk.last)}, the last step, which shows "
                     f"{walk.last.shows.replace('{sh}', prefix)} The walkthrough is complete: tell "
                     f"them so. {other_tools}")
        return lines

    if walk.last is not None:
        lines.append(f"The last thing they ran was {shown(walk.last)}; it should show "
                     f"{walk.last.shows.replace('{sh}', prefix)} If they have not said yet whether "
                     "they see that, ask, and wait.")

    step_index, action_index = walk.current
    action = DEMO_STEPS[step_index].actions[action_index]
    lines.append(f"Next (step {step_index + 1}): {action.intro.replace('{sh}', prefix)} Then have "
                 f"them type {shown(action)}, and end your turn. The server sees when it has run and "
                 "only then tells you what it shows, so do not say beforehand.")
    if walk.problem:
        lines.append(walk.problem.replace("{command}", shown(action)))
    return lines


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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
