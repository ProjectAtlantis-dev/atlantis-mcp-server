import asyncio
import contextvars
import importlib
import atlantis
import logging
import sys

from dynamic_functions.Callback.tick_set import get_tick_tool_path, set_tick_enabled

logger = logging.getLogger("mcp_server")

# server_shared key for the single asyncio.Task running the loop
_LOOP_KEY = "tick_loop"

_TICK_INTERVAL = 1.0


def _resolve_tick_fn():
    """Resolve the currently configured tick function via its toolPath.

    toolPath is a dotted path like 'Callback.tick_callback' — the last segment
    is the function name; the full path is the module under dynamic_functions/.
    Re-imported every call so tick_set takes effect without restart.

    Use a fresh import instead of importlib.reload(): modules such as
    Callback.tick_callback define both the decorator name and the function name
    as ``tick``. Reload re-executes against the existing module globals, so the
    previous function object can shadow the builtins decorator and eventually
    collapse into a coroutine object on repeated reloads.
    """
    tool_path = get_tick_tool_path()
    parts = tool_path.split(".")
    if not parts or not parts[-1]:
        raise ValueError(f"invalid tick toolPath: {tool_path!r}")
    func_name = parts[-1]
    module_name = "dynamic_functions." + tool_path
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


async def _tick_loop():
    """Single global loop: every interval, fan out the configured tick function
    across all active games, each run under its originating game's
    contextvars.Context."""
    logger.info("tick loop started")
    try:
        while True:
            await asyncio.sleep(_TICK_INTERVAL)

            games = atlantis.get_active_games()
            if not games:
                logger.info("tick loop: no active games, exiting")
                atlantis.server_shared.remove(_LOOP_KEY)
                return

            try:
                tick_fn = _resolve_tick_fn()
            except Exception as e:
                logger.exception(f"tick fan-out: failed to resolve tick function: {e}")
                continue

            # Snapshot to avoid mutation-during-iteration
            for game_id, entry in list(games.items()):
                ctx: contextvars.Context = entry["ctx"]
                try:
                    # Run tick() under the game's original context so atlantis.*
                    # calls route to the correct session/caller.
                    asyncio.create_task(
                        tick_fn(game_id),
                        name=f"tick:{game_id}",
                        context=ctx,
                    )
                except Exception as e:
                    logger.exception(f"tick fan-out failed for game={game_id}: {e}")
    except asyncio.CancelledError:
        logger.info("tick loop cancelled")
        raise


def ensure_tick_loop():
    """Start the tick loop if it isn't already running. Idempotent."""
    loop_task = atlantis.server_shared.get(_LOOP_KEY)
    if loop_task is None or loop_task.done():
        loop_task = asyncio.create_task(_tick_loop(), name="tick_loop")
        atlantis.server_shared.set(_LOOP_KEY, loop_task)
    return loop_task


@visible
async def tick_enable() -> str:
    """Enable ticking and start the loop if any games are active.

    Game registration happens server-side; ticking remains opt-in. If there are
    no active games right now, the enabled flag is persisted and the next game
    activation will start the loop.
    """
    set_tick_enabled(True)
    games = atlantis.get_active_games()
    if games:
        ensure_tick_loop()
        return f"tick enabled; loop running (active games={len(games)})"
    return "tick enabled; loop idle until a game becomes active"
