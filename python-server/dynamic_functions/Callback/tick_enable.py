import asyncio
import contextvars
import atlantis
import logging

from dynamic_functions.Callback.tick import tick as _tick_fn

logger = logging.getLogger("mcp_server")

# server_shared keys
_LOOP_KEY = "tick_loop"     # the single asyncio.Task running the loop
_GAMES_KEY = "tick_games"   # dict[session_id -> {"ctx": Context, "caller": str}]
_INTERVAL_KEY = "tick_interval"


async def _tick_loop():
    """Single global loop: every interval, fan out tick() across all active games,
    each run under its originating session's contextvars.Context."""
    logger.info("tick loop started")
    try:
        while True:
            interval = atlantis.server_shared.get(_INTERVAL_KEY, 1.0)
            await asyncio.sleep(interval)

            games = atlantis.server_shared.get(_GAMES_KEY) or {}
            if not games:
                logger.info("tick loop: no active games, exiting")
                atlantis.server_shared.remove(_LOOP_KEY)
                return

            # Snapshot to avoid mutation-during-iteration
            for session_id, entry in list(games.items()):
                ctx: contextvars.Context = entry["ctx"]
                try:
                    # Run tick() under the game's original context so atlantis.*
                    # calls route to the correct session/caller.
                    asyncio.create_task(
                        _tick_fn(),
                        name=f"tick:{session_id}",
                        context=ctx,
                    )
                except Exception as e:
                    logger.exception(f"tick fan-out failed for session={session_id}: {e}")
    except asyncio.CancelledError:
        logger.info("tick loop cancelled")
        raise


@visible
async def tick_enable(interval: float = 1.0) -> str:
    """Register the current game session with the global tick loop.

    The loop is a single server-wide asyncio task that iterates over every
    active game on each interval. If no loop is running it will be started.
    """
    session_id = atlantis.get_session_id()
    if not session_id:
        raise RuntimeError("tick_enable requires a session context")

    caller = atlantis.get_caller() or ""

    games = atlantis.server_shared.get(_GAMES_KEY)
    if games is None:
        games = {}
        atlantis.server_shared.set(_GAMES_KEY, games)

    # Capture the current contextvars so the loop can invoke tick() as if it
    # were called from this game's original request.
    games[session_id] = {
        "ctx": contextvars.copy_context(),
        "caller": caller,
    }
    atlantis.server_shared.set(_INTERVAL_KEY, interval)
    logger.info(f"tick_enable: session={session_id} caller={caller} (active games: {len(games)})")

    # Start the loop if it isn't already running
    loop_task = atlantis.server_shared.get(_LOOP_KEY)
    if loop_task is None or loop_task.done():
        loop_task = asyncio.create_task(_tick_loop(), name="tick_loop")
        atlantis.server_shared.set(_LOOP_KEY, loop_task)

    return f"tick enabled for session {session_id} (interval={interval}s, active games={len(games)})"
