import asyncio
import contextvars
import logging
from typing import Callable, Optional

from atlantis import _user_var

logger = logging.getLogger("mcp_server")

# --- Game Ticks ---
# Tick support is explicit: callers must pass a game_key into every tick API.
# No implicit contextvar fallback — game_key is MCP-local state, not cloud-supplied.
#
# Shape: {game_key: {
#     "ctx": contextvars.Context,   — snapshot for running callbacks in game context
#     "caller": str,                — user who registered the tick
#     "tick_callback": Callable,    — async fn to call each tick
#     "tick_busy": bool,            — True while a tick is in-flight
# }}
_active_game_ticks: dict = {}

# Single global tick loop — one asyncio.Task, iterates all registered game ticks.
_tick_task: Optional[asyncio.Task] = None
_tick_interval: float = 1.0


def _require_game_key(game_key: str) -> str:
    """Validate that game_key is provided."""
    if not game_key:
        raise RuntimeError("A valid game_key is required")
    return game_key


def get_active_game_ticks() -> dict:
    """Return the live server-wide dict of registered game ticks."""
    return _active_game_ticks


def set_tick_interval(seconds: float) -> None:
    """Change the global tick interval. Takes effect on the next sleep cycle."""
    global _tick_interval
    _tick_interval = max(0.1, seconds)
    logger.info(f"tick interval set to {_tick_interval}s")


def deactivate_game_tick(game_key: str) -> bool:
    """Remove a game from the tick registry."""
    if game_key and game_key in _active_game_ticks:
        del _active_game_ticks[game_key]
        logger.info(f"deactivate_game_tick: removed game={game_key} (remaining: {len(_active_game_ticks)})")
        if not _active_game_ticks:
            _stop_tick_loop()
        return True
    return False


def register_tick(game_key: str, callback: Callable) -> None:
    """Register a tick callback for the given game_key."""
    gid = _require_game_key(game_key)
    _active_game_ticks[gid] = {
        "ctx": contextvars.copy_context(),
        "caller": _user_var.get() or "",
        "tick_callback": callback,
        "tick_busy": False,
    }
    logger.info(f"register_tick: callback registered for game={gid}")
    _ensure_tick_loop()


def unregister_tick(game_key: str) -> None:
    """Remove the tick callback for a game."""
    deactivate_game_tick(game_key)


async def _run_one_tick(gid: str, entry: dict) -> None:
    """Run a single game's tick callback inside its saved context."""
    entry["tick_busy"] = True
    try:
        cb = entry["tick_callback"]
        ctx = entry["ctx"]
        coro = ctx.run(cb)
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"tick error for game {gid}: {e}")
    finally:
        if gid in _active_game_ticks:
            entry["tick_busy"] = False


async def _tick_loop() -> None:
    """Single global tick loop for explicitly registered game ticks."""
    logger.info("tick loop started")
    try:
        while _active_game_ticks:
            tasks = []
            for gid, entry in list(_active_game_ticks.items()):
                if not entry.get("tick_busy"):
                    tasks.append(_run_one_tick(gid, entry))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(_tick_interval)
    except asyncio.CancelledError:
        logger.info("tick loop cancelled")
    except Exception as e:
        logger.error(f"tick loop crashed: {e}")
    finally:
        global _tick_task
        _tick_task = None
        logger.info("tick loop stopped")


def _ensure_tick_loop() -> None:
    """Start the global tick loop if it isn't already running."""
    global _tick_task
    if _tick_task is not None and not _tick_task.done():
        game_keys = list(_active_game_ticks.keys())
        logger.info(f"tick loop already running ({len(game_keys)} game tick(s): {game_keys})")
        return
    try:
        loop = asyncio.get_running_loop()
        _tick_task = loop.create_task(_tick_loop(), name="atlantis_tick_loop")
        logger.info("tick loop task created")
    except RuntimeError as exc:
        raise RuntimeError("Cannot start tick loop without a running event loop") from exc


def _stop_tick_loop() -> None:
    """Cancel the global tick loop."""
    global _tick_task
    if _tick_task is not None and not _tick_task.done():
        _tick_task.cancel()
        logger.info("tick loop cancel requested")
    _tick_task = None
