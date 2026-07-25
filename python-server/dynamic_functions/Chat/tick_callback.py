"""Game tick callback — fires on engine cadence to let bots initiate action.

Symmetric to chat_callback but without a transcript trigger. With zero AI
slots in-world there is nothing to initiate; that's the state of an empty
game, not an error.
"""

import atlantis
import logging

from .game import _game_is_running

logger = logging.getLogger("dynamic_function")

_BUSY_KEY = "chat_busy"


@tick
async def tick_callback(game_key: str):
    """Game tick: let an AI bot initiate action if any are in-world."""
    if not atlantis.get_session_key():
        logger.warning("tick_callback fired without session context, skipping")
        return

    if not _game_is_running(game_key):
        logger.debug(f"tick_callback game {game_key!r} is stopped, skipping")
        return

    if atlantis.session_shared.get(_BUSY_KEY):
        logger.debug("tick_callback: chat busy, skipping")
        return

    raise NotImplementedError("tick_callback: slot system removed — needs reimplementation")
