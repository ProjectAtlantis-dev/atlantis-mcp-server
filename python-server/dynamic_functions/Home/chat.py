"""Chat entry point — determine room occupants and next speaker."""

import atlantis
import logging

from dynamic_functions.Home.location import get_player_position, get_players_at
from dynamic_functions.Home.location import position_query

logger = logging.getLogger("mcp_server")


@chat
async def chat():
    """Main chat callback. Figures out who's in the room and who speaks next."""
    caller = atlantis.get_caller()
    if not caller:
        logger.warning("Chat fired without a caller identity")
        return

    # Where is the caller?
    location = get_player_position(caller)
    if not location:
        await atlantis.client_log(f"📍 {caller} has no position — nowhere to chat")
        return

    # Who else is here?
    occupants = position_query(location)
    if not occupants:
        await atlantis.client_log(f"📍 {caller} is alone in {location}")
        return

    # Build a list of everyone in the room (with display names)
    names = []
    bots = []
    for ch in occupants:
        display = ch.get("displayName", ch["sid"])
        names.append(display)
        if ch.get("isBot") and ch["sid"] != caller:
            bots.append(ch)

    await atlantis.client_log(
        f"🏠 Room [{location}]: {', '.join(names)}"
    )

    # Next to speak: first bot in the room that isn't the caller
    if bots:
        next_up = bots[0]
        await atlantis.client_log(
            f"🎤 Next to speak: {next_up.get('displayName', next_up['sid'])}"
        )
    else:
        await atlantis.client_log("🎤 No bots present — waiting for player input")
