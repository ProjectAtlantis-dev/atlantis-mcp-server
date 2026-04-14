"""FlowCentralLobby arrival check-in tools.

MCP-visible tools the receptionist bot uses to walk a new guest
through the FlowCentral front-desk procedure.

Guest data lives in Data/players/{username}/.
"""

import logging
from datetime import datetime

from dynamic_functions.Data.main import (
    get_guest,
    get_interaction_info as _get_interaction_info,
    is_cleared,
    list_all_guests,
)

logger = logging.getLogger("mcp_server")

LOCATION = "FlowCentralLobby"


# =========================================================================
# Helpers (importable by other modules)
# =========================================================================

def get_interaction_info(username: str) -> tuple[int, str]:
    return _get_interaction_info(username)


def is_checkin_complete(username: str) -> bool:
    return is_cleared(username)


async def build_checkin_injections(
    caller: str,
    guest: dict | None,
    interaction: dict | None = None,
) -> list[dict]:
    """Build runtime procedure prompts for FlowCentralLobby check-in."""
    if guest and guest.get("cleared"):
        last_interaction_at = (interaction or {}).get("last_interaction_at") or ""
        if not last_interaction_at:
            return []
        try:
            elapsed = datetime.now() - datetime.fromisoformat(last_interaction_at)
        except (ValueError, TypeError):
            return []
        if elapsed.total_seconds() <= 3600:
            return []
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        return [{'role': 'user', 'content': [{'type': 'text', 'text':
            f"[Some time has passed since your last interaction. The current date and time is now {now_str}.]"
        }]}]

    prior_interaction_count = int((interaction or {}).get("prior_interaction_count") or 0)
    is_new = prior_interaction_count <= 0

    if is_new:
        from dynamic_functions.Game.Content.FlowCentralLobby.overview import get_overview
        overview = await get_overview()
        text = (
            "[NEW GUEST] This guest has not interacted with FlowCentral before. "
            "Welcome them warmly and walk them through the platform overview "
            "below in your own words — keep it conversational, hit the "
            "highlights, don't just dump the whole thing. After the overview, "
            "suggest they try Page Speed — it's the tool that's live right "
            "now. Ask if they have a website URL they'd like to test.\n\n"
            f"{overview}"
        )
    else:
        text = (
            "[RETURNING GUEST] This guest has interacted with FlowCentral before "
            f"({prior_interaction_count} time(s)) but hasn't completed check-in yet. "
            "Welcome them back casually — no need for the full overview. "
            "Ask what they'd like to do today."
        )
    return [{'role': 'system', 'content': [{'type': 'text', 'text': text}]}]


# =========================================================================
# Visible tools
# =========================================================================

@visible
async def list_guests():
    """Returns a list of all known guests."""
    logger.info("FlowCentralLobby list_guests called")
    guests = list_all_guests()
    if not guests:
        return "No guests on record yet."
    return guests


@visible
async def guest_info(username: str):
    """
    Look up all stored data for a guest.

    Args:
        username: The guest's name/identifier
    """
    logger.info(f"FlowCentralLobby guest_info called for: {username}")
    guest = get_guest(username)
    if not guest:
        return f"No record found for {username}. They may be a brand new guest."
    return guest


@visible
async def get_guest_checklist():
    """
    Returns the front-desk check-in checklist for a NEW FlowCentral guest.
    Call this ONCE, then pass the returned array to the todo tool to load it.
    After loading, use todo(merge=true) to update each step's status as you go.
    Do NOT call this again — just use the todo tool to track progress.
    """
    logger.info("FlowCentralLobby get_guest_checklist called")
    return [
        {"id": "overview", "status": "pending", "content": "Ssearch for 'get_overview' on your console and call it to get the platform overview, then walk the guest through what FlowCentral has to offer in your own words. Keep it conversational — hit the highlights, don't just dump the whole thing."},
        {"id": "suggest", "status": "pending", "content": "Suggest they try Page Speed — it's the tool that's live right now. Ask if they have a website URL they'd like to test."},
    ]
