import logging
from typing import Dict, Any

logger = logging.getLogger("mcp_server")


def on_activity(analysis: Dict[str, Any], caller_sid: str) -> Dict[str, Any]:
    """Lobby-specific activity logic. Decides what to do based on who's in the room.

    analysis: output from analyze_participants()
    caller_sid: who just spoke

    Returns action dict for the runtime to act on.
    """
    participants = analysis.get('participants', {})
    others = {sid: info for sid, info in participants.items() if sid != caller_sid}

    if not others:
        # Visitor is alone in the lobby — spawn Kitty
        return {'action': 'spawn', 'bot': 'kitty'}

    # Someone else is already here
    return {'action': 'continue'}
