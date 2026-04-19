from typing import Dict, Any


def on_activity(analysis: Dict[str, Any], caller_sid: str) -> Dict[str, Any]:
    participants = analysis.get("participants", {})
    others = {sid: info for sid, info in participants.items() if sid != caller_sid}

    if not others:
        return {"action": "spawn", "bot": "kitty"}

    return {"action": "continue"}
