"""Home app entry point — index and tool inventory."""

import atlantis
from typing import List, Dict, Any

from dynamic_functions.Home.bot_common import logger, get_base_tools


def _require_game():
    if not atlantis.get_game_id():
        raise RuntimeError("No active game — this tool requires a running game session.")


# Re-export so existing lazy imports from other modules keep working.
def _get_current_game() -> str:
    from dynamic_functions.Home.game import _get_current_game as _impl
    return _impl()


@visible
async def index():
    """Multix CLI readme"""
    pass


@visible
async def show_tools() -> List[Dict[str, Any]]:
    """Show the current runtime tool inventory for this session."""
    tools, lookup = get_base_tools()
    simple: List[Dict[str, Any]] = []
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "any")
            if isinstance(ptype, list):
                ptype = ",".join(ptype)
            parts.append(f"{pname}:{ptype}")
        sig = ", ".join(parts)
        simple.append({
            "name": f"{fn['name']} ({sig})",
            "description": fn.get("description", ""),
        })
    logger.info(f"show_tools: {len(simple)} tools")
    return simple
