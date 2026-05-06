"""Home app tools"""

import atlantis
from typing import List, Dict, Any

from dynamic_functions.Home.chat_common import logger, get_base_tools


def _require_game():
    from dynamic_functions.Home.game import require_game_key
    require_game_key()


# Preserve lazy imports from other modules
def _get_current_game() -> str:
    from dynamic_functions.Home.game import _get_current_game as _impl
    return _impl()


@visible
async def index():
    """Multix CLI readme"""
    pass


@visible
async def show_tools() -> List[Dict[str, Any]]:
    """Show the runtime tool inventory"""
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
