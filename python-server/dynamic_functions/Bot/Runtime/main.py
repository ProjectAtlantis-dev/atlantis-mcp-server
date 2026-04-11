from typing import List, Dict, Any

from dynamic_functions.Bot.Runtime.common import logger, get_session_tools
from dynamic_functions.Misc.todo import list_tasks as _list_tasks


@visible
async def index():
    """Bot mechanics"""
    pass



@visible
async def show_tools() -> List[Dict[str, Any]]:
    """Show the current runtime tool inventory for this session."""
    tools, lookup = get_session_tools()
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


@visible
async def show_todos():
    """Show the current todo/task list for this session."""
    return await _list_tasks()
