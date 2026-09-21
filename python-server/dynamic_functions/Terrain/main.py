import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@public
async def first_menu():
    """Terrain tools"""
    return None


@visible
@index
async def index():
    """Live Terrain controls. Start with instructions for AI/Lobster workflows, then discover current per-object actions."""
    return {"module": "Terrain", "start": "Terrain/instructions",
            "controlGuides": ["vehicles", "objects", "defense", "infrastructure", "demo"],
            "objectActions": "Terrain/Objects/functions"}
