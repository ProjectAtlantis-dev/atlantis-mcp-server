@visible
def index() -> dict:
    """Low-level simulation service controls shared with the Terrain demo. Use Terrain/instructions for the current-world AI workflow; do not reset a commissioned live world."""
    return {"application": "ArcticSimulation", "scope": "MCP-supervised simulation service",
            "terrainGuide": "Terrain/instructions", "defenseGuide": "Terrain/Defense/instructions",
            "tick": "server-owned fixed timestep; dynamic functions issue commands only"}
