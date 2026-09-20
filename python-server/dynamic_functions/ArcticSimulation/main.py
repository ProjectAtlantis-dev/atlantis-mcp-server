@visible
def index() -> dict:
    """Isolated simulation scenarios; not the canonical bank/world authority."""
    return {"application": "ArcticSimulation", "scope": "isolated scenarios",
            "startup": "Set ATLANTIS_SIM_DB_PATH, then call start with a free loopback port",
            "infrastructure": "ArcticSimulation.Infrastructure",
            "tick": "server-owned fixed timestep; dynamic functions issue commands only"}
