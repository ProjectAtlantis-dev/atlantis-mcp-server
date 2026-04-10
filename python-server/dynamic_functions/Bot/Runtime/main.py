@visible
async def index():
    """Bot mechanics"""
    pass

@visible
async def list():
    """Bot pool is temporarily disabled."""
    return {"status": "disabled", "message": "Bot pool is disabled pending game-management rework."}


@visible
async def spawn(name: str):
    """Bot pool is temporarily disabled."""
    return {"status": "disabled", "message": "Bot pool is disabled pending game-management rework.", "requested": name}


@visible
async def remove(index: int):
    """Bot pool is temporarily disabled."""
    return {"status": "disabled", "message": "Bot pool is disabled pending game-management rework.", "requested_index": index}
