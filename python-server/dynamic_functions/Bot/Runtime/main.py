@visible
async def index():
    """Bot mechanics"""
    pass


from dynamic_functions.Game.Runtime.common import BOTS, add_bot, remove_bot, list_bots


@visible
async def list():
    """List all bots in the chat pool."""
    return list_bots()


@visible
async def spawn(name: str):
    """Spawn a bot into the chat pool by name (e.g. 'kitty'). Looks up Bot/Content/{Name}/ for system prompt."""
    bot = add_bot(name)
    return {"spawned": bot, "pool_size": len(BOTS)}


@visible
async def remove(index: int):
    """Remove a bot from the chat pool by its index number (see list)."""
    removed = remove_bot(index)
    return {"removed": removed, "pool_size": len(BOTS)}
