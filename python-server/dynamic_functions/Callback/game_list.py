import atlantis


@visible
async def game_list() -> list[dict]:
    """Return the currently active games registered server-side.

    Each entry: {"game_id": str, "caller": str}. Useful for debugging the
    auto-registration / tick fan-out path.
    """
    games = atlantis.get_active_games()
    return [
        {"game_id": gid, "caller": entry.get("caller", "")}
        for gid, entry in games.items()
    ]
