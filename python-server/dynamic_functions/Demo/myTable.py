import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@visible
async def myTable() -> list[dict[str, object]]:
    """Return a table of anime heroines"""
    # this does not call client_data to display a nicely formatted table, it simply returns an array and lets Atlantis render w default formatting
    return [
        {"id": 1, "name": "Motoko Kusanagi", "series": "Ghost in the Shell", "franchise_started": 1989, "genre": "cyberpunk"},
        {"id": 2, "name": "Misato Katsuragi", "series": "Neon Genesis Evangelion", "franchise_started": 1995, "genre": "mecha"},
        {"id": 3, "name": "Faye Valentine", "series": "Cowboy Bebop", "franchise_started": 1998, "genre": "space western"},
        {"id": 4, "name": "Nico Robin", "series": "One Piece", "franchise_started": 1997, "genre": "pirate adventure"},
        {"id": 5, "name": "Fujiko Mine", "series": "Lupin III", "franchise_started": 1967, "genre": "heist"},
        {"id": 6, "name": "Lynae", "series": "Wuthering Waves", "franchise_started": 2024, "genre": "open-world action RPG"},
        {"id": 7, "name": "Asuka Langley Soryu", "series": "Neon Genesis Evangelion", "franchise_started": 1995, "genre": "mecha"},
    ]
