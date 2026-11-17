import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@visible
async def myTable() -> list[dict[str, object]]:
    """Return a table of Disney characters"""
    # this does not call client_data to display a nicely formatted table, it simply returns an array and lets Atlantis render w default formatting
    return [
        {"id": 1, "name": "Goofy", "favorite_food": "pizza"},
        {"id": 2, "name": "Donald", "favorite_food": "corn"},
        {"id": 3, "name": "Simba", "favorite_food": "gazelle"},
        {"id": 4, "name": "Stitch", "favorite_food": "coconut cake"},
        {"id": 5, "name": "Ratatouille", "favorite_food": "ratatouille"},
    ]
