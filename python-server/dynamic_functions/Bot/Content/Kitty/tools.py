"""Kitty tool policy — post-filtering for search/dir results."""

LOCATION = "KittyLobby"


def filter_results(tools):
    """Filter search/dir results. Remove tools that don't belong to Kitty."""
    return [t for t in tools if "AtlasLobby" not in t.get("searchTerm", "")]
