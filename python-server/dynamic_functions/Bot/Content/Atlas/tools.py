"""Atlas tool policy — post-filtering for search/dir results."""

LOCATION = "AtlasLobby"


def filter_results(tools):
    """Filter search/dir results. Remove tools that don't belong to Atlas."""
    return [t for t in tools if "KittyLobby" not in t.get("searchTerm", "")]
