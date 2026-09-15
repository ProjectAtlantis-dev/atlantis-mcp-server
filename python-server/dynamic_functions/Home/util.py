"""General utility functions."""

from typing import Any, Dict, List

import atlantis


@public
async def get_unused() -> List[Dict[str, Any]]:
    """Return all functions without callers."""
    return atlantis.get_uncalled_dynamic_functions()


@public
async def get_single() -> List[Dict[str, Any]]:
    """Return all functions with exactly one caller."""
    return atlantis.get_single_called_dynamic_functions()
