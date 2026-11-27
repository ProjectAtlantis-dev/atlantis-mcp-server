import logging

import atlantis

logger = logging.getLogger("dynamic_function")

# Bot tools run from their own folder, so a bare name would resolve here, not
# where the user is: the function must be named from an anchor
_ANCHORS = ("$", "%", "~", "/")


@visible
async def get_source(function: str) -> str:
    """The Python source of a function, as the user's own `cat` shows it. Name it by an anchored Multix path, e.g. `$Demo/foo` for foo in the Demo folder. Use it to read what a function really does, or to check an edit the user made, instead of guessing."""
    logger.info(f"Executing get_source for {function!r}...")
    if not function.startswith(_ANCHORS):
        raise ValueError(f"get_source needs an anchored path such as $Demo/foo, got {function!r}")
    # /cat goes through _function_get, so the caller's access rules apply
    source = await atlantis.client_command("/cat", {"searchTerm": function})
    if not isinstance(source, str):
        raise TypeError(f"/cat {function} returned {type(source).__name__}, expected a string")
    return source
