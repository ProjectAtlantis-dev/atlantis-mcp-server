import logging

from ..Home.user import user_bg_video

logger = logging.getLogger("dynamic_function")


@visible
async def play_bg_video() -> None:
    """Play attention.mp4 as the user background video."""
    await user_bg_video("attention.mp4")
