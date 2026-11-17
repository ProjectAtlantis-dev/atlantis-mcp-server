"""User/session-scoped tools."""

import atlantis
import logging

from .term import term_bg_video

logger = logging.getLogger("dynamic_function")


USER_DEFAULT_BG_ALIGN = "bottom"


@visible
def _user_default_bg_path() -> str | None:
    """Server --image is optional; returns None when not configured."""
    image_path = atlantis.get_server_info().get("image")
    if not image_path:
        logger.warning("No server --image configured; skipping user default background")
        return None
    return image_path


@public
async def user_bg_video(video_name: str) -> None:
    """Play the named user background video in the terminal."""
    # The video plays over the current background and is removed when it ends,
    # leaving that background as it was.
    await term_bg_video(f"https://pub-59cb84bebe804fd1b3257bb6c283a2b3.r2.dev/{video_name}")


@public
async def user_bg_default() -> None:
    """Set the user default background image."""
    bg_path = _user_default_bg_path()
    await atlantis.client_command(f"/terminal brightness 0.3")
    await atlantis.client_command(f"/terminal desaturate 1")
    if bg_path:
        await atlantis.set_background(
            bg_path,
            vertical_align=USER_DEFAULT_BG_ALIGN,
        )
