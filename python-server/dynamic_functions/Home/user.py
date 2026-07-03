"""User/session-scoped tools."""

import atlantis
import base64
import json
import logging
import mimetypes
import os

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


def _user_default_bg_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    with open(image_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


async def _restore_user_default_bg_when_bg_video_ends() -> None:
    bg_path = _user_default_bg_path()
    if not bg_path:
        return
    bg_url = _user_default_bg_data_url(bg_path)
    await atlantis.client_terminal_script(f"""
(function(){{
  var bgUrl = {json.dumps(bg_url)};
  var verticalAlign = {json.dumps(USER_DEFAULT_BG_ALIGN)};

  function restoreDefaultBg() {{
    var chatFeedback = document.getElementById('chatFeedback');
    if (!chatFeedback) return;

    var oldMedia = document.querySelectorAll(
      '#feedbackBgVideo, video[data-background-video="true"], iframe[data-background-player="true"]'
    );
    for (var i = 0; i < oldMedia.length; i++) {{
      try {{
        if (oldMedia[i].pause) oldMedia[i].pause();
        oldMedia[i].removeAttribute('src');
        if (oldMedia[i].load) oldMedia[i].load();
      }} catch (_err) {{}}
      oldMedia[i].remove();
    }}

    chatFeedback.style.background = 'black';
    chatFeedback.style.backgroundImage = 'url(' + JSON.stringify(bgUrl) + ')';
    chatFeedback.style.backgroundSize = 'cover';
    chatFeedback.style.backgroundPosition = 'center ' + verticalAlign;
    chatFeedback.style.backgroundRepeat = 'no-repeat';
  }}

  function attachBgVideoRestoreHook() {{
    var bgVideos = document.querySelectorAll('video[data-background-video="true"]');
    var bgVideo = bgVideos.length ? bgVideos[bgVideos.length - 1] : null;
    if (!bgVideo) return false;
    if (bgVideo.dataset.userDefaultRestoreAttached === 'true') return true;
    bgVideo.dataset.userDefaultRestoreAttached = 'true';
    bgVideo.addEventListener('ended', function() {{
      setTimeout(restoreDefaultBg, 0);
    }}, {{ once: true }});
    bgVideo.addEventListener('error', function() {{
      setTimeout(restoreDefaultBg, 0);
    }}, {{ once: true }});
    return true;
  }}

  if (attachBgVideoRestoreHook()) return;
  var attempts = 0;
  var timer = setInterval(function() {{
    attempts += 1;
    if (attachBgVideoRestoreHook() || attempts >= 300) clearInterval(timer);
  }}, 100);
}})();
""")


@public
async def user_bg_video(video_name: str) -> None:
    """Play the named user background video in the terminal."""
    await term_bg_video(f"https://pub-59cb84bebe804fd1b3257bb6c283a2b3.r2.dev/{video_name}")
    await _restore_user_default_bg_when_bg_video_ends()


@public
async def user_bg_default() -> None:
    """Set the user default background image."""
    bg_path = _user_default_bg_path()
    if bg_path:
        await atlantis.set_background(
            bg_path,
            vertical_align=USER_DEFAULT_BG_ALIGN,
        )
    await atlantis.client_command(f"/terminal brightness 0.3")
    await atlantis.client_command(f"/terminal desaturate 1")
