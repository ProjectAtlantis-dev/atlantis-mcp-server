"""Upload callback for files dropped on the chat panel.

Stores the file under Media/users/ (same validation as the Media uploader),
then posts it with atlantis.client_image / client_video where it was dropped:
in chat view (chatView), into the terminal's bound chat shell so everyone in
the chat sees it; otherwise into the caller's terminal. Returns
nothing, so no text lands in the chat input.
Configure with /callback set upload Media/chat_upload (or upload auto).
"""

import logging

import atlantis

from .storage import save_upload

logger = logging.getLogger("dynamic_function")


@public
@upload
async def chat_upload(base64Content: str, filename: str, filetype: str, chatView: bool) -> None:
    """Store an image or video dropped on the chat and show it in the chat."""
    if not filetype.startswith(("image/", "video/")):
        raise ValueError(f"Only images and videos can be dropped on the chat, got {filetype!r}")
    username = atlantis.get_caller()
    if not username:
        raise PermissionError("chat_upload needs a caller")
    saved_path, created = save_upload(username, base64Content, filename, filetype)
    logger.info(f"chat_upload {'saved' if created else 'reused'} {saved_path} for {username}")

    shell = "chat" if chatView else "caller"
    if filetype.startswith("image/"):
        await atlantis.client_image(str(saved_path), image_format=filetype, sid=username, shell=shell)
    else:
        await atlantis.client_video(str(saved_path), video_format=filetype, shell=shell)
