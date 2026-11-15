"""Store one browser upload under Media/users/.

Called by the Media uploader's dropzone (Media/upload.py), which shows its
own preview, so this only stores and returns the path. Files dropped on the
chat go through chat_upload.py instead.

Lives apart from upload.py because `def upload` there would shadow the
@upload decorator.
"""

import logging

import atlantis

from .caption import start_caption
from .storage import save_upload

logger = logging.getLogger("dynamic_function")


@public
async def upload_save(base64Content: str, filename: str, filetype: str) -> str:
    """Store one browser upload in the caller's image, video or audio folder."""
    username = atlantis.get_caller()
    if not username:
        raise PermissionError("upload_save needs a caller")
    saved_path, created = save_upload(username, base64Content, filename, filetype)
    logger.info(f"upload_save {'saved' if created else 'reused'} {saved_path} for {username}")
    if saved_path.parent.name == "images":
        start_caption(saved_path)
    return str(saved_path)
