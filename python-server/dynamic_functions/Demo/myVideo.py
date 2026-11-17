import os
import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@visible
async def myVideo():
    """Display the TaffyWide.mp4 video"""
    video_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "TaffyWide.mp4")
    await atlantis.client_video(video_path)
