import os
import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@visible
async def myImage():
    """Display the happy.png image"""
    img_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "happy.png")
    await atlantis.client_image(img_path)
