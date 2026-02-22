import atlantis
import logging
import os

logger = logging.getLogger("mcp_server")


@game
async def game():
    """
    Main game function
    """

    await atlantis.client_command("/silent on")

    # get user id ('brickhouse')
    user_id = atlantis.get_caller()
    logger.info(f"Game started for user: {user_id}")

    owner_id = atlantis.get_owner()
    await atlantis.client_log(f"Owner ID: {owner_id}")  # TEMP

    await atlantis.client_command("/chat set " + owner_id + "*kitty_claw")

    # set background
    await atlantis.client_command("/silent off")
    image_path = os.path.join(os.path.dirname(__file__), "builder.jpg")



    await atlantis.set_background(image_path)




    # send kitty face image

    kitty_path = os.path.join(os.path.dirname(__file__), "kitty_face_compressed.jpg")
    await atlantis.client_image(kitty_path)



    await atlantis.client_log(f"Kitty is at the front desk!")




