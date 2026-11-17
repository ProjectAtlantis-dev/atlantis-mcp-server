import base64
import json
import mimetypes
import os

import atlantis


def _audio_data_url(audio_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(audio_path)
    if not mime_type or not mime_type.startswith("audio/"):
        mime_type = "audio/mpeg"
    with open(audio_path, "rb") as audio:
        encoded = base64.b64encode(audio.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@visible
async def win_background() -> None:
    """Test the Windows 95 forest tile as a repeated terminal background."""
    await atlantis.client_command("/terminal desaturate 0")
    await atlantis.client_command("/terminal brightness 1")
    forest_path = os.path.join(os.path.dirname(__file__), "win_forest.jpg")
    await atlantis.set_background(
        forest_path,
        vertical_align="top",
        horizontal_align="left",
        background_repeat="repeat",
        background_size="auto",
    )
    chime_path = os.path.join(os.path.dirname(__file__), "win95chime.mp3")
    chime_url = _audio_data_url(chime_path)
    # Use one-shot script so re-engaging the terminal does not replay old chimes.
    await atlantis.client_script(f"""
(function(){{
  var audio = new Audio({json.dumps(chime_url)});
  audio.preload = "auto";
  audio.play().catch(function(err) {{
    console.warn("Unable to play Windows 95 chime", err);
  }});
}})();
""")
