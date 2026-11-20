import atlantis
import base64
import logging
import os

logger = logging.getLogger("dynamic_function")

_HEROINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heroines")
_IMAGE_MIMES = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}

_HEROINES = [
    {"name": "Motoko Kusanagi", "creator": "Masamune Shirow", "image": "motoko.jpg"},
    {"name": "Misato Katsuragi", "creator": "Hideaki Anno", "image": "misato.jpg"},
    {"name": "Faye Valentine", "creator": "Shinichirō Watanabe", "image": "faye.jpg"},
    {"name": "Nico Robin", "creator": "Eiichiro Oda", "image": "robin.jpg"},
    {"name": "Fujiko Mine", "creator": "Monkey Punch", "image": "fujiko.jpg"},
    {"name": "Lynae", "creator": "Kuro Games", "image": "lynae.jpg"},
    {"name": "Asuka Langley Soryu", "creator": "Hideaki Anno", "image": "asuka.jpg"},
]


def _image_data_uri(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in _IMAGE_MIMES:
        raise ValueError(f"Unsupported image type: {filename!r}")
    with open(os.path.join(_HEROINES_DIR, filename), "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/{_IMAGE_MIMES[ext]};base64,{data}"


@visible
async def myTableWithImages() -> list[dict[str, object]]:
    """Return a table of anime heroines with a thumbnail of each"""
    # the image column holds a data URI that Atlantis renders as a thumbnail
    return [
        {
            "image": _image_data_uri(row["image"]),
            "name": row["name"],
            "creator": row["creator"],
        }
        for row in _HEROINES
    ]
