"""Per-caller media storage: Media/users/<caller>/<images|videos|audio>/."""

import base64
import binascii
import hashlib
import os
import re
import tempfile
from pathlib import Path


USERS_ROOT = Path(__file__).resolve().parent / "users"
MAX_UPLOAD_BYTES = 250 * 1024 * 1024

# Browser MIME type -> (collection, suffix). Anything else is rejected.
MEDIA_TYPES = {
    "image/png": ("images", ".png"),
    "image/jpeg": ("images", ".jpg"),
    "image/webp": ("images", ".webp"),
    "video/mp4": ("videos", ".mp4"),
    "video/webm": ("videos", ".webm"),
    "video/ogg": ("videos", ".ogv"),
    "video/quicktime": ("videos", ".mov"),
    "audio/mpeg": ("audio", ".mp3"),
    "audio/mp3": ("audio", ".mp3"),
    "audio/wav": ("audio", ".wav"),
    "audio/x-wav": ("audio", ".wav"),
    "audio/flac": ("audio", ".flac"),
    "audio/x-flac": ("audio", ".flac"),
    "audio/ogg": ("audio", ".ogg"),
    "audio/mp4": ("audio", ".m4a"),
    "audio/x-m4a": ("audio", ".m4a"),
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str) -> str:
    component = _UNSAFE.sub("_", value).strip("._")
    if not component:
        raise ValueError(f"Cannot build a safe path component from {value!r}")
    return component


def _content_matches(filetype: str, payload: bytes) -> bool:
    if filetype == "image/png":
        return payload.startswith(b"\x89PNG\r\n\x1a\n")
    if filetype == "image/jpeg":
        return payload.startswith(b"\xff\xd8\xff")
    if filetype == "image/webp":
        return len(payload) >= 12 and payload.startswith(b"RIFF") and payload[8:12] == b"WEBP"
    if filetype in {"video/mp4", "video/quicktime", "audio/mp4", "audio/x-m4a"}:
        return len(payload) >= 12 and payload[4:8] == b"ftyp"
    if filetype == "video/webm":
        return payload.startswith(b"\x1a\x45\xdf\xa3")
    if filetype in {"audio/mpeg", "audio/mp3"}:
        # An ID3 tag, or straight into an MPEG audio frame sync.
        return payload.startswith(b"ID3") or (len(payload) >= 2 and payload[0] == 0xFF and payload[1] & 0xE0 == 0xE0)
    if filetype in {"audio/wav", "audio/x-wav"}:
        return len(payload) >= 12 and payload.startswith(b"RIFF") and payload[8:12] == b"WAVE"
    if filetype in {"audio/flac", "audio/x-flac"}:
        return payload.startswith(b"fLaC")
    if filetype in {"video/ogg", "audio/ogg"}:
        return payload.startswith(b"OggS")
    raise ValueError(f"No content check for {filetype!r}")


def media_folder(username: str, collection: str) -> Path:
    """Return one caller's folder for a collection (images, videos or audio)."""
    return USERS_ROOT / _safe_component(username) / _safe_component(collection)


def save_upload(username: str, base64_content: str, filename: str, filetype: str) -> tuple[Path, bool]:
    """Validate one browser upload and store it; return (path, created).

    Files are named by content digest, so uploading the same bytes twice
    returns the existing file instead of writing a copy.
    """
    if filetype not in MEDIA_TYPES:
        raise ValueError(f"Upload must be one of {sorted(MEDIA_TYPES)}, got {filetype!r}")
    collection, suffix = MEDIA_TYPES[filetype]

    # The browser's FileReader hands us a data URL; strip its header.
    encoded = base64_content.split(",", 1)[1] if base64_content.startswith("data:") else base64_content
    try:
        payload = base64.b64decode(encoded, validate=True)
    except binascii.Error as error:
        raise ValueError("The uploaded file is not valid base64 data") from error
    if not payload:
        raise ValueError("The uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The uploaded file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not _content_matches(filetype, payload):
        raise ValueError(f"The upload content does not match {filetype}")

    folder = media_folder(username, collection)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    existing = sorted(folder.glob(f"{digest}_*{suffix}"))
    if existing:
        return existing[0], False

    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{digest}_{_safe_component(Path(filename).stem)}{suffix}"
    with tempfile.NamedTemporaryFile(dir=folder, prefix=".saving_", delete=False) as temp:
        temp.write(payload)
        temporary_path = Path(temp.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination, True
