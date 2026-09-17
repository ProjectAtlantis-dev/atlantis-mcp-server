"""UTF-8 file storage for Atlantis file callbacks.

Configure with /callback set file Home/file_callback (or file auto).
Only bare filenames in the script's directory are accepted. Reading
a missing file raises FileNotFoundError. Saving creates files. Paths fail.
"""

import asyncio
from pathlib import Path, PureWindowsPath
from typing import Optional


def _file_path(filename: str) -> Path:
    if not filename.strip():
        raise ValueError("Must specify a filename")
    if (
        filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or PureWindowsPath(filename).drive
    ):
        raise ValueError("Expected a bare filename; paths are not allowed")
    path = Path(__file__).resolve().parent / filename
    if path.is_symlink():
        raise ValueError("Symbolic links are not allowed")
    return path


def file_get(filename: str) -> str:
    """Read a UTF-8 file; raise FileNotFoundError if it does not exist."""
    path = _file_path(filename)
    return path.read_text(encoding="utf-8")


def file_set(filename: str, content: str) -> str:
    """Save a UTF-8 file in the script's directory."""
    path = _file_path(filename)
    path.write_text(content, encoding="utf-8")
    return "Saved " + filename


def _check_suffix(suffix: str) -> None:
    if not suffix.strip() or any(char in suffix for char in ". /\\:"):
        raise ValueError("Suffix must be an extension without a dot or path, e.g. 'json'")


def file_list(suffix: str) -> list[dict[str, str]]:
    """List files with the given extension (without a dot) as name/suffix rows, excluding symlinks."""
    _check_suffix(suffix)
    return [
        {"name": path.stem, "suffix": suffix}
        for path in sorted(Path(__file__).resolve().parent.iterdir())
        if path.suffix == "." + suffix and not path.is_symlink() and path.is_file()
    ]


@file
async def file_callback(
    operation: str, filename: str = "", content: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str | list[dict[str, str]]:
    """Get/set a bare filename, or list files by suffix in the script's directory.

    List takes its suffix as the second argument or named suffix (e.g. "json",
    without a dot). Supply only one of these forms.
    Set requires content. For get/set with a suffix, a filename without an
    extension gets ".<suffix>" appended; an explicit extension must match it.
    """
    if operation == "list":
        if filename:
            if suffix is not None:
                raise ValueError("Supply the list suffix either as the second argument or as suffix, not both")
            suffix = filename
        if suffix is None:
            raise ValueError("The list operation requires a suffix")
        return await asyncio.to_thread(file_list, suffix)
    if operation in {"get", "set"} and suffix is not None:
        _check_suffix(suffix)
        if not filename.strip():
            raise ValueError("Must specify a filename")
        extension = Path(filename).suffix
        if not extension:
            filename += "." + suffix
        elif extension != "." + suffix:
            raise ValueError(f"Filename extension {extension!r} does not match suffix {suffix!r}")
    if operation == "get":
        return await asyncio.to_thread(file_get, filename)
    if operation == "set":
        if content is None:
            raise ValueError("The set operation requires content")
        return await asyncio.to_thread(file_set, filename, content)
    raise ValueError("Unknown file operation: " + operation)
