"""UTF-8 file storage for Atlantis file callbacks.

Configure with /callback set file Home/file_callback (or file auto).
Only bare filenames in the current working directory are accepted. Reading
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
    path = Path.cwd() / filename
    if path.is_symlink():
        raise ValueError("Symbolic links are not allowed")
    return path


def file_get(filename: str) -> str:
    """Read a UTF-8 file; raise FileNotFoundError if it does not exist."""
    path = _file_path(filename)
    return path.read_text(encoding="utf-8")


def file_set(filename: str, content: str) -> str:
    """Save a UTF-8 file in the current working directory."""
    path = _file_path(filename)
    path.write_text(content, encoding="utf-8")
    return "Saved " + filename


def file_list(suffix: str) -> list[str]:
    """List files with the given extension (without a dot), excluding symlinks."""
    if not suffix.strip() or any(char in suffix for char in ". /\\:"):
        raise ValueError("Suffix must be an extension without a dot or path, e.g. 'json'")
    return sorted(
        path.name
        for path in Path.cwd().iterdir()
        if path.suffix == "." + suffix and not path.is_symlink() and path.is_file()
    )


@file
async def file_callback(
    operation: str, filename: str = "", content: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str | list[str]:
    """Get/set a bare filename, or list files by suffix in the current directory.

    List takes its suffix as the second argument or named suffix (e.g. "json",
    without a dot). Supply only one of these forms.
    Set requires content. For get/set, an explicit filename extension must
    match suffix when supplied.
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
        extension = Path(filename).suffix
        if extension and extension != "." + suffix:
            raise ValueError(f"Filename extension {extension!r} does not match suffix {suffix!r}")
    if operation == "get":
        return await asyncio.to_thread(file_get, filename)
    if operation == "set":
        if content is None:
            raise ValueError("The set operation requires content")
        return await asyncio.to_thread(file_set, filename, content)
    raise ValueError("Unknown file operation: " + operation)
