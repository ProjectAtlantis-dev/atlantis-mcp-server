"""Cursor tools"""

import atlantis
import json
import logging
import os
from typing import Dict

logger = logging.getLogger("mcp_server")

from dynamic_functions.Home.common import home_path, _safe_id


def _cursor_path() -> str:
    session_id = atlantis.get_session_id()
    if not session_id:
        raise ValueError("Unable to determine session id")
    d = home_path("Data", "cursors", _safe_id(session_id, "session_id"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "cursor.json")


def _cursor_read() -> Dict:
    path = _cursor_path()
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid cursor.json: expected an object")
    return data


def _cursor_write(data: Dict) -> None:
    path = _cursor_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    logger.info(f"cursor written: {path} keys={list(data.keys())}")


@visible
def cursor_show() -> Dict:
    """Show the cursor structure for the calling session"""
    data = _cursor_read()
    _cursor_write(data)
    return data


@visible
async def cursor_merge(_prior):
    """Merge all values from _prior into the current cursor"""
    if not isinstance(_prior, dict):
        raise ValueError("_prior must be a dict")
    keys = list(_prior.keys())
    if not keys:
        raise ValueError("Prior object has no keys to merge")
    data = _cursor_read()
    data.update(_prior)
    _cursor_write(data)
    await atlantis.client_log(f"Merged {len(keys)} key(s) from prior into cursor: {', '.join(keys)}")

