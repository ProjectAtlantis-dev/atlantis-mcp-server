"""Generic per-path todo list storage.

The module knows nothing about apps, players, games, or bots. Callers supply a
relative path under Data/; layout conventions are the caller's choice. For
example: "FlowCentral/Kitty/game3/greeting_todo".
"""

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger("mcp_server")

DATA_ROOT = os.path.dirname(__file__)
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


def _resolve(path: str) -> str:
    if not path:
        raise ValueError("todo path cannot be empty")
    rel = path.strip("/")
    if not rel.endswith(".json"):
        rel += ".json"
    full = os.path.normpath(os.path.join(DATA_ROOT, rel))
    if not full.startswith(DATA_ROOT + os.sep) and full != DATA_ROOT:
        raise ValueError(f"todo path escapes Data/: {path}")
    return full


def _resolve_dir(directory: str) -> str:
    full = os.path.normpath(os.path.join(DATA_ROOT, directory.strip("/")))
    if not (full == DATA_ROOT or full.startswith(DATA_ROOT + os.sep)):
        raise ValueError(f"todo dir escapes Data/: {directory}")
    return full


def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
            return json.loads(raw) if raw.strip() else default
    except FileNotFoundError:
        return default


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def todo_read(path: str) -> List[Dict[str, Any]]:
    """Read a todo list at Data/<path>.json. Returns [] if missing."""
    return _read_json(_resolve(path), [])


def todo_write(path: str, items: List[Dict[str, Any]]) -> None:
    """Write a todo list to Data/<path>.json."""
    _write_json(_resolve(path), items)


def todo_delete(path: str) -> None:
    """Delete a todo list file if it exists."""
    full = _resolve(path)
    if os.path.exists(full):
        os.remove(full)


def todo_list(directory: str = "") -> List[str]:
    """List todo names directly under Data/<directory>/ (no extension, sorted)."""
    full = _resolve_dir(directory)
    if not os.path.isdir(full):
        return []
    return sorted(
        name[:-5]
        for name in os.listdir(full)
        if name.endswith(".json") and os.path.isfile(os.path.join(full, name))
    )


def todo_walk(directory: str = "") -> List[str]:
    """Recursively list all todos under Data/<directory>/ as paths (no extension)."""
    full = _resolve_dir(directory)
    if not os.path.isdir(full):
        return []
    results = []
    for root, _, files in os.walk(full):
        for name in files:
            if name.endswith(".json"):
                rel = os.path.relpath(os.path.join(root, name), DATA_ROOT)
                results.append(rel[:-5])
    return sorted(results)


def _validate(item: Dict[str, Any]) -> Dict[str, Any]:
    item_id = str(item.get("id", "")).strip() or "?"
    content = str(item.get("content", "")).strip() or "(no description)"
    status = str(item.get("status", "pending")).strip().lower()
    if status not in VALID_STATUSES:
        status = "pending"
    return {"id": item_id, "status": status, "content": content}


def _merge_items(existing_items: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = {item["id"]: item for item in existing_items}
    for t in new_items:
        item_id = str(t.get("id", "")).strip()
        if not item_id:
            continue
        if item_id in existing:
            if "content" in t and t["content"]:
                existing[item_id]["content"] = str(t["content"]).strip()
            if "status" in t and t["status"]:
                status = str(t["status"]).strip().lower()
                if status in VALID_STATUSES:
                    existing[item_id]["status"] = status
        else:
            validated = _validate(t)
            existing[validated["id"]] = validated
            existing_items.append(validated)

    seen = set()
    rebuilt = []
    for item in existing_items:
        current = existing.get(item["id"], item)
        if current["id"] not in seen:
            rebuilt.append(current)
            seen.add(current["id"])
    return rebuilt


def _format_result(items: List[Dict[str, Any]]) -> str:
    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")
    return json.dumps({
        "todos": items,
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        },
    })


# =========================================================================
# Pseudo-tool definition & handler — dispatched from Home/Bot/turn.py
# =========================================================================

TODO_PSEUDO_TOOL = {
    'type': 'function',
    'function': {
        'name': 'todo',
        'description': (
            'Manage a named task list stored at Data/<path>.json. '
            'Call with only "path" to read the current list.\n\n'
            'Writing:\n'
            '- Provide "todos" array to create/update items.\n'
            '- merge=false (default): replace the entire list with a fresh plan.\n'
            '- merge=true: update existing items by id, add any new ones.\n\n'
            'Each item: {id: string, content: string, status: pending|in_progress|completed|cancelled}\n'
            'List order is priority. Only ONE item in_progress at a time. '
            'Mark items completed immediately when done.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Path of the todo list under Data/, e.g. "FlowCentral/<user>/<game>/greeting_todo".'
                },
                'todos': {
                    'type': 'array',
                    'description': 'Task items to write. Omit to read the current list.',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string', 'description': 'Unique item identifier'},
                            'status': {
                                'type': 'string',
                                'enum': ['pending', 'in_progress', 'completed', 'cancelled'],
                                'description': 'Current status'
                            },
                            'content': {'type': 'string', 'description': 'Task description'}
                        },
                        'required': ['id', 'content', 'status']
                    }
                },
                'merge': {
                    'type': 'boolean',
                    'description': 'true: update existing items by id. false (default): replace entire list.',
                    'default': False
                }
            },
            'required': ['path']
        }
    }
}


async def handle_todo_tool(arguments: dict) -> str:
    """Handle the todo pseudo-tool. Requires 'path' in arguments."""
    path = arguments.get('path')
    if not path:
        return json.dumps({"error": "todo tool requires a 'path' argument"})

    todos_arg = arguments.get('todos')
    merge = arguments.get('merge', False)
    items = todo_read(path)

    if todos_arg is not None:
        if not merge:
            items = [_validate(t) for t in todos_arg]
        else:
            items = _merge_items(items, todos_arg)
        todo_write(path, items)
        logger.info(f"todo pseudo-tool: wrote {len(items)} items to '{path}' (merge={merge})")
    else:
        logger.info(f"todo pseudo-tool: read {len(items)} items from '{path}'")

    return _format_result(items)
