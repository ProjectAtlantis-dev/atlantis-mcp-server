"""Game-scoped todo list storage.

Todos live at Data/{game_id}/todos/{sid}/{todo_name}.json.
The game_id comes from atlantis context; callers supply sid and todo_name.
"""

import atlantis
import json
import logging
import os
import re
from typing import Any, Dict, List

from dynamic_functions.Data.main import game_dir

logger = logging.getLogger("mcp_server")

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


def _safe(value: str, label: str = "value") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "").strip())
    if not safe:
        raise ValueError(f"Cannot use an empty {label}")
    return safe


def _require_context() -> tuple[str, str]:
    """Return (game_id, sid) from the atlantis context."""
    game_id = atlantis.get_game_id()
    if not game_id:
        raise ValueError("No active game in context")
    sid = atlantis.get_caller()
    if not sid:
        raise ValueError("No caller in context")
    return game_id, sid


def _todo_dir(game_id: str, sid: str) -> str:
    return os.path.join(game_dir(game_id, create=True), "todos", _safe(sid, "sid"))


def _todo_path(game_id: str, sid: str, todo_name: str) -> str:
    name = _safe(todo_name, "todo_name")
    if not name.endswith(".json"):
        name += ".json"
    return os.path.join(_todo_dir(game_id, sid), name)


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


# =========================================================================
# Core API
# =========================================================================

def todo_read(todo_name: str) -> List[Dict[str, Any]]:
    """Read a todo list. Returns [] if missing."""
    game_id, sid = _require_context()
    return _read_json(_todo_path(game_id, sid, todo_name), [])


def todo_write(todo_name: str, items: List[Dict[str, Any]]) -> None:
    """Write a todo list."""
    game_id, sid = _require_context()
    _write_json(_todo_path(game_id, sid, todo_name), items)


def todo_delete(todo_name: str) -> None:
    """Delete a todo list file if it exists."""
    game_id, sid = _require_context()
    path = _todo_path(game_id, sid, todo_name)
    if os.path.exists(path):
        os.remove(path)


def todo_list() -> List[str]:
    """List todo names for the caller in the current game."""
    game_id, sid = _require_context()
    d = _todo_dir(game_id, sid)
    if not os.path.isdir(d):
        return []
    return sorted(
        name[:-5]
        for name in os.listdir(d)
        if name.endswith(".json") and os.path.isfile(os.path.join(d, name))
    )


def todo_add(todo_name: str, item_id: str, content: str, status: str = "pending") -> Dict[str, Any]:
    """Add a single item to a todo list. Returns the validated item."""
    item = _validate({"id": item_id, "content": content, "status": status})
    items = todo_read(todo_name)
    for i, existing in enumerate(items):
        if existing["id"] == item["id"]:
            items[i] = item
            todo_write(todo_name, items)
            return item
    items.append(item)
    todo_write(todo_name, items)
    return item


def todo_update_status(todo_name: str, item_id: str, status: str) -> Dict[str, Any]:
    """Update the status of a single item. Returns the updated item. Raises ValueError if not found."""
    status = status.strip().lower()
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}. Must be one of {VALID_STATUSES}")
    items = todo_read(todo_name)
    for item in items:
        if item["id"] == item_id:
            item["status"] = status
            todo_write(todo_name, items)
            return item
    raise ValueError(f"Todo item '{item_id}' not found in {todo_name}")


def todo_remove(todo_name: str, item_id: str) -> bool:
    """Remove a single item by id. Returns True if found and removed."""
    items = todo_read(todo_name)
    before = len(items)
    items = [i for i in items if i["id"] != item_id]
    if len(items) < before:
        todo_write(todo_name, items)
        return True
    return False


def todo_get(todo_name: str, item_id: str) -> Dict[str, Any] | None:
    """Get a single item by id, or None."""
    for item in todo_read(todo_name):
        if item["id"] == item_id:
            return item
    return None


# =========================================================================
# Item helpers
# =========================================================================

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
# Pseudo-tool definition & handler — dispatched from Home/turn.py
# =========================================================================

TODO_PSEUDO_TOOL = {
    'type': 'function',
    'function': {
        'name': 'todo',
        'description': (
            'Manage a named task list for the current caller in the current game. '
            'Call with "todo_name" to read the current list.\n\n'
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
                'todo_name': {
                    'type': 'string',
                    'description': 'Name of the todo list, e.g. "greeting_todo".'
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
            'required': ['todo_name']
        }
    }
}


async def handle_todo_tool(arguments: dict) -> str:
    """Handle the todo pseudo-tool. Requires 'todo_name'."""
    todo_name = arguments.get('todo_name')
    if not todo_name:
        return json.dumps({"error": "todo tool requires a 'todo_name' argument"})

    todos_arg = arguments.get('todos')
    merge = arguments.get('merge', False)
    items = todo_read(todo_name)

    if todos_arg is not None:
        if not merge:
            items = [_validate(t) for t in todos_arg]
        else:
            items = _merge_items(items, todos_arg)
        todo_write(todo_name, items)
        logger.info(f"todo: wrote {len(items)} items to {todo_name} (merge={merge})")
    else:
        logger.info(f"todo: read {len(items)} items from {todo_name}")

    return _format_result(items)
