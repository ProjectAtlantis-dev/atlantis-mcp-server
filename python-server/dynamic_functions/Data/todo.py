import atlantis
import logging
import json
from typing import Optional
from dynamic_functions.Data.main import read_player, set_player_field

logger = logging.getLogger("mcp_server")

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


def _get_caller() -> str:
    """Get the current caller username."""
    return atlantis.get_caller() or ""


def _read_store(caller: str = "") -> list:
    """Read the todo list from the player database."""
    caller = caller or _get_caller()
    if not caller:
        return []
    data = read_player(caller)
    return data.get("todos", [])


def _write_store(items: list, caller: str = ""):
    """Write the todo list to the player database."""
    caller = caller or _get_caller()
    if not caller:
        logger.warning("todo _write_store: no caller, cannot persist")
        return
    set_player_field(caller, "todos", items)


def _validate(item):
    """Validate and normalize a todo item."""
    item_id = str(item.get("id", "")).strip() or "?"
    content = str(item.get("content", "")).strip() or "(no description)"
    status = str(item.get("status", "pending")).strip().lower()
    if status not in VALID_STATUSES:
        status = "pending"
    return {"id": item_id, "status": status, "content": content}


def _merge_items(existing_items, new_items):
    """Merge new items into existing list by id. Append unknown ids."""
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

    # Rebuild preserving order
    seen = set()
    rebuilt = []
    for item in existing_items:
        current = existing.get(item["id"], item)
        if current["id"] not in seen:
            rebuilt.append(current)
            seen.add(current["id"])
    return rebuilt


def _format_result(items):
    """Format the todo list with summary counts."""
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
# Pseudo-tool definition & handler — imported by Kitty backends
# =========================================================================

TODO_PSEUDO_TOOL = {
    'type': 'function',
    'function': {
        'name': 'todo',
        'description': (
            'Manage your task list for this session. Use for multi-step procedures. '
            'Call with no parameters to read the current list.\n\n'
            'Writing:\n'
            '- Provide "todos" array to create/update items\n'
            '- merge=false (default): replace the entire list with a fresh plan\n'
            '- merge=true: update existing items by id, add any new ones\n\n'
            'Each item: {id: string, content: string, status: pending|in_progress|completed|cancelled}\n'
            'List order is priority. Only ONE item in_progress at a time.\n'
            'Mark items completed immediately when done.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'todos': {
                    'type': 'array',
                    'description': 'Task items to write. Omit to read current list.',
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
            'required': []
        }
    }
}


async def handle_todo_tool(arguments: dict) -> str:
    """Handle the todo pseudo-tool. Called from Kitty's tool-call loop."""
    todos_arg = arguments.get('todos')
    merge = arguments.get('merge', False)

    items = _read_store()

    if todos_arg is not None:
        if not merge:
            items = [_validate(t) for t in todos_arg]
        else:
            items = _merge_items(items, todos_arg)
        _write_store(items)
        logger.info(f"todo pseudo-tool: wrote {len(items)} items (merge={merge})")
    else:
        logger.info(f"todo pseudo-tool: read {len(items)} items")

    return _format_result(items)


# =========================================================================
# Visible tools — callable via MCP for debugging
# =========================================================================

@visible
async def todo(todos: Optional[str] = None, merge: bool = False):
    """
    Manage your task list for this session. Use for multi-step procedures.
    Call with no parameters to read the current list.

    Writing:
    - Provide 'todos' as a JSON array string to create/update items.
    - merge=false (default): replace the entire list with a fresh plan.
    - merge=true: update existing items by id, add any new ones.

    Each item: {"id": "string", "content": "description", "status": "pending|in_progress|completed|cancelled"}
    List order is priority. Only ONE item in_progress at a time.
    Mark items completed immediately when done.

    Args:
        todos: JSON array string of task items. Omit to read current list.
        merge: If true, update existing items by id. If false, replace entire list.
    """
    items = _read_store()

    if todos is not None:
        if isinstance(todos, str):
            todo_list = json.loads(todos)
        else:
            todo_list = todos

        if not merge:
            items = [_validate(t) for t in todo_list]
        else:
            items = _merge_items(items, todo_list)

        _write_store(items)
        logger.info(f"todo: wrote {len(items)} items (merge={merge})")
    else:
        logger.info(f"todo: read {len(items)} items")

    return _format_result(items)


@visible
async def list_tasks():
    """
    Debug tool — returns the current todo list for this session.
    Use this to inspect task state without modifying anything.
    """
    items = _read_store()
    logger.info(f"list_tasks: {len(items)} items")
    return _format_result(items)
