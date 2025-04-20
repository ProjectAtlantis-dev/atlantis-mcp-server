"""
Manages the lifecycle of dynamic MCP servers (configs).
Stores each server config as a JSON file under SERVERS_DIR.
"""
import os
import json
import logging
from typing import Any, Dict, Optional, List
from state import SERVERS_DIR, logger
from mcp.types import TextContent

# Ensure SERVERS_DIR exists
os.makedirs(SERVERS_DIR, exist_ok=True)

# --- 1. File Save/Load ---
def _fs_save_server(name: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Saves the provided JSON config dict to {name}.json in SERVERS_DIR.
    Returns the full path if successful, None otherwise.
    """
    safe_name = f"{name}.json"
    file_path = os.path.join(SERVERS_DIR, safe_name)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"💾 Saved server config for '{name}' to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ _fs_save_server: Failed to write {file_path}: {e}")
        return None


def _fs_load_server(name: str) -> Optional[Dict[str, Any]]:
    """
    Loads and returns the JSON config dict from {name}.json in SERVERS_DIR.
    Returns None if not found or error.
    """
    safe_name = f"{name}.json"
    file_path = os.path.join(SERVERS_DIR, safe_name)
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ _fs_load_server: Config not found for '{name}' at {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ _fs_load_server: Failed to read {file_path}: {e}")
        return None

# --- 2. CRUD Operations ---
def server_add(name: str, config: Dict[str, Any]) -> bool:
    """
    Adds a new server config. Returns False if it already exists.
    """
    if _fs_load_server(name) is not None:
        logger.warning(f"Add failed: Server '{name}' already exists.")
        return False
    return _fs_save_server(name, config) is not None


def server_remove(name: str) -> bool:
    """
    Removes the server config by deleting its JSON file. Returns False if missing.
    """
    file_path = os.path.join(SERVERS_DIR, f"{name}.json")
    if not os.path.exists(file_path):
        logger.warning(f"Remove failed: Server '{name}' does not exist.")
        return False
    try:
        os.remove(file_path)
        logger.info(f"🗑️ Removed server config '{name}' at {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ server_remove: Failed to delete {file_path}: {e}")
        return False


def server_get(name: str) -> Optional[Dict[str, Any]]:
    """
    Returns the server config dict or None if not found.
    """
    return _fs_load_server(name)


def server_list() -> List[str]:
    """
    Lists all server names available in SERVERS_DIR.
    """
    files = os.listdir(SERVERS_DIR)
    return [os.path.splitext(f)[0] for f in files if f.endswith('.json')]


def server_set(args: Dict[str, Any], server) -> List[TextContent]:
    """
    MCP handler to add/update a server config.
    Expects args['name']: str, args['config']: dict
    """
    name = args.get('name')
    config = args.get('config')
    if not name or not isinstance(config, dict):
        msg = "Missing or invalid parameters: 'name' must be str and 'config' must be dict."
        logger.error(f"❌ server_set: {msg}")
        return [TextContent(type='text', text=msg)]

    existing = _fs_load_server(name)
    action = 'Updated' if existing else 'Added'
    saved = _fs_save_server(name, config)
    if not saved:
        msg = f"Failed to save server config for '{name}'."
        return [TextContent(type='text', text=msg)]

    # Notify clients if server has a notification method
    if hasattr(server, '_notify_tool_list_changed'):
        try:
            # Reuse tool notification for servers list
            await server._notify_tool_list_changed()
        except Exception as e:
            logger.error(f"❌ Failed to notify clients after '{action}' server '{name}': {e}")
    
    return [TextContent(type='text', text=f"Server '{name}' {action.lower()} successfully.")]


def server_validate(name: str) -> Dict[str, Any]:
    """
    Validates that the server config JSON has required keys. Returns a dict with 'valid':bool and 'error':Optional[str].
    """
    config = _fs_load_server(name)
    if config is None:
        return {'valid': False, 'error': f"Server '{name}' config not found."}
    # Basic required keys
    req = ['command', 'args', 'env']
    missing = [k for k in req if k not in config]
    if missing:
        return {'valid': False, 'error': f"Missing keys: {missing}"}
    return {'valid': True, 'error': None}
