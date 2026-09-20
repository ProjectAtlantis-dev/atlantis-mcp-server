"""Discover and call state-backed object functions from MCP or the viewer."""
from atlantis_host_adapters.identity import current_principal
from dynamic_functions.Terrain.Objects import runtime
# Rebind the owner-checked HTTP facade when this dynamic module is loaded.
from dynamic_functions.Terrain import viewer_server


@visible
def functions(asset_id: str) -> dict:
    """List the functions currently available for your bank-owned object, their parameters, bound mission/revision and state. The viewer selection menu uses this same discovery."""
    return runtime.describe(current_principal(), asset_id)


@visible
def inspect(asset_id: str) -> dict:
    """Inspect a bank-owned vehicle or structure, including objects without attached controllers."""
    return runtime.invoke(current_principal(), asset_id, 'inspect', {})


@visible
def call(asset_id: str, action_id: str, parameters: dict) -> dict:
    """Invoke a discovered object function. Ownership and current availability are checked again. Supply bound mission IDs/revisions from functions; stale forms reject."""
    return runtime.invoke(current_principal(), asset_id, action_id, parameters)


@visible
def index() -> dict:
    """Discover and invoke runnable functions on a selected bank-owned object."""
    return {"module": "Terrain/Objects", "discovery": "functions", "execute": "call"}
