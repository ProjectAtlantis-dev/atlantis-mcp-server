"""Discover and call state-backed object functions from MCP or the viewer."""
from atlantis_host_adapters.identity import current_principal
from dynamic_functions.Terrain.Objects import runtime
# Rebind the owner-checked HTTP facade when this dynamic module is loaded.
from dynamic_functions.Terrain import viewer_server


@protected("terrain_access_authorized")
def functions(asset_id: str) -> dict:
    """List the functions currently available for your bank-owned object or the host-owner defense-demo scenario console, their parameters, bound mission/revision and state. The viewer selection menu uses this same discovery.

    :param asset_id: Bank UUID of the selected object, or the documented host-owner defense-demo console identifier. Never substitute a display name.
    """
    return runtime.describe(current_principal(), asset_id)


@protected("terrain_access_authorized")
def inspect(asset_id: str) -> dict:
    """Inspect a bank-owned vehicle or structure, including objects without attached controllers.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    """
    return runtime.invoke(current_principal(), asset_id, 'inspect', {})


@protected("terrain_access_authorized")
def call(asset_id: str, action_id: str, parameters: dict) -> dict:
    """Invoke a discovered object function. Ownership and current availability are checked again. Supply bound mission IDs/revisions from functions; stale forms reject.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param action_id: Exact currently available action ID from Terrain/Objects/functions for this asset.
    :param parameters: Arguments for the selected action from Terrain/Objects/functions, including its bound mission IDs/revisions. Do not reuse stale bindings.
    """
    return runtime.invoke(current_principal(), asset_id, action_id, parameters)


@protected("terrain_access_authorized")
def index() -> dict:
    """Discover and invoke runnable functions on a selected bank-owned object."""
    return {"module": "Terrain/Objects", "discovery": "functions", "execute": "call"}
