from atlantis_simulation.player_control import mcp_command, player_id
from atlantis_simulation.viewer import capabilities
from atlantis_host_adapters.identity import current_principal


@visible
def player_attach() -> dict:
    """Attach your existing registry UUID at its operator-configured spawn; reconnect preserves position."""
    return mcp_command('attach')


@visible
def player_claim() -> dict:
    """Possess your physical player, not the spectator camera. Returns exclusive expiring lease and sequence."""
    return mcp_command('claim')


@visible
def player_walk(lease_id: str, sequence: int, east: float, north: float, duration_ms: int = 350) -> dict:
    """World ENU walking direction, vector length <=1, max 2m/s, duration 50..1000ms. Zero input refreshes presence. No teleportation."""
    return mcp_command('move', {'leaseId': lease_id, 'sequence': sequence, 'east': east, 'north': north, 'durationMs': duration_ms})


@visible
def player_release(lease_id: str) -> dict:
    """Stop walking, relinquish control, and invalidate proximity presence immediately."""
    return mcp_command('release', {'leaseId': lease_id})


@visible
def player_observe() -> dict:
    """Read your authoritative local ENU position, revision and fresh/stale presence."""
    return mcp_command('observe')


@visible
def player_request_entry(airlock_id: str, lease_id: str, expected_revision: int) -> dict:
    """Request outer-door opening: requires resident ACL, fresh player position near configured outer entrance, possession and door interlock. Returns action, not completed passage."""
    return mcp_command('requestEntry', {'airlockId': airlock_id, 'leaseId': lease_id, 'expectedRevision': expected_revision})


@visible
def player_entry_action() -> dict:
    """Read your latest entry action status. Succeeded means outer door reached open, not that passage finished."""
    return mcp_command('action')


@visible
def player_viewer_access(ttl_seconds: int = 300) -> dict:
    """Issue a scoped human-input capability for your physical player only."""
    principal = current_principal()
    return capabilities.issue(principal, ttl_seconds, control_player_id=player_id(principal))
