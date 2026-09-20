from atlantis_simulation.vehicle_control import mcp_command, owned_asset
from atlantis_simulation.viewer import capabilities
from atlantis_host_adapters.identity import current_principal


@visible
def vehicle_attach(asset_id: str) -> dict:
    """Attach an existing bank UUID using operator-configured model, pose and elevation grid; never spawn a duplicate."""
    return mcp_command('attach', asset_id)


@visible
def vehicle_claim(asset_id: str) -> dict:
    """Acquire exclusive control; returns lease ID and sequence. Another active controller blocks this action."""
    return mcp_command('claim', asset_id)


@visible
def vehicle_drive(asset_id: str, lease_id: str, sequence: int, throttle: float,
                  steering: float, brake: float = 0, duration_ms: int = 500) -> dict:
    """Ground input: throttle/steering -1..1, brake 0..1, duration 50..2000ms. Positive steering turns left. Sequence must increase. Expiry brakes."""
    return mcp_command('drive', asset_id, {'leaseId': lease_id, 'sequence': sequence,
        'throttle': throttle, 'steering': steering, 'brake': brake, 'durationMs': duration_ms})


@visible
def vehicle_release(asset_id: str, lease_id: str) -> dict:
    """Release this controller's lease and brake; does not transfer bank ownership."""
    return mcp_command('release', asset_id, {'leaseId': lease_id})


@visible
def vehicle_observe(asset_id: str) -> dict:
    """Read the bank-owned vehicle's authoritative pose, speed and command status."""
    return mcp_command('observe', asset_id)


@visible
def vehicle_viewer_access(asset_id: str, ttl_seconds: int = 300) -> dict:
    """Issue a short-lived browser control capability for one owned vehicle, not arbitrary world writes."""
    principal = current_principal()
    owned_asset(principal, asset_id)
    return capabilities.issue(principal, ttl_seconds, control_vehicle_id=asset_id)
