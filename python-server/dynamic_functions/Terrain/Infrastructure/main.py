"""Bank-linked habitat and infrastructure with server-enforced access."""
from atlantis_simulation import infrastructure_control as infrastructure
from atlantis_simulation.host import simulation_host


@protected("terrain_access_authorized")
def index() -> dict:
    """Bank UUIDs identify models; simulation tick owns all component transitions."""
    return {"module": "Terrain/Infrastructure", "visibility": "authenticated; object access enforced at execution"}


@visible
def catalog() -> dict:
    """List authored infrastructure models available in this host's world."""
    principal = infrastructure.principal_for()
    return simulation_host.command("GET", infrastructure.path(principal, "infrastructure-catalog"))


@visible
def place(model_id: str, instance_key: str, x: float, y: float, z: float,
          heading_deg: float = 0) -> dict:
    """Owner scenario-authoring: register and place a model with a bank UUID. Retry identical terms with the same instance_key. Coordinates are world-local ENU metres, not latitude/longitude."""
    return infrastructure.place(model_id, {"x": x, "y": y, "z": z}, heading_deg, instance_key)


@visible
def inspect(asset_id: str) -> dict:
    """Read ownership and live component positions/targets/revision. Accepted commands are not completed actions."""
    return infrastructure.inspect(asset_id)


@visible
def move(asset_id: str, x: float, y: float, z: float, heading_deg: float = 0) -> dict:
    """Owner scenario-authoring: reposition your registered structure; component state is preserved."""
    return infrastructure.move(asset_id, {"x": x, "y": y, "z": z}, heading_deg)


@visible
def instructions() -> dict:
    """Read supported component actions, preconditions and completion semantics. These are simulation contracts, not real safety certification."""
    principal = infrastructure.principal_for()
    return simulation_host.command("GET", infrastructure.path(principal, "component-contract"))


@protected("terrain_access_authorized")
def component_command(asset_id: str, action: str, expected_revision: int,
                      subject_kind: str, subject_id: str) -> dict:
    """Request a component action through an authorized nearby player or owned vehicle. Server rechecks bank identity, structure policy, authoritative proximity, revision and interlocks; inspect observes actual completion."""
    return infrastructure.control(asset_id, action, expected_revision, subject_kind, subject_id)


@visible
def configure_access(asset_id: str, interaction_radius_m: float = 5, allowed_account_ids: list = None) -> dict:
    """Owner-only: protect the authored structure volume, set interaction range, and replace additional permitted bank accounts. The owner remains permitted. Proximity uses an authoritative player or vehicle, never a camera. Omitted accounts means owner-only."""
    return infrastructure.configure_access(asset_id, interaction_radius_m, [] if allowed_account_ids is None else allowed_account_ids)
