"""Model-specific, persistent mechanism controls shared with the viewer."""
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation import equipment_control


@visible
def model_controls(model_id: str) -> dict:
    """Read an original asset model's authored mechanism fields and movement support before placement. Use Terrain/Placement/assets for model IDs. This describes capabilities, not instance availability: after placement call Terrain/Objects/functions with the bank UUID for currently runnable actions and revisions. Empty mechanisms means no authored articulation, unless legacyDoors identifies the separate door controller."""
    from atlantis_simulation import infrastructure_control
    from atlantis_simulation.host import simulation_host
    principal = current_principal('simulation')
    contract = simulation_host.command('POST', infrastructure_control.path(principal, 'equipment-contract'),
                                       {'modelId': model_id})
    return {'modelId': model_id, 'contract': contract,
            'movement': equipment_control.mobility_models().get(model_id),
            'instanceDiscovery': 'Terrain/Objects/functions'}


@visible
def inspect(asset_id: str) -> dict:
    """Read this owned object's mechanisms, named fields, units, permitted ranges, current targets, actual poses and revisions. Commands are model-specific. Articulations are simulation mechanics, not real hardware performance or utility certification."""
    return equipment_control.describe(current_principal('simulation'), asset_id)


@visible
def set_controls(asset_id: str, mechanism_id: str, values: dict, expected_revision: int,
                 subject_kind: str = None, subject_id: str = None) -> dict:
    """Set one mechanism's named controls using inspect's fields and revision. Server validates ownership, limits and interlocks; protected doors additionally require a nearby authorized physical subject. Accepted means target set, not finished. Re-inspect actual poses. Use a bank UUID, never a model ID."""
    return equipment_control.execute(current_principal('simulation'), asset_id, mechanism_id,
                                     values, expected_revision, subject_kind, subject_id)


@visible
def index() -> dict:
    """Discover server-owned asset mechanisms and issue controls using the bank asset UUID."""
    return {'commands': ['model_controls', 'inspect', 'set_controls', 'attach_movement', 'instructions'], 'state': 'server-owned per asset UUID'}


@visible
def attach_movement(asset_id: str, water_level_m: float = None) -> dict:
    """Attach the implemented ground or water controller to a placed mobile asset while keeping its bank UUID. Then use Terrain/Vehicles drive_to/sail_to, mission_status and mission_control. Boats require verified water and an explicit water_level_m in the terrain vertical datum. Static structures and site-bound defense components cannot be driven independently."""
    return equipment_control.attach_movement(current_principal('simulation'), asset_id, water_level_m)


@visible
def instructions() -> str:
    """Read asset controls, units, state ownership and connected workshop instructions."""
    from pathlib import Path
    return Path(__file__).with_name("README.md").read_text()
