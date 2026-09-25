"""Model-specific, persistent mechanism controls shared with the viewer."""
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation import equipment_control


@visible
def model_controls(model_id: str) -> dict:
    """Read an original asset model's authored mechanism fields and movement support before placement. Use Terrain/Placement/assets for model IDs. This describes capabilities, not instance availability: after placement call Terrain/Objects/functions with the bank UUID for currently runnable actions and revisions. Empty mechanisms means no authored articulation, unless legacyDoors identifies the separate door controller.

    :param model_id: Original model ID from Terrain/Placement/assets; identifies a model type, not a placed bank UUID.
    """
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
    """Read this owned object's mechanisms, named fields, units, permitted ranges, current targets, actual poses and revisions. Commands are model-specific. Articulations are simulation mechanics, not real hardware performance or utility certification.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    """
    return equipment_control.describe(current_principal('simulation'), asset_id)


@visible
def set_controls(asset_id: str, mechanism_id: str, values: dict, expected_revision: int,
                 subject_kind: str = None, subject_id: str = None) -> dict:
    """Set one mechanism's named controls using inspect's fields and revision. Server validates ownership, limits and interlocks; protected doors additionally require a nearby authorized physical subject. Accepted means target set, not finished. Re-inspect actual poses. Use a bank UUID, never a model ID.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param mechanism_id: Exact mechanism ID returned by Equipment/inspect for this asset.
    :param values: Object mapping the selected mechanism field names to values. Use inspect for types, units, limits and interlocks; do not invent fields.
    :param expected_revision: Exact revision returned by the current inspection/discovery for the component being changed. Re-inspect after a conflict.
    :param subject_kind: Physical interaction subject kind: player or vehicle. Use an authorized nearby subject returned by object discovery.
    :param subject_id: Exact ID of the discovered physical subject; a bank UUID for vehicle subjects. Never use the camera position as a subject.
    """
    return equipment_control.execute(current_principal('simulation'), asset_id, mechanism_id,
                                     values, expected_revision, subject_kind, subject_id)


@visible
def index() -> dict:
    """Discover server-owned asset mechanisms and issue controls using the bank asset UUID."""
    return {'commands': ['model_controls', 'inspect', 'set_controls', 'attach_movement', 'instructions'], 'state': 'server-owned per asset UUID'}


@visible
def attach_movement(asset_id: str, water_level_m: float = None) -> dict:
    """Attach the implemented ground or water controller to a placed mobile asset while keeping its bank UUID. Then use Terrain/Vehicles drive_to/sail_to, mission_status and mission_control. Boats require verified water and an explicit water_level_m in the terrain vertical datum. Static structures and site-bound defense components cannot be driven independently.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param water_level_m: Verified water-surface elevation in metres in the terrain vertical datum; not water depth or altitude above ground.
    """
    return equipment_control.attach_movement(current_principal('simulation'), asset_id, water_level_m)


@visible
def instructions() -> str:
    """Read asset controls, units, state ownership and connected workshop instructions."""
    from pathlib import Path
    return Path(__file__).with_name("README.md").read_text()
