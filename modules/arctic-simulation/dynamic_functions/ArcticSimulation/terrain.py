from urllib.parse import quote, urlencode
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation.terrain_adapter import attachment, configuration
from atlantis_simulation.vehicle_control import owned_asset
from atlantis_simulation.host import simulation_host
from atlantis_simulation.viewer import capabilities


def _binding(asset_id):
    principal = current_principal()
    asset = owned_asset(principal, asset_id)
    binding = attachment(principal.scenario, asset_id)
    if binding['definitionId'] != asset['assetType']:
        raise ValueError('Existing terrain model and bank UUID asset type disagree')
    return principal, asset, binding


@visible
def terrain_vehicle_plan(asset_id: str) -> dict:
    """Read-only check of existing terrain instance -> owned bank UUID and real DEM coverage; reports saved altitude discrepancy."""
    principal, asset, binding = _binding(asset_id)
    surface = binding['surface']
    ground = surface['heights'][(surface['rows']//2)*surface['cols']+surface['cols']//2]
    return {'assetId':asset_id,'terrainAssetId':binding['terrainAssetId'],'definitionId':binding['definitionId'],
            'sourcePose':binding['sourcePose'],'groundElevationM':ground,
            'savedHeightDifferenceM':binding['sourcePose']['z']-ground,'sources':surface['sources'],
            'coverageRadiusM':-surface['minX'],'sourceDatabasesModified':False}


@visible
def terrain_vehicle_attach(asset_id: str, allow_ground_snap: bool = False) -> dict:
    """Bind an existing Terrain instance to its owned bank UUID, loading real EGM2008 DEM. Explicitly acknowledge altitude mismatch over 2m. Reattachment preserves server pose."""
    principal, asset, binding = _binding(asset_id)
    surface = binding['surface']
    ground = surface['heights'][(surface['rows']//2)*surface['cols']+surface['cols']//2]
    if abs(binding['sourcePose']['z']-ground)>2 and allow_ground_snap is not True:
        raise ValueError('Saved vehicle elevation differs from current DEM by more than 2m; inspect terrain_vehicle_plan before explicitly allowing ground snap')
    payload = dict(binding, operation='attach', id=asset_id, ownerAccountId=asset['ownerAccountId'],
                   actor=f'mcp:{principal.external_user_id}:{principal.user_game_id}')
    return simulation_host.command('POST',f'/games/{quote(principal.scenario,safe="")}/vehicle-control',payload)


@visible
def terrain_vehicle_viewer_access(asset_id: str, ttl_seconds: int = 300) -> dict:
    """Open this bank-owned vehicle through NORMAL terrain viewer startup, with the existing full vehicle catalog and authoritative poses."""
    principal = current_principal()
    owned_asset(principal, asset_id)
    config = configuration(principal.scenario)
    if asset_id not in config['vehicles']:
        raise ValueError('Vehicle has no terrain binding')
    state = simulation_host.command('POST',f'/games/{quote(principal.scenario,safe="")}/vehicle-control',{'operation':'observe','id':asset_id})
    if state['terrainAssetId'] != config['vehicles'][asset_id]['terrainAssetId']:
        raise ValueError('Vehicle terrain identity conflict')
    grant = capabilities.issue(principal, ttl_seconds, control_vehicle_id=asset_id)
    grant['viewerFragment'] = urlencode({'simulation_game':principal.scenario,'simulation_token':grant['token'],
                                         'simulation_vehicle':asset_id,'simulation_terrain':'1'})
    return grant
