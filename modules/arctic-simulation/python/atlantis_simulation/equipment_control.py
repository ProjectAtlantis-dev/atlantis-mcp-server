"""Shared, bank-authorized mechanism commands for MCP and connected viewers."""
from . import infrastructure_control as infrastructure
from .host import simulation_host
from atlantis_economy.gateway import account_for


def describe(principal, asset_id):
    asset = infrastructure.owned(principal, asset_id)
    snapshot = simulation_host.command('GET', infrastructure.path(principal, 'snapshot'))
    entity = next((e for e in snapshot['infrastructure'] if e['id'] == asset_id), None)
    if entity is None:
        raise ValueError('Place this bank-owned object before controlling its mechanisms')
    contract = simulation_host.command('POST', infrastructure.path(principal, 'equipment-contract'), {'modelId': asset['assetType']})
    return {'asset': asset, 'entity': entity, 'contract': contract}


def execute(principal, asset_id, mechanism_id, values, expected_revision, subject_kind=None, subject_id=None):
    asset = infrastructure.owned(principal, asset_id)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError('Current nonnegative mechanism revision required')
    if not isinstance(values, dict) or not values:
        raise ValueError('Provide the named control values from the mechanism contract')
    contract = simulation_host.command('POST', infrastructure.path(principal, 'equipment-contract'), {'modelId': asset['assetType']})
    mechanism = next((m for m in contract['mechanisms'] if m['id'] == mechanism_id), None)
    if mechanism is None:
        raise ValueError('Mechanism not supported by this model')
    if mechanism.get('physicalAccess'):
        allowed = infrastructure.access_options(principal, asset_id)['subjects']
        if not any(s['allowed'] and s['kind'] == subject_kind and s['id'] == subject_id for s in allowed):
            raise PermissionError('A nearby authorized physical subject is required')
    account = account_for(principal)
    return simulation_host.command('POST', infrastructure.path(principal, 'equipment-command'),
        {'id': asset_id, 'mechanismId': mechanism_id, 'values': values, 'expectedRevision': expected_revision,
         'accountId': account['id'], 'subjectKind': subject_kind, 'subjectId': subject_id})


def mobility_models():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parent / 'runtime/src/asset-mobility.json').read_text())


def attach_movement(principal, asset_id, water_level_m=None):
    import math
    from .mission_terrain import make_surface
    asset = infrastructure.owned(principal, asset_id)
    profile = mobility_models().get(asset['assetType'])
    if profile is None or asset.get('metadata', {}).get('defenseSiteId'):
        raise ValueError('This asset has no independent movement controller')
    snapshot = simulation_host.command('GET', infrastructure.path(principal, 'snapshot'))
    existing = next((v for v in snapshot['controlledVehicles'] if v['id'] == asset_id), None)
    if existing is not None:
        return {'attached': True, 'state': existing, 'alreadyAttached': True}
    entity = next((e for e in snapshot['infrastructure'] if e['id'] == asset_id), None)
    if entity is None:
        raise ValueError('Place the asset before attaching movement')
    origin = snapshot['origin']
    lat = origin['latitude'] + entity['position']['y']/6378137*180/math.pi
    lon = origin['longitude'] + entity['position']['x']/(6378137*math.cos(math.radians(origin['latitude'])))*180/math.pi
    altitude = entity['position']['z']+origin['altitudeM']
    if profile['domain'] == 'water':
        if type(water_level_m) not in (int, float) or not math.isfinite(water_level_m):
            raise ValueError('A finite water_level_m in the terrain vertical datum is required for boats')
        altitude = water_level_m
    elif water_level_m is not None:
        raise ValueError('water_level_m applies only to boats')
    state = {'definitionId': asset['assetType'], 'authority':'server-boat-v1' if profile['domain']=='water' else 'server-ground-v1',
             'position':{'x':0,'y':0,'z':altitude}, 'navigationFrame':{'origin':{'lat':lat,'lon':lon}}}
    surface = make_surface(principal.scenario, state, radius=64)
    payload = {'operation':'attach','id':asset_id,'terrainAssetId':asset_id,'definitionId':asset['assetType'],
               'ownerAccountId':asset['ownerAccountId'],'presentation':'infrastructure',
               'pose':{'x':0,'y':0,'headingRad':-math.radians(entity['headingDeg'])},
               'renderOffsetM': 0 if profile['domain']=='water' else altitude-surface['heights'][(surface['rows']//2)*surface['cols']+surface['cols']//2],
               'surface':surface,'sourcePose':{'lat':lat,'lon':lon,'z':altitude}}
    if profile['domain']=='water':
        payload['definition'] = {'boat':profile['boat']}
    result = simulation_host.command('POST', infrastructure.path(principal, 'vehicle-control'),payload)
    if result.get('error'):
        raise ValueError(result.get('message', result['error']))
    return {'attached':True, 'state':result, 'alreadyAttached':False}
