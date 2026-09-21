"""Shared owner-authorized gateway for map-picked fictional incoming commands."""
import math
from urllib.parse import quote
import atlantis
from atlantis_simulation import terrain_adapter
from atlantis_simulation.host import simulation_host

SCENARIO_ID = 'defense-demo'


def authorize(principal):
    if 'simulation' not in principal.permissions or principal.caller not in atlantis.get_owner_usernames():
        raise PermissionError('Authenticated host owner required for defense scenario controls')


def spawn(principal, *, incoming_type, request_id, latitude, longitude, heading_deg=90,
          approach_distance_m=5000, altitude_m=300, speed_mps=70):
    authorize(principal)
    if not isinstance(request_id,str) or not request_id.strip() or len(request_id)>100:
        raise ValueError('request_id must contain 1..100 characters')
    for name, value, low, high in [('latitude',latitude,-85,85),('longitude',longitude,-180,180),
        ('heading_deg',heading_deg,0,360),('approach_distance_m',approach_distance_m,10,100000),
        ('altitude_m',altitude_m,10,20000),('speed_mps',speed_mps,1,3000)]:
        if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} must be finite and between {low} and {high}')
    if heading_deg == 360:
        raise ValueError('heading_deg must be less than 360')
    if incoming_type not in ('drone','cruise','ballistic'):
        raise ValueError('incoming_type must be drone, cruise or ballistic')
    path=f'/games/{quote(principal.scenario,safe="")}/'
    snapshot=simulation_host.command('GET',path+'snapshot');origin=snapshot['origin']
    config=terrain_adapter.configuration(principal.scenario)
    grid=terrain_adapter.elevation_grid(config,{'lat':latitude,'lon':longitude},radius=2,step=2)
    ground=grid['heights'][4]
    if not isinstance(ground,(int,float)) or not math.isfinite(ground):
        raise ValueError('Selected destination has no verified terrain elevation')
    destination={'x':math.radians(longitude-origin['longitude'])*6378137*math.cos(math.radians(origin['latitude'])),
                 'y':math.radians(latitude-origin['latitude'])*6378137,'z':ground-origin['altitudeM']}
    return simulation_host.command('POST',path+'scenario-incoming',{
        'incomingType':incoming_type,'requestId':request_id,'destination':destination,
        'destinationCoordinates':{'latitude':latitude,'longitude':longitude},'headingDeg':heading_deg,
        'approachDistanceM':approach_distance_m,'altitudeM':altitude_m,'speedMps':speed_mps})


def asset_status(principal, asset_id):
    from atlantis_simulation import infrastructure_control
    asset = infrastructure_control.owned(principal, asset_id)
    metadata = asset.get('metadata', {})
    site_id, role = metadata.get('defenseSiteId'), metadata.get('defenseRole')
    if not site_id or not role:
        raise ValueError('This asset is a standalone display, not a defense-site component')
    state = simulation_host.command('GET', f'/games/{quote(principal.scenario, safe="")}/defense-observation')
    layer_id = role.removeprefix('launcher-') if role.startswith('launcher-') else None
    targets = [track for track in state['tracks'] if any(action['siteId'] == site_id
               and action['layerId'] == layer_id for action in track['availableActions'])] if layer_id else []
    return {'asset_id': asset_id, 'site_id': site_id, 'role': role, 'layer_id': layer_id,
            'tick': state['tick'], 'mode': state['mode'],
            'sensors': [sensor for sensor in state['sensors'] if sensor['siteId'] == site_id],
            'layers': [layer for layer in state['layers'] if layer['siteId'] == site_id],
            'available_targets': targets}


def intercept_asset(principal, asset_id, target_id):
    binding = asset_status(principal, asset_id)
    if not binding['layer_id'] or not any(track['id'] == target_id for track in binding['available_targets']):
        raise ValueError('No currently runnable interception for this launcher and target')
    return simulation_host.command('POST', f'/games/{quote(principal.scenario, safe="")}/intercept',
        {'targetId': target_id, 'siteId': binding['site_id'], 'layerId': binding['layer_id'], 'requireTracked': True})
