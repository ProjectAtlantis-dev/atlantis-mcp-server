"""Scene authoring through the same bank and simulation authority as MCP."""
import math
from urllib.parse import quote
import atlantis
from atlantis_simulation import infrastructure_control, terrain_adapter, equipment_control
from atlantis_simulation.host import simulation_host
from atlantis_economy.gateway import account_for, bank_request, canonical_uuid

CONSOLE_ID = 'scene-placement'


def authorize(principal):
    if not {'simulation', 'bank'} <= set(principal.permissions) or principal.caller not in atlantis.get_owner_usernames():
        raise PermissionError('Authenticated host owner with simulation and bank permissions required')


def path(principal, action):
    return f'/games/{quote(principal.scenario, safe="")}/{action}'


def assets(principal):
    authorize(principal)
    return simulation_host.command('GET', path(principal, 'infrastructure-catalog'))


def ground_position(principal, latitude, longitude):
    for name, value, limit in [('latitude', latitude, 85), ('longitude', longitude, 180)]:
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
            raise ValueError(f'Invalid {name}')
    origin = simulation_host.command('GET', path(principal, 'snapshot'))['origin']
    grid = terrain_adapter.elevation_grid(terrain_adapter.configuration(principal.scenario),
                                         {'lat': latitude, 'lon': longitude}, radius=2, step=2)
    ground = grid['heights'][4]
    if type(ground) not in (int, float) or not math.isfinite(ground):
        raise ValueError('Placement requires verified terrain elevation')
    return {'x': math.radians(longitude-origin['longitude'])*6378137*math.cos(math.radians(origin['latitude'])),
            'y': math.radians(latitude-origin['latitude'])*6378137, 'z': ground-origin['altitudeM']}


def place_model(principal, model_id, instance_key, latitude, longitude, heading_deg=0, height_offset_m=0):
    models = assets(principal)['models']
    model = next((m for m in models if m['id'] == model_id), None)
    if model is None:
        raise ValueError('Unknown asset model')
    for value in (heading_deg, height_offset_m):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Heading and height offset must be finite')
    if model['upAxis'] != 'Y':
        raise ValueError('Placement requires a Y-up asset model')
    position = ground_position(principal, latitude, longitude)
    position['z'] += height_offset_m-model['bounds']['min'][1]
    placed = infrastructure_control.place_for(principal, model_id, position, heading_deg, instance_key)
    mobility = equipment_control.mobility_models().get(model_id)
    if mobility and mobility['domain'] == 'ground':
        # Reconcile under the issued UUID on retries; never create a second
        # vehicle identity or leave successful placement without its controller.
        placed['movement'] = equipment_control.attach_movement(principal, placed['bankAssetId'])
    return placed


def deploy_demo_site(principal, site_id, layer_id, latitude, longitude):
    authorize(principal)
    if not isinstance(site_id, str) or not site_id.strip() or len(site_id) > 100:
        raise ValueError('A unique site_id of 1..100 characters is required')
    if layer_id not in ('upper-tier', 'middle-tier', 'point-defense', 'directed-energy'):
        raise ValueError('Unknown demo layer')
    position = ground_position(principal, latitude, longitude)
    account = account_for(principal)
    models = {'radar': 'defense-radar', 'command': 'support-command',
              'resupply': 'support-resupply', 'recovery': 'support-recovery',
              'launcher-' + layer_id: {'upper-tier': 'defense-upper-tier',
                  'middle-tier': 'defense-middle-tier', 'point-defense': 'defense-point-defense',
                  'directed-energy': 'defense-laser'}[layer_id]}
    intent = {'siteId': site_id, 'layerId': layer_id, 'latitude': latitude, 'longitude': longitude}
    snapshot = simulation_host.command('GET', path(principal, 'snapshot'))
    existing = next((site for site in snapshot['sites'] if site['id'] == site_id), None)
    if existing and (existing.get('bankAssets') is None or existing.get('placementIntent') != intent):
        raise ValueError('Site ID already exists with different terms or is an unregistered legacy site')
    assets = {}
    for role, model in models.items():
        asset = bank_request('POST', '/assets/issue', {
            'kind': 'structure', 'assetType': model, 'ownerAccountId': account['id'],
            'actorAccountId': account['id'], 'sourceNamespace': 'defense:' + principal.scenario,
            'sourceId': site_id + ':' + role,
            'metadata': {'world': principal.scenario, 'stateAuthority': 'arctic-simulation',
                         'defenseSiteId': site_id, 'defenseRole': role, 'placementIntent': intent}})
        if (asset['ownerAccountId'] != account['id'] or asset['assetType'] != model
                or asset.get('metadata', {}).get('placementIntent') != intent):
            raise ValueError('Site instance key already belongs to different issuance terms')
        assets[role] = {'id': canonical_uuid(asset['id']), 'modelId': model,
                        'ownerAccountId': account['id'], 'ownerUsername': principal.caller}
    return simulation_host.command('POST', path(principal, 'sites'),
                                   {'id': site_id, 'layerIds': [layer_id], 'position': position,
                                    'bankAssets': assets, 'placementIntent': intent})
