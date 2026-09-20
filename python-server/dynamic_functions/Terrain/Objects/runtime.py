"""Object function discovery and invocation shared by MCP and the viewer."""
from urllib.parse import quote
from atlantis_economy.gateway import account_for, bank_request, canonical_uuid
from atlantis_simulation.host import simulation_host
from atlantis_simulation.vehicle_control import execute as vehicle_execute
from atlantis_simulation import infrastructure_control


def _owned(principal, asset_id):
    canonical_uuid(asset_id)
    if 'simulation' not in principal.permissions:
        raise PermissionError('Simulation permission required')
    account = account_for(principal)
    result = bank_request('GET', f'/assets/{asset_id}/verify')
    asset = result.get('asset') or {}
    if (not result.get('authentic') or not result.get('spendable')
            or asset.get('ownerAccountId') != account['id']
            or asset.get('metadata', {}).get('world') != principal.scenario):
        raise PermissionError('An active object owned by you in this world is required')
    return asset


def _snapshot(principal):
    return simulation_host.command('GET', f'/games/{quote(principal.scenario, safe="")}/snapshot')


def field(name, kind='number', required=False, default=None):
    value = {'name': name, 'type': kind, 'required': required}
    if default is not None:
        value['default'] = default
    return value


def action(identifier, function, label, parameters=(), bound=None):
    return {'id': identifier, 'function': function, 'label': label,
            'parameters': list(parameters), 'bound': bound or {}}


def component_action_available(component, command):
    entry = component.get('entry', component)
    outer_closed = entry['outer'] == 0 and entry['target']['outer'] == 0
    inner_closed = entry['inner'] == 0 and entry['target']['inner'] == 0
    freight_closed = component.get('freight', 0) == 0 and component.get('freightTarget', 0) == 0
    available = {
        'airlock_close': True, 'airlock_open_outer': inner_closed, 'airlock_open_inner': outer_closed,
        'facility_entry_close': True, 'facility_freight_close': True,
        'facility_entry_outer': inner_closed and freight_closed,
        'facility_entry_inner': outer_closed and freight_closed,
        'facility_freight_open': outer_closed and inner_closed,
    }
    return available.get(command, False)


def describe(principal, asset_id):
    asset = _owned(principal, asset_id)
    snapshot = _snapshot(principal)
    result = {'commandPrefix': f'%{principal.caller}/**/', 'assetId': asset_id, 'model': asset['assetType'], 'kind': asset['kind'],
              'name': asset.get('metadata', {}).get('terrainAssetId', asset['assetType']),
              'actions': [action('inspect', 'Terrain/Objects/inspect', 'Inspect object')], 'state': None}
    if asset['kind'] == 'vehicle':
        state = next((v for v in snapshot.get('controlledVehicles', []) if v['id'] == asset_id), None)
        result['state'] = state
        if state is None:
            result['message'] = 'No server controller is attached. Register and attach a supported controller through Terrain/Vehicles before sending movement commands.'
            return result
        caps = vehicle_execute(principal, 'capabilities', asset_id, actor='object-discovery')
        current_mission = state.get('mission')
        can_dispatch = not current_mission or current_mission['status'] in ('completed', 'cancelled', 'failed')
        can_dispatch = can_dispatch and not state.get('controlled')
        for command in caps['actions'] if can_dispatch else []:
            if command['id'] not in ('drive_to', 'fly_to'):
                continue
            params = [field('latitude', required=True), field('longitude', required=True),
                      field('request_id', 'string', True), field('return_latitude'), field('return_longitude'),
                      field('wait_for_task', 'boolean', default=False)]
            if command['id'] == 'fly_to':
                params += [field('altitude_agl_m', default=60), field('land', 'boolean', default=False),
                           field('return_altitude_agl_m', default=60), field('return_land', 'boolean', default=False)]
            result['actions'].append(action(command['id'], 'Terrain/Vehicles/' + command['id'],
                                            'Fly to coordinates' if command['id'] == 'fly_to' else 'Drive to coordinates', params))
        result['actions'].append(action('mission_status', 'Terrain/Vehicles/mission_status', 'Read mission status'))
        mission = state.get('mission')
        if mission and mission['status'] not in ('completed', 'cancelled', 'failed'):
            for command in caps['missionActions']:
                if command == 'complete_task' and mission['status'] != 'awaiting_task':
                    continue
                if command == 'resume' and mission['status'] not in ('paused', 'blocked'):
                    continue
                if command == 'pause' and mission['status'] == 'paused':
                    continue
                result['actions'].append(action(command,
                    'Terrain/Vehicles/complete_task' if command == 'complete_task' else 'Terrain/Vehicles/mission_control',
                    'Confirm task completed' if command == 'complete_task' else command.title() + ' mission',
                    bound={'mission_id': mission['id'], **({} if command == 'complete_task' else {'action': command})}))
    elif asset['kind'] == 'structure':
        infrastructure_control.owned(principal, asset_id)
        state = next((v for v in snapshot.get('infrastructure', []) if v['id'] == asset_id), None)
        result['state'] = state
        component = state.get('componentState') if state else None
        if component:
            contract = simulation_host.command('GET', infrastructure_control.path(principal, 'component-contract'))
            if component['schema'] == 'facility-access-v1':
                contract = contract['facilities']
            for command, terms in contract['actions'].items():
                if not component_action_available(component, command):
                    continue
                descriptor = action(command, 'Terrain/Infrastructure/component_command', command.replace('_', ' ').title(),
                                    bound={'action': command, 'expected_revision': component['revision']})
                descriptor['requires'] = terms.get('requires', [])
                result['actions'].append(descriptor)
    return result


def invoke(principal, asset_id, action_id, parameters):
    if not isinstance(parameters, dict):
        raise ValueError('Function parameters must be an object')
    description = describe(principal, asset_id)
    descriptor = next((a for a in description['actions'] if a['id'] == action_id), None)
    if descriptor is None:
        raise ValueError('Function is not available for this object in its current state')
    expected = {p['name'] for p in descriptor['parameters']} | set(descriptor['bound'])
    if set(parameters) - expected:
        raise ValueError('Unknown function parameter')
    args = {p['name']: p['default'] for p in descriptor['parameters'] if 'default' in p}
    args.update(parameters)
    for param in descriptor['parameters']:
        if param['required'] and param['name'] not in args:
            raise ValueError(f"Missing {param['name']}")
    # Bound mission IDs and component revisions must come from the selected form.
    # Reject stale forms rather than silently directing an action at newer state.
    for key, value in descriptor['bound'].items():
        if parameters.get(key) != value:
            raise ValueError('Object state changed; refresh its available functions')
    if action_id == 'inspect':
        return {'assetId': asset_id, 'model': description['model'], 'kind': description['kind'],
                'state': description['state'], 'message': description.get('message')}
    if description['kind'] == 'vehicle':
        payload = {}
        operation = action_id
        if action_id in ('drive_to', 'fly_to'):
            payload = {'destination': {'lat': args['latitude'], 'lon': args['longitude']},
                       'requestId': args['request_id'], 'waitForTask': args['wait_for_task']}
            lat, lon = args.get('return_latitude'), args.get('return_longitude')
            if (lat is None) != (lon is None):
                raise ValueError('Supply both return coordinates')
            if lat is not None:
                payload['returnDestination'] = {'lat': lat, 'lon': lon}
            if action_id == 'fly_to':
                payload.update(altitudeAglM=args['altitude_agl_m'], landing=args['land'])
                if lat is not None:
                    payload['returnDestination'].update(altitudeAglM=args['return_altitude_agl_m'], landing=args['return_land'])
        elif action_id in ('pause', 'resume', 'cancel', 'complete_task'):
            operation = 'mission_control'
            payload = {'missionId': args['mission_id'], 'action': action_id}
        return vehicle_execute(principal, operation, asset_id,
                               actor=f'object-functions:{principal.external_user_id}:{principal.user_game_id}', parameters=payload)
    if description['kind'] == 'structure':
        return simulation_host.command('POST', infrastructure_control.path(principal, 'component-command'),
            {'id': asset_id, 'action': args['action'], 'expectedRevision': args['expected_revision']})
    raise ValueError('No callable function for this object')
