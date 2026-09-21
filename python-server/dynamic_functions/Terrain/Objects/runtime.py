"""Object function discovery and invocation shared by MCP and the viewer."""
from urllib.parse import quote
import atlantis
from atlantis_economy.gateway import account_for, bank_request, canonical_uuid
from atlantis_simulation.host import simulation_host
from atlantis_simulation.vehicle_control import execute as vehicle_execute
from atlantis_simulation import infrastructure_control, equipment_control
from dynamic_functions.Terrain.Defense import gateway as defense_gateway
from dynamic_functions.Terrain.Placement import gateway as placement_gateway


def _owned(principal, asset_id):
    canonical_uuid(asset_id)
    if 'simulation' not in principal.permissions:
        raise PermissionError('Simulation permission required')
    account = account_for(principal)
    result = bank_request('GET', f'/assets/{asset_id}/verify')
    asset = result.get('asset') or {}
    if (not result.get('authentic') or not result.get('spendable')
            or (asset.get('ownerAccountId') != account['id'] and asset.get('kind') != 'structure')
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


def command_prefix():
    info = atlantis.get_server_info()
    owner, remote = info["owner"], info["remote_name"]
    if any(not isinstance(v, str) or not v or any(c.isspace() or c in "/%*?" for c in v)
           for v in (owner, remote)):
        raise RuntimeError("Exact server owner and remote name required for copied MCP commands")
    return f"@/{owner}/{remote}/"


def describe(principal, asset_id):
    if asset_id == placement_gateway.CONSOLE_ID:
        models = placement_gateway.assets(principal)['models']
        model = field('model_id', 'string', True)
        model['choices'] = [m['id'] for m in models]
        model['choiceLabels'] = {m['id']: m['label'] for m in models}
        layer = field('layer_id', 'string', True, 'point-defense')
        layer['choices'] = ['upper-tier', 'middle-tier', 'point-defense', 'directed-energy']
        descriptors = [
            action('placement_instructions', 'Terrain/Placement/instructions', 'Read placement recipe'),
            action('place_model', 'Terrain/Placement/place_model', 'Place a vehicle or infrastructure asset', [
                model, field('instance_key', 'string', True), field('latitude', required=True),
                field('longitude', required=True), field('heading_deg', default=0), field('height_offset_m', default=0)]),
            action('deploy_demo_site', 'Terrain/Placement/deploy_demo_site', 'Deploy demo interceptor site', [
                field('site_id', 'string', True), layer, field('latitude', required=True), field('longitude', required=True)])]
        for descriptor in descriptors:
            descriptor['includeAssetId'] = False
        return {'commandPrefix': command_prefix(), 'assetId': asset_id, 'kind': 'placement',
                'model': asset_id, 'name': 'Place assets', 'state': None, 'actions': descriptors,
                'message': str(len(models)) + ' authored assets available. This list is for placing new instances, not your current fleet. Each model and demo-site component gets its own bank UUID. New instance/site keys create additional copies. Pick a point or enter coordinates. Guide: Terrain/Placement/instructions.'}
    if asset_id == defense_gateway.SCENARIO_ID:
        defense_gateway.authorize(principal)
        kind=field('incoming_type','string',True,'drone');kind['choices']=['drone','cruise','ballistic']
        descriptor=action('spawn_incoming','Terrain/Defense/spawn_incoming','Send test incoming',[
            kind,field('latitude',required=True),field('longitude',required=True),
            field('request_id','string',True),field('heading_deg',default=90),
            field('approach_distance_m',default=5000),field('altitude_m',default=300),field('speed_mps',default=70)])
        descriptor['includeAssetId']=False
        return {'commandPrefix':command_prefix(),'assetId':asset_id,'kind':'scenario','model':'defense-demo',
                'name':'Send test incoming','state':None,'actions':[descriptor],
                'message':'Choose incoming type and heading, pick destination on the map or an object, then Run. Radar alerts and intercept authorization are separate.'}
    asset = _owned(principal, asset_id)
    snapshot = _snapshot(principal)
    result = {'commandPrefix': command_prefix(), 'assetId': asset_id, 'model': asset['assetType'], 'kind': asset['kind'],
              'name': asset.get('metadata', {}).get('terrainAssetId', asset['assetType']),
              'actions': [action('inspect', 'Terrain/Objects/inspect', 'Inspect object')], 'state': None}
    mobile_state = next((v for v in snapshot.get('controlledVehicles', []) if v['id'] == asset_id
                         and (asset['kind'] == 'vehicle' or (v.get('presentation') == 'infrastructure'
                              and v.get('definitionId') == asset['assetType']))), None)
    result['controller'] = mobile_state is not None
    if asset['kind'] == 'vehicle' or mobile_state is not None:
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
            if command['id'] not in ('drive_to', 'sail_to', 'fly_to'):
                continue
            params = [field('latitude', required=True), field('longitude', required=True),
                      field('request_id', 'string', True), field('return_latitude'), field('return_longitude'),
                      field('wait_for_task', 'boolean', default=False)]
            if caps.get('autoReturn', True):
                params.append(field('auto_return', 'boolean', default=False))
            if command['id'] == 'fly_to':
                params += [field('altitude_agl_m', default=60), field('return_altitude_agl_m', default=60)]
                params.append(field('water_level_m'))
                if caps.get('landing', True):
                    params += [field('land', 'boolean', default=False), field('return_land', 'boolean', default=False)]
                if caps.get('takeoffHeading'):
                    params.append(field('takeoff_heading_deg', default=state['headingRad'] * 180 / 3.141592653589793))
            result['actions'].append(action(command['id'], 'Terrain/Vehicles/' + command['id'],
                                            command['label'], params))
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
    if asset['kind'] == 'structure':
        infrastructure_control.verified(principal, asset_id)
        state = next((v for v in snapshot.get('infrastructure', []) if v['id'] == asset_id), None)
        result['state'] = state
        if state and state.get('label'):
            result['name'] = state['label']
        component = state.get('componentState') if state else None
        if state and not mobile_state and asset['assetType'] in equipment_control.mobility_models() and not asset.get('metadata', {}).get('defenseSiteId'):
            parameters = [field('water_level_m', required=True)] if equipment_control.mobility_models()[asset['assetType']]['domain'] == 'water' else []
            result['actions'].append(action('attach_movement','Terrain/Equipment/attach_movement','Enable movement controller', parameters))
        if state is None:
            result['message'] = 'Registered structure has no simulation placement.'
        elif state.get('simulationScope') == 'defense-site-component':
            result['message'] = ('Defense site: ' + state['siteId'] + '. Role: ' + asset.get('metadata', {}).get('defenseRole', asset['assetType']) + '. State: ' + state['operationalState']['status'] + '. Selecting this object does not fire it; site interception uses Terrain/Defense functions.')
        elif component is None:
            mechanisms = state.get('equipmentState', {}).get('mechanisms', {})
            result['message'] = (str(len(mechanisms)) + ' authored mechanism groups. Controls below set server-owned poses.' if mechanisms else 'Static display; no mechanism controls are implemented for this model.')
            if asset['assetType'].startswith('defense-'):
                result['message'] += ' Standalone display: not connected to a defense site or interception.'
        if state and state.get('simulationScope') == 'defense-site-component':
            binding = defense_gateway.asset_status(principal, asset_id)
            result['actions'].append(action('defense_status', 'Terrain/Defense/asset_status', 'Read site and radar status'))
            if binding['available_targets']:
                target = field('target_id', 'string', True)
                target['choices'] = [track['id'] for track in binding['available_targets']]
                target['choiceLabels'] = {track['id']: track['label'] + ' (' + track['kind'] + ')' for track in binding['available_targets']}
                result['actions'].append(action('intercept_asset', 'Terrain/Defense/intercept_asset', 'Intercept tracked demo target', [target]))
        if state and state.get('equipmentState'):
            contract = simulation_host.command('POST', infrastructure_control.path(principal, 'equipment-contract'), {'modelId': asset['assetType']})
            physical = None
            for mechanism in contract['mechanisms']:
                if mobile_state and mechanism['kind'] in ('tracks', 'propulsion'):
                    continue
                current = state['equipmentState']['mechanisms'][mechanism['id']]
                parameters = []
                for name, spec in mechanism['fields'].items():
                    parameter = field(name, spec['type'], True, current['targets'][name])
                    parameter.update({key: spec[key] for key in ('min', 'max', 'unit') if key in spec})
                    parameters.append(parameter)
                subjects = [None]
                if mechanism.get('physicalAccess'):
                    if physical is None:
                        physical = infrastructure_control.access_options(principal, asset_id)['subjects']
                    subjects = [subject for subject in physical if subject['allowed']]
                for subject in subjects:
                    bound = {'mechanism_id': mechanism['id'], 'expected_revision': current['revision']}
                    suffix = ''
                    if subject:
                        bound.update(subject_kind=subject['kind'], subject_id=subject['id'])
                        suffix = ':' + subject['kind'] + ':' + subject['id']
                    descriptor = action('equipment:' + mechanism['id'] + suffix, 'Terrain/Equipment/set_controls',
                        mechanism.get('label', mechanism['id']).replace('-', ' '), parameters, bound)
                    descriptor['parameterObject'] = 'values'
                    result['actions'].append(descriptor)
        if component:
            contract = simulation_host.command('GET', infrastructure_control.path(principal, 'component-contract'))
            if component['schema'] == 'facility-access-v1':
                contract = contract['facilities']
            access = infrastructure_control.access_options(principal, asset_id)
            subjects = [subject for subject in access['subjects'] if subject['allowed']]
            result['access'] = access
            if not subjects:
                result['message'] = ('Access policy is not commissioned.' if not access['protected'] else
                                     'No authorized player or vehicle is within interaction range. Camera position does not grant access.')
            for subject in subjects:
                for command, terms in contract['actions'].items():
                    if not component_action_available(component, command):
                        continue
                    descriptor = action(command + ':' + subject['kind'] + ':' + subject['id'],
                        'Terrain/Infrastructure/component_command',
                        command.replace('_', ' ').title() + ' · ' + subject['label'],
                        bound={'action': command, 'expected_revision': component['revision'],
                               'subject_kind': subject['kind'], 'subject_id': subject['id']})
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
    if action_id == 'defense_status':
        return defense_gateway.asset_status(principal, asset_id)
    if action_id == 'intercept_asset':
        return defense_gateway.intercept_asset(principal, asset_id, args['target_id'])
    if action_id == 'attach_movement':
        return equipment_control.attach_movement(principal, asset_id, parameters.get('water_level_m'))
    if action_id.startswith('equipment:'):
        values = {parameter['name']: args[parameter['name']] for parameter in descriptor['parameters']}
        return equipment_control.execute(principal, asset_id, args['mechanism_id'], values,
            args['expected_revision'], args.get('subject_kind'), args.get('subject_id'))
    if action_id == 'inspect':
        return {'assetId': asset_id, 'model': description['model'], 'kind': description['kind'],
                'state': description['state'], 'message': description.get('message')}
    if description['kind'] == 'placement':
        if action_id == 'placement_instructions':
            from pathlib import Path
            return {'instructions': Path(placement_gateway.__file__).with_name('README.md').read_text()}
        if action_id == 'place_model':
            return placement_gateway.place_model(principal, **args)
        if action_id == 'deploy_demo_site':
            return placement_gateway.deploy_demo_site(principal, **args)
        raise ValueError('Unknown placement action')
    if description['kind'] == 'scenario':
        return defense_gateway.spawn(principal, **args)
    if description['kind'] == 'vehicle' or description.get('controller'):
        payload = {}
        operation = action_id
        if action_id in ('drive_to', 'sail_to', 'fly_to'):
            payload = {'destination': {'lat': args['latitude'], 'lon': args['longitude']},
                       'requestId': args['request_id'], 'waitForTask': args['wait_for_task'], 'autoReturn': args.get('auto_return', False)}
            lat, lon = args.get('return_latitude'), args.get('return_longitude')
            if (lat is None) != (lon is None):
                raise ValueError('Supply both return coordinates')
            if args.get('auto_return', False) and lat is not None:
                raise ValueError('auto_return cannot be combined with explicit return coordinates')
            if lat is not None:
                payload['returnDestination'] = {'lat': lat, 'lon': lon}
            if action_id == 'fly_to':
                payload.update(altitudeAglM=args['altitude_agl_m'], landing=args.get('land', False))
                if 'water_level_m' in args:
                    payload['waterLevelM'] = args['water_level_m']
                if 'takeoff_heading_deg' in args:
                    payload['takeoffHeadingDeg'] = args['takeoff_heading_deg']
                if lat is not None:
                    payload['returnDestination'].update(altitudeAglM=args['return_altitude_agl_m'], landing=args.get('return_land', False))
        elif action_id in ('pause', 'resume', 'cancel', 'complete_task'):
            operation = 'mission_control'
            payload = {'missionId': args['mission_id'], 'action': action_id}
        return vehicle_execute(principal, operation, asset_id,
                               actor=f'object-functions:{principal.external_user_id}:{principal.user_game_id}', parameters=payload)
    if description['kind'] == 'structure':
        return infrastructure_control.execute_component(principal, asset_id, args['action'], args['expected_revision'],
                                                        args['subject_kind'], args['subject_id'])
    raise ValueError('No callable function for this object')
