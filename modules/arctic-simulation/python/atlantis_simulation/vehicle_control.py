"""Shared MCP/browser command gateway. Bank ownership is checked for every action."""
import json
import os
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from atlantis_host_adapters.identity import current_principal
from atlantis_economy.gateway import bank_request, account_for
from .host import simulation_host


class VehicleCommandRejected(RuntimeError):
    """A validated command was refused by the vehicle state machine."""


def owned_asset(principal, asset_id):
    if str(UUID(asset_id)) != asset_id:
        raise ValueError('canonical lowercase asset UUID required')
    account = account_for(principal, request=bank_request)
    verified = bank_request('GET', f'/assets/{asset_id}/verify')
    asset = verified.get('asset') or {}
    if (not verified.get('authentic') or not verified.get('spendable') or asset.get('kind') != 'vehicle'
            or asset.get('ownerAccountId') != account['id']
            or asset.get('metadata', {}).get('world') != principal.scenario):
        raise PermissionError('Owned, active bank vehicle in this world required')
    return asset


def execute(principal, operation, asset_id, *, actor, parameters=None):
    if 'simulation' not in principal.permissions:
        raise PermissionError('simulation permission required')
    asset = owned_asset(principal, asset_id)
    payload = {'operation': operation, 'id': asset_id, 'actor': actor, 'ownerAccountId': asset['ownerAccountId']}
    parameters = parameters or {}
    allowed = {'claim': set(), 'observe': set(), 'release': {'leaseId'},
               'drive': {'leaseId', 'sequence', 'throttle', 'steering', 'brake', 'durationMs'}, 'attach': set(),
               'drive_to': {'destination','requestId','returnDestination','waitForTask'}, 'fly_to': {'destination','requestId','altitudeAglM','landing','returnDestination','waitForTask'}, 'mission_control': {'missionId','action'},
               'mission_status': set(), 'capabilities': set()}
    if operation not in allowed or set(parameters) - allowed[operation]:
        raise ValueError('unsupported vehicle operation or fields')
    payload.update(parameters)
    if operation == 'attach':
        filename = os.environ.get('ATLANTIS_VEHICLE_DEPLOYMENTS')
        if not filename:
            raise RuntimeError('ATLANTIS_VEHICLE_DEPLOYMENTS must select explicit server deployment/terrain data')
        config = json.loads(Path(filename).read_text())
        if config.get('version') != 1:
            raise ValueError('unsupported vehicle deployment version')
        binding = config['worlds'][principal.scenario]['vehicles'][asset_id]
        if binding['definitionId'] != asset['assetType']:
            raise ValueError('bank model and deployed model disagree')
        payload.update({key: binding[key] for key in ('terrainAssetId', 'definitionId', 'pose', 'surface')})
    if operation == 'fly_to':
        import math
        from .terrain_adapter import configuration, elevation_grid
        destination = parameters.get('destination')
        if not isinstance(destination, dict) or set(destination) != {'lat','lon'}:
            raise ValueError('Flight destination requires latitude and longitude')
        for key, limit in (('lat',85),('lon',180)):
            value = destination[key]
            if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>limit:
                raise ValueError('Invalid flight destination')
        altitude = parameters.get('altitudeAglM',60)
        landing = parameters.get('landing',False)
        if type(landing) is not bool or type(altitude) not in (int,float) or not math.isfinite(altitude) or not 10<=altitude<=300:
            raise ValueError('Flight altitude must be 10..300 metres above terrain')
        grid = elevation_grid(configuration(principal.scenario),destination,radius=2,step=2)
        ground = grid['heights'][4]
        payload.pop('altitudeAglM',None)
        payload['altitudeM'] = ground + (.3 if landing else altitude)
        payload['landing'] = landing
        return_destination = parameters.get('returnDestination')
        if return_destination is not None:
            if not isinstance(return_destination, dict) or set(return_destination) != {'lat', 'lon', 'altitudeAglM', 'landing'}:
                raise ValueError('Return flight requires coordinates, altitude and landing option')
            for key, limit in (('lat', 85), ('lon', 180)):
                value = return_destination[key]
                if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
                    raise ValueError('Invalid return flight destination')
            return_altitude = return_destination['altitudeAglM']
            if (type(return_destination['landing']) is not bool or type(return_altitude) not in (int, float)
                    or not math.isfinite(return_altitude) or not 10 <= return_altitude <= 300):
                raise ValueError('Return flight altitude must be 10..300 metres above terrain')
            return_grid = elevation_grid(configuration(principal.scenario),
                                         {key: return_destination[key] for key in ('lat', 'lon')}, radius=2, step=2)
            payload['returnDestination'] = {
                'lat': return_destination['lat'], 'lon': return_destination['lon'],
                'altitudeM': return_grid['heights'][4] + (.3 if return_destination['landing'] else return_altitude),
                'landing': return_destination['landing'],
            }
    path = f'/games/{quote(principal.scenario, safe="")}/vehicle-control'
    result = simulation_host.command('POST', path, payload)
    if result.get('error') == 'vehicle_command_rejected':
        raise VehicleCommandRejected(result.get('message', 'Vehicle command rejected'))
    return result


def mcp_command(operation, asset_id, parameters=None):
    principal = current_principal()
    return execute(principal, operation, asset_id, actor=f'mcp:{principal.external_user_id}:{principal.user_game_id}', parameters=parameters)
