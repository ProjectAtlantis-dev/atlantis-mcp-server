"""Read-only, expiring, game-scoped viewer capabilities. Child token stays private."""
import logging
import hashlib
import secrets
import threading
import time
from urllib.parse import quote

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from .host import simulation_host
from atlantis_host_adapters.identity import current_principal, resolve_principal


log = logging.getLogger(__name__)


class ViewerCapabilities:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.grants = {}

    def issue(self, principal, ttl=300, control_vehicle_id=None, control_player_id=None):
        if type(ttl) is not int or not 1 <= ttl <= 900:
            raise ValueError('Viewer TTL must be 1..900 seconds')
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).digest()
        with self.lock:
            now = self.clock()
            self.grants = {k: v for k, v in self.grants.items() if v[1] > now}
            if len(self.grants) >= 4096:
                raise RuntimeError('Viewer capability limit reached')
            self.grants[digest] = (principal, now + ttl, control_vehicle_id, control_player_id)
        return {'token': token, 'expiresInSeconds': ttl,
                'gameId': principal.scenario, 'controlVehicleId': control_vehicle_id, 'controlPlayerId': control_player_id,
                'snapshotUrl': f'/api/simulation/{quote(principal.scenario, safe="")}/snapshot'}

    def check(self, token, scenario):
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise PermissionError('Viewer capability required')
        with self.lock:
            grant = self.grants.get(hashlib.sha256(token.encode()).digest())
        if grant is None or grant[1] <= self.clock() or grant[0].scenario != scenario:
            raise PermissionError('Invalid, expired or out-of-scope viewer capability')
        # Policy revocation applies on every request, not only when issued.
        principal = grant[0]
        from types import SimpleNamespace
        current = resolve_principal(SimpleNamespace(caller_sid=principal.caller,
                                    user_game_id=principal.user_game_id), 'simulation')
        if current != principal:
            raise PermissionError('Viewer capability binding changed')
        return principal

    def check_control(self, token, scenario, vehicle_id):
        principal = self.check(token, scenario)
        with self.lock:
            grant = self.grants[hashlib.sha256(token.encode()).digest()]
        if not vehicle_id or grant[2] != vehicle_id:
            raise PermissionError('Vehicle control capability required')
        return principal

    def check_player(self, token, scenario):
        from .player_control import player_id
        principal = self.check(token, scenario)
        with self.lock:
            grant = self.grants[hashlib.sha256(token.encode()).digest()]
        if not grant[3] or grant[3] != player_id(principal):
            raise PermissionError('Player control capability required')
        return principal


capabilities = ViewerCapabilities()


def issue_viewer_access(ttl_seconds=300):
    return capabilities.issue(current_principal(), ttl_seconds)


async def snapshot(request):
    headers = {'Cache-Control': 'no-store'}
    authorization = request.headers.get('authorization', '')
    token = authorization[7:] if authorization.startswith('Bearer ') else ''
    scenario = request.path_params['game_id']
    try:
        capabilities.check(token, scenario)
    except PermissionError:
        return JSONResponse({'error': 'viewer_access_denied'}, status_code=403, headers=headers)
    try:
        result = await run_in_threadpool(simulation_host.command, 'GET',
                                        f'/games/{quote(scenario, safe="")}/snapshot')
    except RuntimeError:
        log.exception('Viewer snapshot request failed')
        return JSONResponse({'error': 'simulation_unavailable'}, status_code=503, headers=headers)
    return JSONResponse(result, headers=headers)


def routes():
    return [Route('/api/simulation/{game_id}/snapshot', snapshot, methods=['GET']),
            Route('/api/simulation/{game_id}/terrain-assets', terrain_assets, methods=['GET']),
            Route('/api/simulation/{game_id}/player-control', player_control, methods=['POST']),
            Route('/api/simulation/{game_id}/vehicle-control', vehicle_control, methods=['POST'])]


async def terrain_assets(request):
    from .terrain_adapter import startup
    token = request.headers.get('authorization', '').removeprefix('Bearer ')
    world = request.path_params['game_id']
    try:
        principal = capabilities.check(token, world)
        from atlantis_economy.gateway import account_for, bank_request
        account = await run_in_threadpool(account_for, principal)
        portfolio = await run_in_threadpool(bank_request, 'GET', f"/accounts/{account['id']}/portfolio")
        state = await run_in_threadpool(simulation_host.command, 'GET', f'/games/{quote(world,safe="")}/snapshot')
        result = await run_in_threadpool(startup, world, state, portfolio['ownedAssets'])
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except PermissionError:
        return JSONResponse({'error': 'terrain_access_denied'}, status_code=403)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        return JSONResponse({'error': 'terrain_binding_unavailable', 'message': str(error)}, status_code=503)


async def player_control(request):
    import json
    from urllib.error import URLError
    from .player_control import execute
    token = request.headers.get('authorization', '').removeprefix('Bearer ')
    try:
        body = await request.body()
        if len(body) > 4096:
            return JSONResponse({'error': 'body_too_large'}, status_code=413)
        payload = json.loads(body)
        if not isinstance(payload, dict) or set(payload) - {'operation', 'parameters'}:
            raise ValueError('Only operation and parameters allowed')
        principal = await run_in_threadpool(capabilities.check_player, token, request.path_params['game_id'])
        result = await run_in_threadpool(execute, principal, payload['operation'],
            actor='browser:' + hashlib.sha256(token.encode()).hexdigest(), parameters=payload.get('parameters', {}))
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except PermissionError:
        return JSONResponse({'error': 'player_access_denied'}, status_code=403)
    except (ValueError, KeyError, TypeError):
        return JSONResponse({'error': 'invalid_player_command'}, status_code=400)
    except (RuntimeError, URLError):
        log.exception('Player command failed')
        return JSONResponse({'error': 'player_command_rejected'}, status_code=409)


async def vehicle_control(request):
    from .vehicle_control import execute, VehicleCommandRejected
    authorization = request.headers.get('authorization', '')
    token = authorization[7:] if authorization.startswith('Bearer ') else ''
    try:
        body = await request.body()
        if len(body) > 4096:
            return JSONResponse({'error': 'body_too_large'}, status_code=413)
        import json
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError('object required')
        asset_id = payload['assetId']
        principal = capabilities.check_control(token, request.path_params['game_id'], asset_id)
        operation = payload['operation']
        if operation not in ('claim', 'drive', 'release', 'observe', 'drive_to', 'mission_control', 'mission_status', 'capabilities'):
            raise PermissionError('browser operation forbidden')
        actor = 'browser:' + hashlib.sha256(token.encode()).hexdigest()
        result = await run_in_threadpool(execute, principal, operation, asset_id, actor=actor,
                                        parameters=payload.get('parameters', {}))
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except PermissionError:
        return JSONResponse({'error': 'vehicle_access_denied'}, status_code=403)
    except (ValueError, KeyError, TypeError):
        return JSONResponse({'error': 'invalid_vehicle_command'}, status_code=400)
    except VehicleCommandRejected as exc:
        return JSONResponse({'error': 'vehicle_command_rejected', 'message': str(exc)}, status_code=409)
    except RuntimeError:
        log.exception('Vehicle command failed')
        return JSONResponse({'error': 'vehicle_command_rejected'}, status_code=409)
