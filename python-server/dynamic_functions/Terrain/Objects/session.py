"""Keep an actively used owner viewer connected without changing its control scope."""
import hashlib
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from atlantis_simulation.viewer import capabilities
from atlantis_simulation.vehicle_control import owned_asset


async def keep_alive(request):
    token = request.headers.get('authorization', '').removeprefix('Bearer ')
    headers = {'Cache-Control': 'no-store'}
    try:
        principal = capabilities.check(token, request.path_params['game_id'])
        digest = hashlib.sha256(token.encode()).digest()
        with capabilities.lock:
            grant = capabilities.grants.get(digest)
        if grant is None or not grant[2]:
            raise PermissionError('An active vehicle viewer session is required')
        await run_in_threadpool(owned_asset, principal, grant[2])
        with capabilities.lock:
            current = capabilities.grants.get(digest)
            if current != grant or current[1] <= capabilities.clock():
                raise PermissionError('Viewer session expired or changed')
            capabilities.grants[digest] = (principal, capabilities.clock() + 900, grant[2], grant[3])
        return JSONResponse({'connected': True, 'expiresInSeconds': 900}, headers=headers)
    except PermissionError as error:
        return JSONResponse({'error': str(error)}, status_code=403, headers=headers)
    except RuntimeError as error:
        return JSONResponse({'error': str(error)}, status_code=503, headers=headers)
