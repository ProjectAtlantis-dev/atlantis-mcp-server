from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from atlantis_simulation.viewer import capabilities
from dynamic_functions.Terrain.Objects import runtime


async def handle(request):
    headers = {'Cache-Control': 'no-store'}
    try:
        principal = capabilities.check(request.headers.get('authorization', '').removeprefix('Bearer '), request.path_params['game_id'])
        asset_id = request.path_params['asset_id']
        if request.method == 'GET':
            result = await run_in_threadpool(runtime.describe, principal, asset_id)
        else:
            if len(await request.body()) > 8192:
                return JSONResponse({'error': 'Request too large'}, status_code=413, headers=headers)
            data = await request.json()
            if not isinstance(data, dict) or set(data) != {'action', 'parameters'}:
                raise ValueError('Action and parameters required')
            result = await run_in_threadpool(runtime.invoke, principal, asset_id, data['action'], data['parameters'])
        return JSONResponse(result, headers=headers)
    except PermissionError as error:
        return JSONResponse({'error': str(error)}, status_code=403, headers=headers)
    except (ValueError, KeyError, TypeError) as error:
        return JSONResponse({'error': str(error)}, status_code=400, headers=headers)
    except RuntimeError as error:
        return JSONResponse({'error': str(error)}, status_code=409, headers=headers)
