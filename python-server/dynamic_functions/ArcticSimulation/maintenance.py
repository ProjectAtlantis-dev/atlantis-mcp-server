"""Explicit owner-only reload for the installed simulation extension."""


@visible
def reload_controllers() -> dict:
    """Restart the simulation child and reload its installed Python adapters after package installation. Persisted missions restore paused. Preserve the bank, Terrain service and existing viewer grants."""
    import importlib
    import atlantis
    from atlantis_host_adapters.identity import current_principal
    from atlantis_simulation.host import simulation_host
    from atlantis_simulation import terrain_adapter, commissioning, vehicle_control, mission_terrain, infrastructure_control, equipment_control, player_control, viewer
    from starlette.routing import Route, request_response
    from dynamic_functions.Terrain import viewer_server
    from dynamic_functions.Terrain.Objects import runtime as object_runtime
    from dynamic_functions.Terrain.Defense import gateway as defense_gateway

    from dynamic_functions.Terrain.Placement import gateway as placement_gateway

    principal=current_principal()
    if principal.caller not in atlantis.get_owner_usernames():
        raise PermissionError('Authenticated host owner required for controller reload')
    state=simulation_host.status()
    if not state['running']:
        raise RuntimeError('Start the simulation before reloading controllers')
    runtime=atlantis.server_shared.get(viewer_server._RUNTIME_KEY)
    if runtime is None:
        raise RuntimeError('Start Terrain HTTP before reloading its simulation routes')
    app=runtime.server.config.app
    if not hasattr(app,'router'):
        raise RuntimeError('Terrain HTTP does not expose the expected route owner')
    grants=viewer.capabilities
    simulation_host.stop()
    from atlantis_simulation import host as host_module
    importlib.reload(host_module)
    for module in (terrain_adapter,commissioning,vehicle_control,mission_terrain,infrastructure_control,equipment_control,player_control,defense_gateway,placement_gateway,object_runtime,viewer):
        importlib.reload(module)
    viewer.capabilities=grants
    replacements={route.path:route for route in viewer.routes()}
    for route in app.router.routes:
        if isinstance(route,Route) and route.path in replacements:
            replacement=replacements.pop(route.path)
            route.endpoint=replacement.endpoint
            route.app=request_response(replacement.endpoint)
    app.router.routes.extend(replacements.values())
    result=simulation_host.start(host=state['host'],port=state['port'],database_path=state['databasePath'])
    return {'reloaded':True,'simulation':result,'missions':'restored paused; resume explicitly',
            'terrain':'running','bank':'unchanged','viewerGrants':'preserved'}


@visible
def routing_status() -> dict:
    """Owner-only terrain-worker health. Reports whether routing is running and its current stack location; does not expose credentials or restart a mission."""
    import sys
    import traceback
    import atlantis
    from atlantis_host_adapters.identity import current_principal
    from atlantis_simulation.host import simulation_host
    if current_principal().caller not in atlantis.get_owner_usernames():
        raise PermissionError('Authenticated host owner required')
    worker = simulation_host._terrain_worker
    if worker is None:
        return {'workerRunning': False, 'reason': 'Terrain supply worker is absent'}
    frame = sys._current_frames().get(worker.thread.ident)
    return {'workerRunning': worker.thread.is_alive(), 'stopping': worker.stopped.is_set(),
            'stack': [{'function': frame.name, 'file': frame.filename.rsplit('/', 1)[-1], 'line': frame.lineno}
                      for frame in traceback.extract_stack(frame)[-6:]] if frame else [],
            'progress': getattr(worker, 'progress', {})}


@visible
def reload_terrain_worker() -> dict:
    """Owner-only installed terrain adapter reload without restarting vehicle simulation or pausing missions. Joins the prior worker before starting one replacement."""
    import importlib
    import atlantis
    from atlantis_host_adapters.identity import current_principal
    from atlantis_simulation.host import simulation_host
    from atlantis_simulation import terrain_adapter, mission_terrain
    if current_principal().caller not in atlantis.get_owner_usernames():
        raise PermissionError('Authenticated host owner required')
    if not simulation_host.status()['running']:
        raise RuntimeError('Simulation must be running')
    worker = simulation_host._terrain_worker
    if worker is not None:
        worker.stop()
        worker.thread.join(timeout=10)
        if worker.thread.is_alive():
            raise RuntimeError('Previous terrain worker is still finishing; no replacement started')
    from dynamic_functions.Terrain.Database import tiles
    from dynamic_functions.Terrain import composition
    importlib.reload(tiles)
    importlib.reload(composition)
    importlib.reload(terrain_adapter)
    importlib.reload(mission_terrain)
    replacement = mission_terrain.MissionTerrainWorker(simulation_host)
    simulation_host._terrain_worker = replacement
    replacement.start()
    return {'workerRunning': replacement.thread.is_alive(), 'simulationRestarted': False,
            'missionsPaused': False}


@visible
def reload_asset_functions() -> dict:
    """Reload owner-authorized placement and building-grounding functions without restarting simulation or pausing missions."""
    import importlib
    import atlantis
    from atlantis_host_adapters.identity import current_principal
    if current_principal().caller not in atlantis.get_owner_usernames():
        raise PermissionError('Authenticated host owner required')
    from dynamic_functions.Terrain.Placement import gateway
    from dynamic_functions.Terrain.Asset import grounding
    from dynamic_functions.Terrain.Objects import runtime
    for module in (gateway, grounding, runtime):
        importlib.reload(module)
    return {'reloaded': True, 'simulationRestarted': False, 'missionsPaused': False}
