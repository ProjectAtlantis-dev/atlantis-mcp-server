"""Explicit owner-only reload for the installed simulation extension."""


@visible
def reload_controllers() -> dict:
    """Restart the simulation child and reload its installed Python adapters after package installation. Persisted missions restore paused. Preserve the bank, Terrain service and existing viewer grants."""
    import importlib
    import atlantis
    from atlantis_host_adapters.identity import current_principal
    from atlantis_simulation.host import simulation_host
    from atlantis_simulation import terrain_adapter, commissioning, vehicle_control, mission_terrain, viewer
    from starlette.routing import Route, request_response
    from dynamic_functions.Terrain import viewer_server

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
    for module in (terrain_adapter,commissioning,vehicle_control,mission_terrain,viewer):
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
