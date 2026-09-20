"""Installed application route extensions; no module-specific host imports."""
from importlib.metadata import entry_points
from starlette.routing import Route


def guard_asset_write(asset_id):
    """Installed state owners may reject legacy catalog writes for managed IDs."""
    for provider in sorted(entry_points(group='atlantis.asset_write_guards'), key=lambda item: item.name):
        provider.load()(asset_id)


def install_viewer_routes(app, providers=None):
    providers = entry_points(group='atlantis.viewer_routes') if providers is None else providers
    paths = {getattr(route, 'path', None) for route in app.routes}
    pending = []
    for provider in sorted(providers, key=lambda item: item.name):
        for route in provider.load()():
            if not isinstance(route, Route) or not route.path.startswith('/api/'):
                raise ValueError(f'Invalid viewer route from {provider.name}')
            if route.path in paths:
                raise ValueError(f'Conflicting viewer route: {route.path}')
            paths.add(route.path)
            pending.append(route)
    app.router.routes.extend(pending)
