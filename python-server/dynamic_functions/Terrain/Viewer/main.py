"""Open the owner-scoped 3D viewer with a complete usable URL."""
import os
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
from uuid import uuid4
import atlantis
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation.vehicle_control import owned_asset, mcp_command
from atlantis_simulation.viewer import capabilities
from dynamic_functions.Terrain import viewer_server


def _viewer_session(asset_id, ttl_seconds, map_mode=False):
    principal = current_principal('simulation')
    if asset_id is not None:
        owned_asset(principal, asset_id)
        mcp_command("observe", asset_id)
    elif principal.caller not in atlantis.get_owner_usernames():
        raise PermissionError("Authenticated host owner required to open the scene without a vehicle")
    configured = os.environ.get("ATLANTIS_VIEWER_URL")
    if not configured:
        raise RuntimeError("Configure ATLANTIS_VIEWER_URL as the reachable viewer URL")
    parts = urlsplit(configured)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.fragment:
        raise RuntimeError("ATLANTIS_VIEWER_URL must be an HTTP(S) URL without a fragment")
    grant = capabilities.issue(principal, ttl_seconds, control_vehicle_id=asset_id)
    values = {"simulation_game": principal.scenario, "simulation_token": grant["token"],
              "simulation_terrain": "1"}
    if asset_id is not None:
        values["simulation_vehicle"] = asset_id
    if map_mode:
        values["simulation_view"] = "map"
    fragment = urlencode(values)
    # A fresh document must consume each new grant; changing only the hash does not reload it.
    query = [(key, value) for key, value in parse_qsl(parts.query) if key != 'viewer_session']
    query.append(('viewer_session', str(uuid4())))
    url = urlunsplit((parts.scheme, parts.netloc, parts.path or '/', urlencode(query), fragment))
    return principal, url


@visible
def index() -> dict:
    """Use link for a browser URL or open to display the owner-scoped viewer in this terminal."""
    return {"module": "Terrain/Viewer", "visibility": "owner-only", "browser": "link", "terminal": "open", "map": "open_map", "mapBrowser": "map_link"}


@visible
def link(asset_id: str, ttl_seconds: int = 900) -> str:
    """Return a complete browser URL that loads your bank ownership, fleet table and server state. Open this URL directly; do not append an access-response JSON object to the viewer address.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param ttl_seconds: Requested lifetime of the scoped viewer access grant in seconds. The returned link contains a temporary credential.
    """
    _, url = _viewer_session(asset_id, ttl_seconds)
    return url


@visible
async def open(asset_id: str, ttl_seconds: int = 900) -> dict:
    """Display the owner-scoped viewer in the calling terminal. The returned viewerUrl also opens it in a browser with the same fleet and ownership.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param ttl_seconds: Requested lifetime of the scoped viewer access grant in seconds. The returned link contains a temporary credential.
    """
    principal, url = _viewer_session(asset_id, ttl_seconds)
    await atlantis.set_background_player(url, frame=True, interactive=True, remove_on_ended=False)
    return {"world": principal.scenario, "vehicleId": asset_id, "expiresInSeconds": ttl_seconds,
            "viewerUrl": url, "instructions": "My vehicles & drones shows your fleet. M switches to the map; map labels show owner and bank UUID."}


@visible
def map_link(asset_id: str = None, ttl_seconds: int = 900) -> str:
    """Return an authenticated viewer URL that starts in map mode. No vehicle UUID is required; optionally focus the session on an owned vehicle. The URL contains a temporary access token: open it directly and do not commit or publish it.

    :param asset_id: Optional owned vehicle bank UUID to focus the session. Omit for a scene session, which requires the authenticated host owner.
    :param ttl_seconds: Requested lifetime of the scoped viewer access grant in seconds. The returned link contains a temporary credential.
    """
    _, url = _viewer_session(asset_id, ttl_seconds, map_mode=True)
    return url


@visible
async def open_map(asset_id: str = None, ttl_seconds: int = 900) -> dict:
    """Open the authenticated map in the calling terminal. Same viewer, fleet, placement tools and server state as 3D mode. Optional owned vehicle UUID; no separate map simulation is created.

    :param asset_id: Optional owned vehicle bank UUID to focus the session. Omit for a scene session, which requires the authenticated host owner.
    :param ttl_seconds: Requested lifetime of the scoped viewer access grant in seconds. The returned link contains a temporary credential.
    """
    principal, url = _viewer_session(asset_id, ttl_seconds, map_mode=True)
    await atlantis.set_background_player(url, frame=True, interactive=True, remove_on_ended=False)
    return {"world": principal.scenario, "view": "map", "vehicleId": asset_id,
            "expiresInSeconds": ttl_seconds, "viewerUrl": url}


@visible
def workshop_link(asset_id: str, ttl_seconds: int = 900) -> str:
    """Open one bank-owned placed model in the habitat/equipment workshop with live dynamic-function controls. Requires its asset UUID. The workshop shares server state with Terrain; this is not a separate simulation. Unauthenticated model previews remain art inspection only.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param ttl_seconds: Requested lifetime of the scoped viewer access grant in seconds. The returned link contains a temporary credential.
    """
    from atlantis_simulation import infrastructure_control
    principal = current_principal('simulation')
    infrastructure_control.owned(principal, asset_id)
    _, url = _viewer_session(None, ttl_seconds)
    parts = urlsplit(url)
    values = dict(parse_qsl(parts.fragment))
    values['simulation_asset'] = asset_id
    base_path = parts.path.rsplit('/', 1)[0]
    return urlunsplit((parts.scheme, parts.netloc, base_path + '/infrastructure-preview.html', parts.query, urlencode(values)))


@visible
async def open_workshop(asset_id: str, ttl_seconds: int = 900) -> dict:
    """Display the connected habitat/equipment workshop in the calling terminal. Each visible mechanism is controlled by dynamic functions and persistent server state.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param ttl_seconds: Requested lifetime of the scoped viewer access grant in seconds. The returned link contains a temporary credential.
    """
    url = workshop_link(asset_id, ttl_seconds)
    await atlantis.set_background_player(url, frame=True, interactive=True, remove_on_ended=False)
    return {'assetId': asset_id, 'viewerUrl': url, 'mode': 'connected-workshop'}
