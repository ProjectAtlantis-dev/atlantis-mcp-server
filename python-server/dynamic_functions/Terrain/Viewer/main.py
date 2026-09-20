"""Open the owner-scoped 3D viewer with a complete usable URL."""
import os
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl
from uuid import uuid4
import atlantis
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation.vehicle_control import owned_asset, mcp_command
from atlantis_simulation.viewer import capabilities
from dynamic_functions.Terrain import viewer_server


def _viewer_session(asset_id, ttl_seconds):
    principal = current_principal()
    owned_asset(principal, asset_id)
    mcp_command("observe", asset_id)
    configured = os.environ.get("ATLANTIS_VIEWER_URL")
    if not configured:
        raise RuntimeError("Configure ATLANTIS_VIEWER_URL as the reachable viewer URL")
    parts = urlsplit(configured)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.fragment:
        raise RuntimeError("ATLANTIS_VIEWER_URL must be an HTTP(S) URL without a fragment")
    grant = capabilities.issue(principal, ttl_seconds, control_vehicle_id=asset_id)
    fragment = urlencode({"simulation_game": principal.scenario, "simulation_token": grant["token"],
                          "simulation_vehicle": asset_id, "simulation_terrain": "1"})
    # A fresh document must consume each new grant; changing only the hash does not reload it.
    query = [(key, value) for key, value in parse_qsl(parts.query) if key != 'viewer_session']
    query.append(('viewer_session', str(uuid4())))
    url = urlunsplit((parts.scheme, parts.netloc, parts.path or '/', urlencode(query), fragment))
    return principal, url


@visible
def index() -> dict:
    """Use link for a browser URL or open to display the owner-scoped viewer in this terminal."""
    return {"module": "Terrain/Viewer", "visibility": "owner-only", "browser": "link", "terminal": "open"}


@visible
def link(asset_id: str, ttl_seconds: int = 900) -> str:
    """Return a complete browser URL that loads your bank ownership, fleet table and server state. Open this URL directly; do not append an access-response JSON object to the viewer address."""
    _, url = _viewer_session(asset_id, ttl_seconds)
    return url


@visible
async def open(asset_id: str, ttl_seconds: int = 900) -> dict:
    """Display the owner-scoped viewer in the calling terminal. The returned viewerUrl also opens it in a browser with the same fleet and ownership."""
    principal, url = _viewer_session(asset_id, ttl_seconds)
    await atlantis.set_background_player(url, frame=True, interactive=True, remove_on_ended=False)
    return {"world": principal.scenario, "vehicleId": asset_id, "expiresInSeconds": ttl_seconds,
            "viewerUrl": url, "instructions": "My vehicles & drones shows your fleet. M switches to the map; map labels show owner and bank UUID."}
