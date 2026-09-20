"""Strategic infrastructure placement; no timers or simulation authority here."""
from urllib.parse import quote
from atlantis_simulation.host import simulation_host
from atlantis_host_adapters.identity import authorized_scenario

def _path(game_id: str, action: str = "infrastructure") -> str:
    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError("game_id must be nonempty")
    return f"/games/{quote(authorized_scenario(game_id.strip()), safe='')}/{action}"

@visible
def index() -> dict:
    """Infrastructure catalog and durable visual placement commands."""
    return {"commands": ["catalog", "list_assets", "place", "move", "remove", "component_instructions", "airlock_open_outer", "airlock_open_inner", "airlock_close", "facility_entry_outer", "facility_entry_inner", "facility_entry_close", "facility_freight_open", "facility_freight_close"],
            "coordinates": "local ENU metres", "state": "visual placement, not operational activation"}

@visible
def catalog(game_id: str = "default") -> dict:
    """List stable model IDs, dimensions and explicit audit limitations."""
    return simulation_host.command("GET", _path(game_id, "infrastructure-catalog"))

@visible
def list_assets(game_id: str = "default") -> dict:
    """Inspect all infrastructure placed in this authoritative room."""
    return simulation_host.command("GET", _path(game_id))

@visible
def place(model_id: str, x: float, y: float, z: float, heading_deg: float = 0,
          game_id: str = "default", source_vehicle_id: str | None = None) -> dict:
    """Place a catalog model in local ENU metres; does not activate a weapon or network."""
    return simulation_host.command("POST", _path(game_id), {"modelId": model_id,
        "position": {"x": x, "y": y, "z": z}, "headingDeg": heading_deg,
        "sourceVehicleId": source_vehicle_id})

@visible
def move(asset_id: str, x: float, y: float, z: float, heading_deg: float = 0,
         game_id: str = "default") -> dict:
    """Reposition an existing visual infrastructure instance by stable ID."""
    return simulation_host.command("PATCH", _path(game_id), {"id": asset_id,
        "position": {"x": x, "y": y, "z": z}, "headingDeg": heading_deg})

@visible
def remove(asset_id: str, game_id: str = "default") -> dict:
    """Remove a visual infrastructure placement from this room."""
    return simulation_host.command("DELETE", _path(game_id), {"id": asset_id})

@visible
def component_instructions(game_id: str = "default") -> dict:
    """Discover airlock and facility access preconditions, completion semantics and simulation limitations."""
    return simulation_host.command("GET", _path(game_id, "component-contract"))

def _airlock_command(asset_id: str, action: str, expected_revision: int, game_id: str) -> dict:
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        raise ValueError("expected_revision must be a nonnegative integer from list_assets")
    return simulation_host.command("POST", _path(game_id, "component-command"),
        {"id": asset_id, "action": action, "expectedRevision": expected_revision})

@visible
def airlock_open_outer(asset_id: str, expected_revision: int, game_id: str = "default") -> dict:
    """Request outer-door opening. Inner position AND target must be closed. Read list_assets for completion; not a safety controller."""
    return _airlock_command(asset_id, "airlock_open_outer", expected_revision, game_id)

@visible
def airlock_open_inner(asset_id: str, expected_revision: int, game_id: str = "default") -> dict:
    """Request inner-door opening after outer closure completes. Server tick owns animation. No pressure/occupancy guarantee."""
    return _airlock_command(asset_id, "airlock_open_inner", expected_revision, game_id)

@visible
def airlock_close(asset_id: str, expected_revision: int, game_id: str = "default") -> dict:
    """Request both doors closed. Acceptance is not completion; inspect componentState positions. Not emergency evacuation control."""
    return _airlock_command(asset_id, "airlock_close", expected_revision, game_id)

@visible
def facility_entry_outer(asset_id: str, expected_revision: int, game_id: str = "default") -> dict:
    """Open a standalone facility's outer entry door. Inner and freight doors must be fully closed. Read list_assets for completion; no pressure safety claim."""
    return _airlock_command(asset_id, "facility_entry_outer", expected_revision, game_id)

@visible
def facility_entry_inner(asset_id: str, expected_revision: int, game_id: str = "default") -> dict:
    """Open a standalone facility's inner entry door after outer and freight closure. Server tick moves it; not an emergency egress controller."""
    return _airlock_command(asset_id, "facility_entry_inner", expected_revision, game_id)

@visible
def facility_entry_close(asset_id: str, expected_revision: int, game_id: str = "default") -> dict:
    """Request facility entry doors closed. Acceptance is not completed closure; inspect returned state and list_assets."""
    return _airlock_command(asset_id, "facility_entry_close", expected_revision, game_id)

@visible
def facility_freight_open(asset_id: str, expected_revision: int, game_id: str = "default") -> dict:
    """Open simulated freight doors only after entry doors and targets are closed. This bypasses the envelope; no environmental release or obstacle safety is simulated."""
    return _airlock_command(asset_id, "facility_freight_open", expected_revision, game_id)

@visible
def facility_freight_close(asset_id: str, expected_revision: int, game_id: str = "default") -> dict:
    """Request freight closure. Read componentState.freight for completion before opening an entry door."""
    return _airlock_command(asset_id, "facility_freight_close", expected_revision, game_id)
