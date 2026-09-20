"""Terrain vehicle control facade; bank and asset catalog remain mandatory.

The implementation is packaged in the local simulation module so a smaller
Terrain/habitat host can install the same gateway, not a second state writer.
"""
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation.vehicle_control import mcp_command, owned_asset
from atlantis_simulation import terrain_adapter
from atlantis_simulation.viewer import capabilities as viewer_capabilities
from urllib.parse import urlencode
from atlantis_simulation import commissioning


@visible
def index() -> dict:
    """Owner-only bank-registered Terrain vehicle controls."""
    return {"module": "Terrain/Vehicles", "visibility": "owner-only"}


@visible
def register(terrain_asset_id: str) -> dict:
    """Owner-only: register an existing supported Terrain vehicle in the bank. Retries retain its UUID. Does not move it."""
    return commissioning.register_vehicle(terrain_asset_id)


@visible
def attach(asset_id: str, allow_ground_snap: bool = False) -> dict:
    """Attach using real Terrain DEM. Height discrepancies over 2m require explicit ground-snap approval. Restored server pose is preserved."""
    from atlantis_simulation.host import simulation_host
    from urllib.parse import quote
    principal = current_principal()
    asset = owned_asset(principal, asset_id)
    plan = terrain_adapter.attachment(principal.scenario, asset_id)
    if plan["definitionId"] != asset["assetType"]:
        raise ValueError("Bank model and Terrain model disagree")
    surface = plan["surface"]
    ground = surface["heights"][(surface["rows"] // 2) * surface["cols"] + surface["cols"] // 2]
    if plan["definitionId"] == "patria-amv" and abs(plan["sourcePose"]["z"] - ground) > 2 and allow_ground_snap is not True:
        raise ValueError("Saved altitude differs from DEM by over 2m; inspect commissioning_plan and approve ground snap explicitly")
    payload = dict(plan, operation="attach", id=asset_id, ownerAccountId=asset["ownerAccountId"],
                   actor=f"mcp:{principal.external_user_id}:{principal.user_game_id}")
    result = simulation_host.command("POST", f"/games/{quote(principal.scenario, safe='')}/vehicle-control", payload)
    if result.get("id") == asset_id:
        commissioning.activate_binding(principal.scenario, asset_id)
    return result


@visible
def observe(asset_id: str) -> dict:
    """Read a bank-owned Terrain vehicle's durable server pose. Requires its canonical bank UUID."""
    return mcp_command("observe", asset_id)


@visible
def claim(asset_id: str) -> dict:
    """Acquire exclusive control of a bank-owned Terrain vehicle. Returns expiring lease and sequence."""
    return mcp_command("claim", asset_id)


@visible
def drive(asset_id: str, lease_id: str, sequence: int, throttle: float,
          steering: float, brake: float = 0, duration_ms: int = 500) -> dict:
    """Bounded ground controls: throttle/steering -1..1, brake 0..1, duration 50..2000ms. Not a pose write."""
    return mcp_command("drive", asset_id, {"leaseId": lease_id, "sequence": sequence,
        "throttle": throttle, "steering": steering, "brake": brake, "durationMs": duration_ms})


@visible
def release(asset_id: str, lease_id: str) -> dict:
    """Release this controller's lease and brake. Does not change asset ownership."""
    return mcp_command("release", asset_id, {"leaseId": lease_id})


@visible
def capabilities(asset_id: str) -> dict:
    """Discover supported server-side actions for your bank-owned vehicle."""
    return mcp_command("capabilities", asset_id)


def _return_destination(latitude, longitude):
    if (latitude is None) != (longitude is None):
        raise ValueError("Supply both return_latitude and return_longitude")
    return None if latitude is None else {"lat": latitude, "lon": longitude}


@visible
def drive_to(asset_id: str, latitude: float, longitude: float, request_id: str,
             return_latitude: float = None, return_longitude: float = None,
             wait_for_task: bool = False) -> dict:
    """Drive to supplied coordinates. Optional return coordinates queue a linked return after completion. wait_for_task holds at arrival until complete_task is called. Surveyed roads are preferred with local terrain detours; blocked is not completed. Same request_id retries retain identity. See instructions and README.md."""
    return mcp_command("drive_to", asset_id, {"destination": {"lat": latitude, "lon": longitude},
        "requestId": request_id, "returnDestination": _return_destination(return_latitude, return_longitude),
        "waitForTask": wait_for_task})


@visible
def fly_to(asset_id: str, latitude: float, longitude: float, request_id: str,
           altitude_agl_m: float = 60, land: bool = False,
           return_latitude: float = None, return_longitude: float = None,
           return_altitude_agl_m: float = 60, return_land: bool = False,
           wait_for_task: bool = False) -> dict:
    """Fly Black Hornet/Osprey to supplied coordinates at 10..300m above destination terrain, optionally landing. Optional return coordinates/altitude/landing define a second leg. wait_for_task requires explicit complete_task after arrival. RQ-180 is unsupported. See instructions and README.md."""
    destination = _return_destination(return_latitude, return_longitude)
    if destination is not None:
        destination.update(altitudeAglM=return_altitude_agl_m, landing=return_land)
    return mcp_command("fly_to", asset_id, {"destination": {"lat": latitude, "lon": longitude},
        "requestId": request_id, "altitudeAglM": altitude_agl_m, "landing": land,
        "returnDestination": destination, "waitForTask": wait_for_task})


@visible
def takeoff(asset_id: str, request_id: str, altitude_agl_m: float = 30) -> dict:
    """Spin up and climb vertically at the attached VTOL aircraft's current location. Observe flightPhase and mission status for actual completion."""
    state = mcp_command("observe", asset_id)
    return mcp_command("fly_to", asset_id, {"destination": {"lat": state["lat"], "lon": state["lon"]},
        "requestId": request_id, "altitudeAglM": altitude_agl_m, "landing": False})


@visible
def land(asset_id: str, request_id: str) -> dict:
    """Descend at the VTOL aircraft's current location using verified terrain. Building clearance can block touchdown; inspect the result."""
    state = mcp_command("observe", asset_id)
    return mcp_command("fly_to", asset_id, {"destination": {"lat": state["lat"], "lon": state["lon"]},
        "requestId": request_id, "altitudeAglM": 30, "landing": True})


@visible
def mission_status(asset_id: str) -> dict:
    """Read current leg and history: mission UUID, journeyId, parentMissionId, outbound/return leg, state, destination, remainingM and reason. Outbound completed is not return completed."""
    return mcp_command("mission_status", asset_id)


@visible
def mission_control(asset_id: str, mission_id: str, action: str) -> dict:
    """Pause, resume or cancel this vehicle's current mission. Restarted missions remain paused until explicit resume."""
    return mcp_command("mission_control", asset_id, {"missionId": mission_id, "action": action})


@visible
def complete_task(asset_id: str, mission_id: str) -> dict:
    """Acknowledge actual task completion ONLY after awaiting_task. Queues the saved return leg, if supplied. Does not execute a task itself. Retrying the same acknowledgement is idempotent; inspect mission_status for the current return leg."""
    return mcp_command("mission_control", asset_id, {"missionId": mission_id, "action": "complete_task"})


@visible
def instructions() -> dict:
    """Read the parameterized vehicle mission contract and Lobster orchestration steps. Full examples are in Terrain/Vehicles/README.md."""
    return {
        "dispatch": {
            "ground": "drive_to(asset_id, latitude, longitude, request_id)",
            "vtol": "fly_to(asset_id, latitude, longitude, request_id, altitude_agl_m=60, land=False)",
            "optional_return": "return_latitude + return_longitude; aircraft also return_altitude_agl_m and return_land",
            "task_at_destination": "wait_for_task=True holds after arrival until complete_task(asset_id, outbound_mission_id)",
        },
        "sequence": [
            "Use the owned bank UUID; register/attach its supported controller once; inspect capabilities.",
            "Supply destination coordinates. For return to start, observe first and pass that lat/lon as return coordinates.",
            "Save the returned mission id. Poll mission_status; queued means accepted, not arrived.",
            "If awaiting_task, execute the desired dynamic_function and observe its actual completion, then call complete_task.",
            "If return coordinates were supplied, follow the new current mission with leg=return and parentMissionId=outbound id.",
            "The round trip finishes only when the return leg reports completed. Blocked/paused/cancelled are not success.",
        ],
        "states": ["queued", "running", "awaiting_task", "completed", "blocked", "paused", "cancelled"],
        "authority": "Server simulation state persisted in SQLite; viewer animates snapshots; Lobster orchestrates task functions.",
        "restart": "Queued/running legs restore paused; resume explicitly. Awaiting tasks remain waiting. No task function is replayed automatically.",
        "retry": "Reuse request_id with identical parameters to recover the original mission; new parameters need a new request_id. return: prefix is reserved.",
        "routing": "Runtime destination-based local road preference and offroad detours over verified terrain, not saved demo waypoints or a settlement-wide road graph.",
        "limitations": "AMV ground + Black Hornet/Osprey VTOL. Fixed-wing/boat controllers and generic mission-task execution are not implemented.",
    }


@visible
def commissioning_plan(asset_id: str) -> dict:
    """Read-only check of bank ownership, Terrain model/instance binding and verified terrain elevation."""
    principal = current_principal()
    asset = owned_asset(principal, asset_id)
    plan = terrain_adapter.attachment(principal.scenario, asset_id)
    if plan["definitionId"] != asset["assetType"]:
        raise ValueError("Bank model and Terrain model disagree")
    surface = plan["surface"]
    ground = surface["heights"][(surface["rows"] // 2) * surface["cols"] + surface["cols"] // 2]
    return {"assetId": asset_id, "world": principal.scenario,
            "terrainAssetId": plan["terrainAssetId"], "definitionId": plan["definitionId"],
            "sourcePose": plan["sourcePose"], "surfaceId": plan["surface"]["id"],
            "groundElevationM": ground, "savedHeightDifferenceM": plan["sourcePose"]["z"] - ground,
            "ownershipAuthority": "bank", "positionAuthority": "simulation"}


@visible
def viewer_access(asset_id: str, ttl_seconds: int = 300) -> dict:
    """Issue scoped human control for the SAME bank-owned vehicle and server controller used by these tools."""
    principal = current_principal()
    owned_asset(principal, asset_id)
    if asset_id not in terrain_adapter.configuration(principal.scenario)["vehicles"]:
        raise ValueError("Vehicle has no Terrain deployment binding")
    grant = viewer_capabilities.issue(principal, ttl_seconds, control_vehicle_id=asset_id)
    grant["terrainCatalog"] = True
    grant["viewerFragment"] = urlencode({"simulation_game": principal.scenario,
        "simulation_token": grant["token"], "simulation_vehicle": asset_id, "simulation_terrain": "1"})
    return grant
