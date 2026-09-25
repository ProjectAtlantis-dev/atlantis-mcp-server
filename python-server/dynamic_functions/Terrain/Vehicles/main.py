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
    """Owner-only: register an existing supported Terrain vehicle in the bank. Retries retain its UUID. Does not move it.

    :param terrain_asset_id: Existing terrain vehicle instance identifier to register. This is the pre-registration terrain ID, not a bank UUID or model type.
    """
    return commissioning.register_vehicle(terrain_asset_id)


@visible
def attach(asset_id: str, allow_ground_snap: bool = False) -> dict:
    """Attach using real Terrain DEM. Height discrepancies over 2m require explicit ground-snap approval. Restored server pose is preserved.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param allow_ground_snap: Explicit approval to snap supported ground vehicles to verified DEM when saved height differs by over 2m. Inspect commissioning_plan first.
    """
    from atlantis_simulation.host import simulation_host
    from urllib.parse import quote
    principal = current_principal()
    asset = owned_asset(principal, asset_id)
    plan = terrain_adapter.attachment(principal.scenario, asset_id)
    if plan["definitionId"] != asset["assetType"]:
        raise ValueError("Bank model and Terrain model disagree")
    surface = plan["surface"]
    ground = surface["heights"][(surface["rows"] // 2) * surface["cols"] + surface["cols"] // 2]
    if plan["definitionId"] in ("patria-amv", "at1-hrim") and abs(plan["sourcePose"]["z"] - ground) > 2 and allow_ground_snap is not True:
        raise ValueError("Saved altitude differs from DEM by over 2m; inspect commissioning_plan and approve ground snap explicitly")
    payload = dict(plan, operation="attach", id=asset_id, ownerAccountId=asset["ownerAccountId"],
                   actor=f"mcp:{principal.external_user_id}:{principal.user_game_id}")
    result = simulation_host.command("POST", f"/games/{quote(principal.scenario, safe='')}/vehicle-control", payload)
    if result.get("id") == asset_id:
        commissioning.activate_binding(principal.scenario, asset_id)
    return result


@visible
def observe(asset_id: str) -> dict:
    """Read a bank-owned Terrain vehicle's durable server pose. Requires its canonical bank UUID.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    """
    return mcp_command("observe", asset_id)


@visible
def claim(asset_id: str) -> dict:
    """Acquire exclusive control of a bank-owned Terrain vehicle. Returns expiring lease and sequence.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    """
    return mcp_command("claim", asset_id)


@visible
def drive(asset_id: str, lease_id: str, sequence: int, throttle: float,
          steering: float, brake: float = 0, duration_ms: int = 500) -> dict:
    """Bounded ground controls: throttle/steering -1..1, brake 0..1, duration 50..2000ms. Not a pose write.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param lease_id: Unexpired control lease identifier returned by claim for this vehicle.
    :param sequence: Next control sequence number for this lease, based on the claim/control responses.
    :param throttle: Signed normalized throttle from -1 to 1.
    :param steering: Signed normalized steering from -1 to 1.
    :param brake: Normalized braking from 0 to 1.
    :param duration_ms: Control interval in milliseconds, from 50 to 2000.
    """
    return mcp_command("drive", asset_id, {"leaseId": lease_id, "sequence": sequence,
        "throttle": throttle, "steering": steering, "brake": brake, "durationMs": duration_ms})


@visible
def release(asset_id: str, lease_id: str) -> dict:
    """Release this controller's lease and brake. Does not change asset ownership.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param lease_id: Unexpired control lease identifier returned by claim for this vehicle.
    """
    return mcp_command("release", asset_id, {"leaseId": lease_id})


@visible
def capabilities(asset_id: str) -> dict:
    """Discover supported server-side actions for your bank-owned vehicle.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    """
    return mcp_command("capabilities", asset_id)


def _return_destination(latitude, longitude):
    if (latitude is None) != (longitude is None):
        raise ValueError("Supply both return_latitude and return_longitude")
    return None if latitude is None else {"lat": latitude, "lon": longitude}


@visible
def drive_to(asset_id: str, latitude: float, longitude: float, request_id: str,
             return_latitude: float = None, return_longitude: float = None,
             wait_for_task: bool = False, auto_return: bool = False) -> dict:
    """Drive to supplied coordinates. auto_return=True captures the departure point and returns there after completion; otherwise supply optional return coordinates. wait_for_task holds at arrival until complete_task is called. Surveyed roads are preferred with local terrain detours; blocked is not completed. Same request_id retries retain identity. See instructions and README.md.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param latitude: Destination WGS84 latitude in decimal degrees.
    :param longitude: Destination WGS84 longitude in decimal degrees.
    :param request_id: Caller-chosen unique operation key. Retain it for retries of identical terms; use a new key for a new operation.
    :param return_latitude: Explicit return WGS84 latitude in degrees; supply with return_longitude, and omit when auto_return is true.
    :param return_longitude: Explicit return WGS84 longitude in degrees; supply with return_latitude, and omit when auto_return is true.
    :param wait_for_task: When true, hold at the destination in awaiting_task until complete_task acknowledges the task; does not execute or time a task.
    :param auto_return: When true, capture the departure position and return after completion; do not also supply explicit return coordinates.
    """
    return mcp_command("drive_to", asset_id, {"destination": {"lat": latitude, "lon": longitude},
        "requestId": request_id, "returnDestination": _return_destination(return_latitude, return_longitude),
        "waitForTask": wait_for_task, "autoReturn": auto_return})


@visible
def sail_to(asset_id: str, latitude: float, longitude: float, request_id: str,
            return_latitude: float = None, return_longitude: float = None,
            wait_for_task: bool = False, auto_return: bool = False) -> dict:
    """Sail through verified water to supplied coordinates, preferring open water with hull clearance from the asset dimensions and turning room from its steering profile. Unsafe endpoints are rejected, not moved. Optional auto_return captures departure; wait_for_task holds until complete_task. Unknown water/land is impassable.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param latitude: Destination WGS84 latitude in decimal degrees.
    :param longitude: Destination WGS84 longitude in decimal degrees.
    :param request_id: Caller-chosen unique operation key. Retain it for retries of identical terms; use a new key for a new operation.
    :param return_latitude: Explicit return WGS84 latitude in degrees; supply with return_longitude, and omit when auto_return is true.
    :param return_longitude: Explicit return WGS84 longitude in degrees; supply with return_latitude, and omit when auto_return is true.
    :param wait_for_task: When true, hold at the destination in awaiting_task until complete_task acknowledges the task; does not execute or time a task.
    :param auto_return: When true, capture the departure position and return after completion; do not also supply explicit return coordinates.
    """
    return mcp_command("sail_to", asset_id, {"destination": {"lat": latitude, "lon": longitude},
        "requestId": request_id, "returnDestination": _return_destination(return_latitude, return_longitude),
        "waitForTask": wait_for_task, "autoReturn": auto_return})


@visible
def fly_to(asset_id: str, latitude: float, longitude: float, request_id: str,
           altitude_agl_m: float = 60, land: bool = False,
           return_latitude: float = None, return_longitude: float = None,
           return_altitude_agl_m: float = 60, return_land: bool = False,
           wait_for_task: bool = False, auto_return: bool = False, takeoff_heading_deg: float = None,
           water_level_m: float = None) -> dict:
    """Fly an attached aircraft up to 10km at 10..300m above destination terrain, optionally landing. For flight over verified water, supply water_level_m in the terrain vertical datum; missing land elevation remains impassable and water landing is rejected. auto_return=True returns to the departure coordinates and initial landing/hover state; alternatively supply return coordinates/altitude/landing. wait_for_task requires explicit complete_task after arrival. RQ-180 performs a fixed-wing flyover then loiters; landing is unsupported. Ground departures require a clear level takeoff roll; takeoff_heading_deg sets its heading counterclockwise from north. Fixed-wing auto_return requires an airborne departure. See instructions and README.md.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param latitude: Destination WGS84 latitude in decimal degrees.
    :param longitude: Destination WGS84 longitude in decimal degrees.
    :param request_id: Caller-chosen unique operation key. Retain it for retries of identical terms; use a new key for a new operation.
    :param altitude_agl_m: Target altitude in metres above destination terrain. Inspect aircraft capabilities; this is not an absolute elevation.
    :param land: When true, request touchdown at the destination. Fixed-wing landing and landing on water are unsupported.
    :param return_latitude: Explicit return WGS84 latitude in degrees; supply with return_longitude, and omit when auto_return is true.
    :param return_longitude: Explicit return WGS84 longitude in degrees; supply with return_latitude, and omit when auto_return is true.
    :param return_altitude_agl_m: Altitude in metres above terrain for an explicitly supplied return destination.
    :param return_land: Whether to land at an explicitly supplied return destination. auto_return instead restores the departure landing/hover state.
    :param wait_for_task: When true, hold at the destination in awaiting_task until complete_task acknowledges the task; does not execute or time a task.
    :param auto_return: When true, capture the departure position and return after completion; do not also supply explicit return coordinates.
    :param takeoff_heading_deg: Fixed-wing takeoff heading in degrees counterclockwise from north; relevant to ground departures requiring a clear takeoff roll.
    :param water_level_m: Verified water-surface elevation in metres in the terrain vertical datum; not water depth or altitude above ground.
    """
    destination = _return_destination(return_latitude, return_longitude)
    if destination is not None:
        destination.update(altitudeAglM=return_altitude_agl_m, landing=return_land)
    return mcp_command("fly_to", asset_id, {"destination": {"lat": latitude, "lon": longitude},
        "requestId": request_id, "altitudeAglM": altitude_agl_m, "landing": land, "takeoffHeadingDeg": takeoff_heading_deg,
        "waterLevelM": water_level_m,
        "returnDestination": destination, "waitForTask": wait_for_task, "autoReturn": auto_return})


@visible
def takeoff(asset_id: str, request_id: str, altitude_agl_m: float = 30) -> dict:
    """Spin up and climb vertically at the attached VTOL aircraft's current location. Observe flightPhase and mission status for actual completion.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param request_id: Caller-chosen unique operation key. Retain it for retries of identical terms; use a new key for a new operation.
    :param altitude_agl_m: Target altitude in metres above destination terrain. Inspect aircraft capabilities; this is not an absolute elevation.
    """
    state = mcp_command("observe", asset_id)
    if state["authority"] != "server-vtol-v1":
        raise ValueError("Vertical takeoff/landing requires a VTOL controller")
    return mcp_command("fly_to", asset_id, {"destination": {"lat": state["lat"], "lon": state["lon"]},
        "requestId": request_id, "altitudeAglM": altitude_agl_m, "landing": False})


@visible
def land(asset_id: str, request_id: str) -> dict:
    """Descend at the VTOL aircraft's current location using verified terrain. Building clearance can block touchdown; inspect the result.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param request_id: Caller-chosen unique operation key. Retain it for retries of identical terms; use a new key for a new operation.
    """
    state = mcp_command("observe", asset_id)
    if state["authority"] != "server-vtol-v1":
        raise ValueError("Vertical takeoff/landing requires a VTOL controller")
    return mcp_command("fly_to", asset_id, {"destination": {"lat": state["lat"], "lon": state["lon"]},
        "requestId": request_id, "altitudeAglM": 30, "landing": True})


@visible
def mission_status(asset_id: str) -> dict:
    """Read current leg and history: mission UUID, journeyId, parentMissionId, outbound/return leg, state, destination, remainingM and reason. Outbound completed is not return completed.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    """
    return mcp_command("mission_status", asset_id)


@visible
def mission_control(asset_id: str, mission_id: str, action: str) -> dict:
    """Pause, resume or cancel this vehicle's current mission. For a blocked mission, resume clears the failed route and maneuvers and queues fresh terrain/route planning, preserving the mission UUID, destination and return settings. It does not bypass hazards or guarantee arrival. Inspect mission_status after resuming. Restarted missions remain paused until explicit resume.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param mission_id: Exact current mission UUID from mission_status. Use the outbound awaiting_task mission for complete_task, not a parent journey ID.
    :param action: pause stops progress; resume retries a blocked route or continues a paused mission; cancel terminates this mission. Use complete_task separately to acknowledge task completion.
    """
    return mcp_command("mission_control", asset_id, {"missionId": mission_id, "action": action})


@visible
def complete_task(asset_id: str, mission_id: str) -> dict:
    """Acknowledge actual task completion ONLY after awaiting_task. Queues the saved return leg, if supplied. Does not execute a task itself. Retrying the same acknowledgement is idempotent; inspect mission_status for the current return leg.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param mission_id: Exact current mission UUID from mission_status. Use the outbound awaiting_task mission for complete_task, not a parent journey ID.
    """
    return mcp_command("mission_control", asset_id, {"missionId": mission_id, "action": "complete_task"})


@visible
def instructions() -> dict:
    """Read the parameterized vehicle mission contract and Lobster orchestration steps. Full examples are in Terrain/Vehicles/README.md."""
    return {
        "dispatch": {
            "ground": "drive_to(asset_id, latitude, longitude, request_id)",
            "vtol": "fly_to(asset_id, latitude, longitude, request_id, altitude_agl_m=60, land=False)",
            "auto_return": "auto_return=True captures the departure point atomically; do not also supply return coordinates",
            "optional_return": "return_latitude + return_longitude; aircraft also return_altitude_agl_m and return_land",
            "task_at_destination": "wait_for_task=True holds after arrival until complete_task(asset_id, outbound_mission_id)",
        },
        "sequence": [
            "Use the owned bank UUID; register/attach its supported controller once; inspect capabilities.",
            "Supply destination coordinates. Set auto_return=True to return to the departure point without entering return coordinates.",
            "Save the returned mission id. Poll mission_status; queued means accepted, not arrived.",
            "If awaiting_task, execute the desired dynamic_function and observe its actual completion, then call complete_task.",
            "If auto_return or return coordinates were supplied, follow the new current mission with leg=return and parentMissionId=outbound id.",
            "The round trip finishes only when the return leg reports completed. Blocked/paused/cancelled are not success.",
        ],
        "states": ["queued", "running", "awaiting_task", "completed", "blocked", "paused", "cancelled"],
        "authority": "Server simulation state persisted in SQLite; viewer animates snapshots; Lobster orchestrates task functions.",
        "restart": "Queued/running legs restore paused; resume explicitly. Awaiting tasks remain waiting. No task function is replayed automatically.",
        "retry": "Reuse request_id with identical parameters to recover the original mission; new parameters need a new request_id. return: prefix is reserved.",
        "routing": "A complete destination route with road preference, detailed terrain validation and turning-aware local maneuvers. Coordinates are supplied per command.",
        "limitations": "AMV/Hrim ground, Black Hornet/Osprey VTOL, water-only boat missions, RQ-180 fixed-wing flyover/loiter. Fixed-wing landing and generic mission-task execution are not implemented. Hrim and catalog aircraft/boat values are simulation tuning unless explicitly sourced.",
    }


@visible
def commissioning_plan(asset_id: str) -> dict:
    """Read-only check of bank ownership, Terrain model/instance binding and verified terrain elevation.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    """
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
    """Issue scoped human control for the SAME bank-owned vehicle and server controller used by these tools.

    :param asset_id: Canonical bank UUID returned by fleet or placement; never a display name or model ID.
    :param ttl_seconds: Requested lifetime of the scoped viewer access grant in seconds. The returned link contains a temporary credential.
    """
    principal = current_principal()
    owned_asset(principal, asset_id)
    if asset_id not in terrain_adapter.configuration(principal.scenario)["vehicles"]:
        raise ValueError("Vehicle has no Terrain deployment binding")
    grant = viewer_capabilities.issue(principal, ttl_seconds, control_vehicle_id=asset_id)
    grant["terrainCatalog"] = True
    grant["viewerFragment"] = urlencode({"simulation_game": principal.scenario,
        "simulation_token": grant["token"], "simulation_vehicle": asset_id, "simulation_terrain": "1"})
    return grant
