"""Bank-owned infrastructure shared by Terrain and standalone habitat hosts.

Bank commits identity first; simulation owns placement/component motion. A stable
instance key allows reconciliation after interruption between the two services.
No component position or completion state is synthesized by this gateway.
"""
import math
from urllib.parse import quote

from atlantis_economy.gateway import account_for, bank_request, canonical_uuid
from atlantis_host_adapters.identity import current_principal
from .host import simulation_host


def path(principal, action="infrastructure"):
    return f'/games/{quote(principal.scenario, safe="")}/{action}'


def principal_for(game_id=None):
    principal = current_principal("simulation")
    if "bank" not in principal.permissions:
        raise PermissionError("bank permission required")
    if game_id is not None and game_id != principal.scenario:
        raise PermissionError("Infrastructure belongs to another world")
    return principal


def verified(principal, asset_id, require_owner=False):
    canonical_uuid(asset_id)
    account = account_for(principal)
    result = bank_request("GET", f"/assets/{asset_id}/verify")
    asset = result.get("asset") or {}
    if (not result.get("authentic") or not result.get("spendable")
            or asset.get("kind") != "structure"
            or asset.get("metadata", {}).get("world") != principal.scenario
            or asset.get("metadata", {}).get("stateAuthority") != "arctic-simulation"
            or (require_owner and asset.get("ownerAccountId") != account["id"])):
        raise PermissionError("Active registered infrastructure in this world required")
    return asset


def owned(principal, asset_id):
    return verified(principal, asset_id, require_owner=True)


def configure_access(asset_id, interaction_radius_m=5, allowed_account_ids=()):
    principal = principal_for()
    asset = owned(principal, asset_id)
    if type(interaction_radius_m) not in (int, float) or not math.isfinite(interaction_radius_m) or not 0 < interaction_radius_m <= 100:
        raise ValueError("Interaction radius must be greater than zero and at most 100 metres")
    for account_id in allowed_account_ids:
        canonical_uuid(account_id)
    catalog = simulation_host.command("GET", path(principal, "infrastructure-catalog"))
    model = next(item for item in catalog["models"] if item["id"] == asset["assetType"])
    if model["upAxis"] != "Y":
        raise ValueError("Protection bounds require the authored Y-up model contract")
    policy = _model_policy(model, asset["ownerAccountId"], interaction_radius_m, allowed_account_ids)
    return simulation_host.command("POST", path(principal, "infrastructure-access"),
        {"operation": "configure", "id": asset_id, "accountId": asset["ownerAccountId"], "policy": policy})


def _model_policy(model, owner_account_id, interaction_radius_m=5, allowed_account_ids=()):
    if model["upAxis"] != "Y":
        raise ValueError("Protection bounds require the authored Y-up model contract")
    lower, upper = model["bounds"]["min"], model["bounds"]["max"]
    policy = {"version": 1, "ownerAccountId": owner_account_id,
              "allowedAccountIds": list(allowed_account_ids), "interactionRadiusM": interaction_radius_m,
              "bounds": {"minX": -upper[0], "maxX": -lower[0], "minY": lower[2], "maxY": upper[2], "minZ": lower[1], "maxZ": upper[1]}}
    return policy


def access_options(principal, asset_id):
    verified(principal, asset_id)
    account = account_for(principal)
    assets = bank_request("GET", f"/accounts/{account['id']}/portfolio")["ownedAssets"]
    vehicle_ids = {a["id"] for a in assets if a["kind"] in ("vehicle", "structure") and a["status"] == "active" and a.get("metadata", {}).get("world") == principal.scenario}
    snapshot = simulation_host.command("GET", path(principal, "snapshot"))
    subjects = [{"kind": "vehicle", "id": v["id"], "label": v["terrainAssetId"]}
                for v in snapshot["controlledVehicles"] if v["id"] in vehicle_ids and v["ownerAccountId"] == account["id"]]
    subjects.extend({"kind": "player", "id": p["id"], "label": "Player"}
                    for p in snapshot["players"] if p.get("ownerAccountId") == account["id"])
    return simulation_host.command("POST", path(principal, "infrastructure-access"),
        {"operation": "discover", "id": asset_id, "accountId": account["id"], "subjects": subjects})


def execute_component(principal, asset_id, action, expected_revision, subject_kind, subject_id):
    verified(principal, asset_id)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Use the current nonnegative component revision")
    canonical_uuid(subject_id)
    if subject_kind == "vehicle":
        from .vehicle_control import owned_asset
        owned_asset(principal, subject_id)
    elif subject_kind == "player":
        from .player_control import player_id
        if player_id(principal) != subject_id:
            raise PermissionError("The physical player must belong to this authenticated caller")
    else:
        raise ValueError("Select a physical player or owned vehicle; camera position is not accepted")
    account = account_for(principal)
    return simulation_host.command("POST", path(principal, "infrastructure-access"),
        {"operation": "command", "id": asset_id, "accountId": account["id"], "action": action,
         "expectedRevision": expected_revision, "subjectKind": subject_kind, "subjectId": subject_id})


def place(model_id, position, heading_deg, instance_key, game_id=None):
    return place_for(principal_for(game_id), model_id, position, heading_deg, instance_key)


def place_for(principal, model_id, position, heading_deg, instance_key):
    """Shared placement using an explicitly authenticated MCP or viewer principal."""
    if not {"simulation", "bank"} <= set(principal.permissions):
        raise PermissionError("Simulation and bank permissions required")
    if (not isinstance(instance_key, str) or not instance_key.strip() or len(instance_key) > 128):
        raise ValueError("A stable instance_key is required to make placement retries safe")
    if (set(position) != {"x", "y", "z"}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in position.values())
            or type(heading_deg) not in (int, float) or not math.isfinite(heading_deg)):
        raise ValueError("Finite ENU coordinates and heading required")
    catalog = simulation_host.command("GET", path(principal, "infrastructure-catalog"))
    if model_id not in {item["id"] for item in catalog["models"]}:
        raise ValueError("Unknown authored infrastructure model")
    account = account_for(principal)
    intent = {"modelId": model_id, "position": position, "headingDeg": heading_deg % 360}
    asset = bank_request("POST", "/assets/issue", {
        "kind": "structure", "assetType": model_id, "ownerAccountId": account["id"],
        "actorAccountId": account["id"], "sourceNamespace": "infrastructure:" + principal.scenario,
        "sourceId": instance_key, "metadata": {"world": principal.scenario,
            "stateAuthority": "arctic-simulation", "placementIntent": intent}})
    if asset.get("metadata", {}).get("placementIntent") != intent:
        raise ValueError("Instance key already identifies different placement terms")
    existing = simulation_host.command("GET", path(principal))["entities"]
    entity = next((item for item in existing if item["id"] == asset["id"]), None)
    if entity is not None:
        if entity["modelId"] != model_id:
            raise ValueError("Bank/simulation model conflict")
        return {"entity": entity, "bankAssetId": asset["id"], "alreadyPlaced": True}
    model = next(item for item in catalog["models"] if item["id"] == model_id)
    policy = _model_policy(model, asset["ownerAccountId"])
    placed = simulation_host.command("POST", path(principal), {**intent, "id": asset["id"], "accessPolicy": policy})
    return {**placed, "bankAssetId": asset["id"], "alreadyPlaced": False}


def control(asset_id, action, expected_revision, subject_kind, subject_id, game_id=None):
    return execute_component(principal_for(game_id), asset_id, action, expected_revision, subject_kind, subject_id)


def move(asset_id, position, heading_deg, game_id=None):
    principal = principal_for(game_id)
    owned(principal, asset_id)
    return simulation_host.command("PATCH", path(principal), {
        "id": asset_id, "position": position, "headingDeg": heading_deg})


def inspect(asset_id, game_id=None):
    principal = principal_for(game_id)
    asset = owned(principal, asset_id)
    entity = next((item for item in simulation_host.command("GET", path(principal))["entities"]
                   if item["id"] == asset_id), None)
    return {"asset": asset, "entity": entity, "placed": entity is not None}
