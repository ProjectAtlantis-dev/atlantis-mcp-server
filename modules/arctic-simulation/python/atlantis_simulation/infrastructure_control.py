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


def owned(principal, asset_id):
    canonical_uuid(asset_id)
    account = account_for(principal)
    verified = bank_request("GET", f"/assets/{asset_id}/verify")
    asset = verified.get("asset") or {}
    if (not verified.get("authentic") or not verified.get("spendable")
            or asset.get("ownerAccountId") != account["id"]
            or asset.get("kind") != "structure"
            or asset.get("metadata", {}).get("world") != principal.scenario
            or asset.get("metadata", {}).get("stateAuthority") != "arctic-simulation"):
        raise PermissionError("Active bank-owned infrastructure in this world required")
    return asset


def place(model_id, position, heading_deg, instance_key, game_id=None):
    principal = principal_for(game_id)
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
    placed = simulation_host.command("POST", path(principal), {**intent, "id": asset["id"]})
    return {**placed, "bankAssetId": asset["id"], "alreadyPlaced": False}


def control(asset_id, action, expected_revision, game_id=None):
    principal = principal_for(game_id)
    owned(principal, asset_id)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Use the current nonnegative component revision")
    return simulation_host.command("POST", path(principal, "component-command"), {
        "id": asset_id, "action": action, "expectedRevision": expected_revision})


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
