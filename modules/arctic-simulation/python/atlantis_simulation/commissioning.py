"""Explicit bank registration of existing Terrain instances, without moving them."""
import fcntl
import json
import os
from pathlib import Path
import tempfile

from atlantis_economy.gateway import account_for, bank_request
from atlantis_host_adapters.identity import current_principal
from .terrain_adapter import catalog


def register_vehicle(terrain_asset_id):
    import atlantis
    context = atlantis.get_context()
    if context is None or context.caller_sid not in atlantis.get_owner_usernames():
        raise PermissionError("Only the authenticated host owner may commission catalog assets")
    principal = current_principal("bank")
    keys = ("TERRAIN_ASSET_DB_PATH", "TERRAIN_DB_PATH", "ATLANTIS_TERRAIN_BINDINGS")
    if any(not os.environ.get(key) for key in keys):
        raise RuntimeError("Explicit Terrain databases and binding path required")
    config = {"assetDatabase": str(Path(os.environ[keys[0]]).resolve(strict=True)),
              "terrainDatabase": str(Path(os.environ[keys[1]]).resolve(strict=True)), "vehicles": {}}
    instance = next((item for item in catalog(config)["vehicle_instances"] if item["id"] == terrain_asset_id), None)
    if instance is None:
        raise ValueError("Enabled vehicle instance not found in the selected Terrain catalog")
    # Commission only what the current authoritative controller implements.
    if instance["definitionId"] not in {"patria-amv", "black-hornet", "v22-osprey"}:
        raise ValueError("This model's authoritative controller is not implemented; registration did not change its authority")
    path = Path(os.environ[keys[2]])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        policy = json.loads(path.read_text()) if path.exists() else {"version": 1, "worlds": {}}
        if policy.get("version") != 1:
            raise ValueError("Unsupported Terrain binding policy")
        world = policy["worlds"].setdefault(principal.scenario, config)
        if any(world[key] != config[key] for key in ("assetDatabase", "terrainDatabase")):
            raise ValueError("Existing world selects different Terrain databases")
        account = account_for(principal)
        asset = bank_request("POST", "/assets/issue", {
            "kind": "vehicle", "assetType": instance["definitionId"],
            "ownerAccountId": account["id"], "actorAccountId": account["id"],
            "sourceNamespace": "terrain:" + principal.scenario, "sourceId": terrain_asset_id,
            "metadata": {"stateAuthority": "arctic-simulation", "world": principal.scenario,
                         "terrainAssetId": terrain_asset_id, "initialCatalogPose": instance}})
        if asset.get("metadata", {}).get("world") != principal.scenario:
            raise ValueError("Existing bank registration has a conflicting world")
        world["vehicles"].setdefault(asset["id"], {"terrainAssetId": terrain_asset_id, "controlEnabled": False})
        # Bank commit first; source binding makes a crash before this projection
        # retryable without issuing another UUID. Original catalog stays intact.
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            json.dump(policy, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
            staged = stream.name
        os.replace(staged, path)
    return {"asset": asset, "world": principal.scenario, "terrainAssetId": terrain_asset_id,
            "attached": False, "next": "Inspect commissioning_plan, then attach; no position has changed"}


def activate_binding(world_id, asset_id):
    """Enable the legacy-pose write guard only after simulation attachment succeeds."""
    path = Path(os.environ["ATLANTIS_TERRAIN_BINDINGS"])
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        policy = json.loads(path.read_text())
        policy["worlds"][world_id]["vehicles"][asset_id]["controlEnabled"] = True
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            json.dump(policy, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
            staged = stream.name
        os.replace(staged, path)
