import json
import os
import uuid
from typing import Any, Dict, List

from ._client import client


def _configuration() -> Dict[str, Any]:
    try:
        prices = json.loads(os.getenv("WAREHOUSE_PRICES_JSON", "{}"))
    except json.JSONDecodeError as error:
        raise RuntimeError("WAREHOUSE_PRICES_JSON must be valid JSON") from error
    if not isinstance(prices, dict):
        raise RuntimeError("WAREHOUSE_PRICES_JSON must be an object keyed by asset type")
    commission_rate = float(os.getenv("WAREHOUSE_COMMISSION_RATE", "0.05"))
    if commission_rate < 0 or commission_rate > 1:
        raise RuntimeError("WAREHOUSE_COMMISSION_RATE must be between 0 and 1")
    return {
        "name": os.getenv("WAREHOUSE_NAME", "Greenland Warehouse"),
        "location": os.getenv("WAREHOUSE_LOCATION", "Greenland"),
        "prices": prices,
        "commissionRate": commission_rate,
    }


async def _service_identity() -> Dict[str, Any]:
    identity = await client.request("GET", "/services/me", service_auth=True)
    if identity.get("serviceType") != "warehouse":
        raise RuntimeError("WAREHOUSE_SERVICE_TOKEN is not a warehouse credential")
    return identity


@visible
async def warehouse_info() -> Dict[str, Any]:
    """Return this named warehouse's bank identity, location, prices, and commission."""
    identity = await _service_identity()
    return {"success": True, "warehouse": identity, "configuration": _configuration()}


@visible
async def warehouse_inventory() -> Dict[str, Any]:
    """Return authentic assets currently held in this warehouse's custody."""
    identity = await _service_identity()
    portfolio = await client.request("GET", f"/accounts/{identity['ownerAccountId']}/portfolio")
    return {
        "success": True,
        "warehouse": identity,
        "assetsInCustody": portfolio.get("assetsInCustody", []),
    }


@visible
async def request_warehouse_purchase_quote(
    asset_ids: List[str],
    expires_in_seconds: int = 300,
) -> Dict[str, Any]:
    """Price deposited UUID assets and register an expiring, bank-verifiable quote."""
    if not asset_ids or len(asset_ids) > 100 or len(set(asset_ids)) != len(asset_ids):
        raise ValueError("asset_ids must contain 1-100 distinct UUIDs")
    identity = await _service_identity()
    config = _configuration()
    verified_assets = []
    seller_account_id = None
    gross_amount = 0.0

    for asset_id in asset_ids:
        verification = await client.request("GET", f"/assets/{asset_id}/verify")
        if not verification.get("authentic") or not verification.get("spendable"):
            raise ValueError(f"Asset is not spendable: {asset_id}")
        asset = verification["asset"]
        if asset.get("custodianAccountId") != identity["ownerAccountId"] or asset.get("status") != "stored":
            raise ValueError(f"Asset is not stored at this warehouse: {asset_id}")
        if seller_account_id is None:
            seller_account_id = asset["ownerAccountId"]
        elif seller_account_id != asset["ownerAccountId"]:
            raise ValueError("One quote cannot combine assets from different sellers")
        if asset["assetType"] not in config["prices"]:
            raise ValueError(f"No warehouse price configured for {asset['assetType']}")
        quantity = asset["quantity"] if asset["quantity"] is not None else 1
        gross_amount += float(config["prices"][asset["assetType"]]) * float(quantity)
        verified_assets.append(asset)

    gross_amount = round(gross_amount, 6)
    commission_amount = round(gross_amount * config["commissionRate"], 6)
    quote = await client.request(
        "POST",
        "/warehouse/quotes",
        body={
            "clientQuoteId": str(uuid.uuid4()),
            "sellerAccountId": seller_account_id,
            "assetIds": asset_ids,
            "grossAmount": gross_amount,
            "commissionAmount": commission_amount,
            "currency": "GLC",
            "expiresInSeconds": expires_in_seconds,
            "terms": {
                "warehouseName": config["name"],
                "location": config["location"],
                "pricingVersion": os.getenv("WAREHOUSE_PRICING_VERSION", "1"),
            },
        },
        service_auth=True,
    )
    return {"success": True, "quote": quote, "assets": verified_assets}


@visible
async def request_warehouse_partial_purchase_quote(
    resource_asset_id: str,
    quantity: float,
    expires_in_seconds: int = 300,
) -> Dict[str, Any]:
    """Quote part of one divisible resource lot; settlement mints sold and remainder UUID lots."""
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    identity = await _service_identity()
    config = _configuration()
    verification = await client.request("GET", f"/assets/{resource_asset_id}/verify")
    if not verification.get("authentic") or not verification.get("spendable"):
        raise ValueError(f"Asset is not spendable: {resource_asset_id}")
    asset = verification["asset"]
    if asset.get("kind") != "resource_lot":
        raise ValueError("Partial quotes require a resource_lot UUID")
    if asset.get("custodianAccountId") != identity["ownerAccountId"] or asset.get("status") != "stored":
        raise ValueError("Resource lot is not stored at this warehouse")
    available = float(asset["quantity"])
    requested = float(quantity)
    if requested > available:
        raise ValueError(f"Requested quantity exceeds available lot quantity ({available} {asset['unit']})")
    unit_price = config["prices"].get(asset["assetType"])
    if unit_price is None:
        raise ValueError(f"No warehouse price configured for {asset['assetType']}")
    gross_amount = round(float(unit_price) * requested, 6)
    commission_amount = round(gross_amount * config["commissionRate"], 6)
    quote = await client.request(
        "POST",
        "/warehouse/quotes",
        body={
            "clientQuoteId": str(uuid.uuid4()),
            "sellerAccountId": asset["ownerAccountId"],
            "assetIds": [resource_asset_id],
            "resourceLines": [{"assetId": resource_asset_id, "quantity": requested}],
            "grossAmount": gross_amount,
            "commissionAmount": commission_amount,
            "currency": "GLC",
            "expiresInSeconds": expires_in_seconds,
            "terms": {
                "warehouseName": config["name"],
                "location": config["location"],
                "pricingVersion": os.getenv("WAREHOUSE_PRICING_VERSION", "1"),
                "partialResourceSale": True,
            },
        },
        service_auth=True,
    )
    return {"success": True, "quote": quote, "asset": asset}


@visible
async def get_warehouse_purchase_quote(quote_id: str) -> Dict[str, Any]:
    """Return the bank-recorded terms and current status of a warehouse quote."""
    return await client.request("GET", f"/warehouse/quotes/{quote_id}")
