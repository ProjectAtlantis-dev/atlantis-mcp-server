"""Player actions derive ownership from authenticated host context, never arguments.

The bank executes validation and settlement transactionally. This gateway does not
mint assets, advance production time, move vehicles, or maintain a second ledger.
It can be installed by the Terrain host or a smaller habitat/demo host.
"""
import hashlib
import json
import math
import os
import re
from urllib.request import Request, urlopen
from uuid import UUID


def canonical_uuid(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Canonical lowercase UUID required")
    return value


def bank_request(method, path, body=None):
    base = os.environ.get("GAME_BANK_URL", "").rstrip("/")
    token = os.environ.get("GAME_BANK_AUTHORITY_TOKEN", "")
    if not base or not token:
        raise RuntimeError("Explicit GAME_BANK_URL and GAME_BANK_AUTHORITY_TOKEN are required")
    request = Request(base + path, method=method,
                      data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                      headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def account_for(principal, *, request=None):
    """Resolve the same bank account used by both economic and vehicle commands."""
    if not principal.external_user_id or not principal.permissions.intersection({"bank", "simulation", "world"}):
        raise PermissionError("An authorized bank-bound principal is required")
    return (request or bank_request)("POST", "/accounts/resolve", {
        "externalUserId": principal.external_user_id,
        "displayName": principal.caller, "accountType": "player"})


def request_key(principal, operation, value):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("idempotency_key must be a nonempty string of at most 200 characters")
    # Same retry from another chat uses the same key; another player cannot steal it.
    identity = json.dumps([principal.external_user_id, operation, value], separators=(",", ":"))
    return "terrain:" + hashlib.sha256(identity.encode()).hexdigest()


def execute(principal, operation, *, request=None, **args):
    if "bank" not in principal.permissions:
        raise PermissionError("bank permission required")
    fields = {
        "portfolio": set(), "verify": {"asset_id"}, "provenance": {"asset_id"},
        "transfer_credits": {"to_account_id", "amount", "currency", "idempotency_key"},
        "purchase_land": {"tile_ids", "idempotency_key"},
        "settle_quote": {"quote_id", "idempotency_key"},
    }
    if operation not in fields or set(args) != fields[operation]:
        raise ValueError("Unsupported economy operation or arguments")
    send = request or bank_request
    if operation in {"verify", "provenance"}:
        asset_id = canonical_uuid(args["asset_id"])
        return send("GET", f"/assets/{asset_id}/{operation}")
    if operation == "transfer_credits":
        canonical_uuid(args["to_account_id"])
        amount = args["amount"]
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0:
            raise ValueError("amount must be finite and positive")
        if not isinstance(args["currency"], str) or not re.fullmatch(r"[A-Z]{3,12}", args["currency"]):
            raise ValueError("currency must be an uppercase currency code")
    if operation == "purchase_land":
        tiles = args["tile_ids"]
        if (not isinstance(tiles, list) or not tiles or len(tiles) > 100
                or any(not isinstance(tile, str) or not re.fullmatch(r"12-\d+-\d+", tile) for tile in tiles)
                or len(set(tiles)) != len(tiles)):
            raise ValueError("Provide 1..100 distinct depth-12 terrain tile IDs")
    if operation == "settle_quote":
        canonical_uuid(args["quote_id"])
    key = None if operation == "portfolio" else request_key(principal, operation, args["idempotency_key"])
    account = account_for(principal, request=send)
    account_id = canonical_uuid(account["id"])
    if operation == "portfolio":
        return {"account": account,
                "portfolio": send("GET", f"/accounts/{account_id}/portfolio"),
                "balance": send("GET", f"/accounts/{account_id}/balance")}
    body = {"actorAccountId": account_id, "idempotencyKey": key}
    if operation == "transfer_credits":
        body.update(fromAccountId=account_id, toAccountId=args["to_account_id"],
                    amount=args["amount"], currency=args["currency"])
        path = "/credits/transfer"
    elif operation == "purchase_land":
        body.update(buyerAccountId=account_id, tileIds=args["tile_ids"])
        path = "/parcels/purchase"
    else:
        body.update(buyerAccountId=account_id)
        path = f'/warehouse/quotes/{args["quote_id"]}/settle'
    return send("POST", path, body)


def command(operation, **args):
    from atlantis_host_adapters.identity import current_principal
    return execute(current_principal("bank"), operation, **args)
