"""Terrain economy tools backed by the shared bank, not viewer-local balances.

Owner-only routing for now; the gateway additionally requires an approved
authenticated principal. No tool accepts a caller-selected source account.
"""
from atlantis_economy.gateway import command


@visible
def index() -> dict:
    """Owner-only Terrain bank, resource, land and settlement tools."""
    return {"module": "Terrain/Economy", "authority": "bank", "visibility": "owner-only"}


@visible
def portfolio() -> dict:
    """Read your bank account, money, land titles, vehicles, structures and resource lots."""
    return command("portfolio")


@visible
def verify(asset_id: str) -> dict:
    """Check whether a UUID is bank-issued and active. Verification does not grant control."""
    return command("verify", asset_id=asset_id)


@visible
def provenance(asset_id: str) -> dict:
    """Inspect bank-recorded issuance, ownership and resource-lot ancestry."""
    return command("provenance", asset_id=asset_id)


@visible
def transfer_credits(to_account_id: str, amount: float, idempotency_key: str,
                     currency: str = "GLC") -> dict:
    """Pay from YOUR authenticated account. Retry with the same key and identical terms."""
    return command("transfer_credits", to_account_id=to_account_id, amount=amount,
                   currency=currency, idempotency_key=idempotency_key)


@visible
def purchase_land(tile_ids: list[str], idempotency_key: str) -> dict:
    """Buy bank-listed depth-12 Terrain parcels; bank checks prices, funds and title ownership."""
    return command("purchase_land", tile_ids=tile_ids, idempotency_key=idempotency_key)


@visible
def settle_quote(quote_id: str, idempotency_key: str) -> dict:
    """Pay a warehouse quote from your account; bank atomically settles money and resource UUIDs."""
    return command("settle_quote", quote_id=quote_id, idempotency_key=idempotency_key)
