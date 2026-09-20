"""Owner-only persistent bank lifecycle, separate from simulation motion/tick."""
from atlantis_economy.host import bank_host
from atlantis_economy.provision import enroll_owner


@visible
def index() -> dict:
    """Owner-only persistent bank lifecycle and enrollment."""
    return {"module": "Terrain/Economy/Server", "visibility": "owner-only"}


@visible
def start() -> dict:
    """Start the configured local bank; restore existing accounts, UUIDs and ledger. No starter assets are minted."""
    return bank_host.start()


@visible
def status() -> dict:
    """Inspect this host's bank process without exposing its authority credential."""
    return bank_host.status()


@visible
def stop() -> dict:
    """Stop the bank cleanly. Persistent data remains; new economic/vehicle authorization requests fail closed."""
    return bank_host.stop()


@visible
def enroll(world: str) -> dict:
    """Bind the authenticated host owner to a persistent world and bank account. No cash or assets are granted. Existing identity policy is never overwritten."""
    return enroll_owner(world)
