from atlantis_host_adapters.identity import bank_identity


def authenticated_game_identity() -> dict:
    """Use the shared immutable identity binding with world permission."""
    return bank_identity('world')
