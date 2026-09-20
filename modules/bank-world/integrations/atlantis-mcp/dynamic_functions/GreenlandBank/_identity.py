from atlantis_host_adapters.identity import bank_identity


def authenticated_game_identity() -> dict:
    """Use an explicit operator-approved x_user binding, never SID fallback."""
    return bank_identity('bank')
