from atlantis_host_adapters.identity import current_principal
from atlantis_simulation.viewer import issue_viewer_access


@visible
def context() -> dict:
    """Inspect the current caller's approved scenario; never creates an identity."""
    principal = current_principal()
    return {'gameId': principal.scenario, 'hostGameId': principal.user_game_id,
            'permissions': sorted(principal.permissions)}


@visible
def viewer_access(ttl_seconds: int = 300) -> dict:
    """Issue short-lived read-only snapshot access. Keep token out of URLs/logs."""
    return issue_viewer_access(ttl_seconds)
