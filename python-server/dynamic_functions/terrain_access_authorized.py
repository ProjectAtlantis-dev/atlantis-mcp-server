"""Identity gate for protected Terrain tools; object policy is checked at execution."""
import atlantis
from atlantis_host_adapters.identity import current_principal


@visible
def terrain_access_authorized(caller_sid: str) -> bool:
    """Require the authenticated caller's simulation/bank binding. Never accepts an identity supplied instead of the call context."""
    context = atlantis.get_context()
    if context is None or context.caller_sid != caller_sid:
        return False
    try:
        principal = current_principal('simulation')
    except PermissionError:
        return False
    return 'bank' in principal.permissions
