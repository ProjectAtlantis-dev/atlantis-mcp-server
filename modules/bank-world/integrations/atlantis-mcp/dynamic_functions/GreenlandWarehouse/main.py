@visible
def index() -> dict:
    """Separately configured warehouse service and quote interface."""
    return {'application': 'GreenlandWarehouse', 'requires': 'dedicated warehouse service credentials'}
