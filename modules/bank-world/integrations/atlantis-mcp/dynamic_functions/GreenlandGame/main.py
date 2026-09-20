@visible
def index() -> dict:
    """World logistics using the bank's canonical asset UUIDs."""
    return {'application': 'GreenlandGame', 'authority': 'logistics-world',
            'identity': 'same asset UUID as GreenlandBank; not scenario target IDs'}
