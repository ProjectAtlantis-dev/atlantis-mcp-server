@visible
def index() -> dict:
    """Bank ownership, accounting and conserved resource lots; requires approved identity."""
    return {'application': 'GreenlandBank', 'authority': 'ownership-and-accounting',
            'deployment': 'test-only until production storage and identity provisioning are approved'}
