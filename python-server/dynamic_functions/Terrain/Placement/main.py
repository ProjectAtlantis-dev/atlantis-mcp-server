"""Owner scene authoring: authored assets and fictional defense demo packages."""
from pathlib import Path
from atlantis_host_adapters.identity import current_principal
from dynamic_functions.Terrain.Placement import gateway


@visible
def index() -> dict:
    """Start with assets and instructions. Viewer Place assets uses these same functions."""
    return {'commands': ['assets', 'instructions', 'place_model', 'deploy_demo_site']}


@visible
def catalog() -> dict:
    """Compatibility name for assets(). Returns the original asset IDs without renaming them."""
    return gateway.assets(current_principal('simulation'))


@visible
def instructions() -> str:
    """Read placement recipes, coordinate semantics and implemented simulation limits."""
    return Path(__file__).with_name('README.md').read_text()


@visible
def place_model(model_id: str, instance_key: str, latitude: float, longitude: float,
                heading_deg: float = 0, height_offset_m: float = 0) -> dict:
    """Place an owned asset model using coordinates or the viewer picker. The authored base rests on the sampled terrain; optional offset is in metres. Reuse instance_key only for identical terms. Creates a bank UUID and automatically attaches supported ground movement under that UUID. A defense model alone is visual, not an active interceptor. Does not level terrain or validate the full footprint."""
    return gateway.place_model(current_principal('simulation'), model_id, instance_key,
                               latitude, longitude, heading_deg, height_offset_m)


@visible
def deploy_demo_site(site_id: str, layer_id: str, latitude: float, longitude: float) -> dict:
    """Place an existing fictional game package at selected coordinates: radar, one chosen layer and support entities. Uses existing game defaults and construction state; no real-world siting or performance model. A new site_id creates another package; identical retries reuse its bank UUIDs. Observe Terrain/Defense/observe for readiness. Does not change defense mode or authorize an interception. Each radar, launcher, command, resupply and recovery component has its own bank-owned UUID."""
    return gateway.deploy_demo_site(current_principal('simulation'), site_id, layer_id, latitude, longitude)


@visible
def assets() -> dict:
    """List the original repository's asset IDs, model dimensions and motion metadata without renaming them."""
    return gateway.assets(current_principal('simulation'))
