"""AI-readable entry point for the live control contracts."""
from pathlib import Path


@visible
def instructions(topic: str = 'overview') -> dict:
    """Read control documentation directly through MCP. Topics: overview, vehicles, objects, defense, infrastructure, placement, equipment, demo. Start here before orchestrating a demo; function docstrings/type hints supply exact callable schemas."""
    topics = {'vehicles': 'Vehicles', 'objects': 'Objects', 'defense': 'Defense',
              'infrastructure': 'Infrastructure', 'placement': 'Placement', 'equipment': 'Equipment', 'demo': 'Demo'}
    if topic == 'overview':
        return {'topics': topics, 'read': 'Terrain/instructions(topic)',
                'map': 'Terrain/Viewer/open_map or Terrain/Viewer/map_link; optional owned vehicle asset_id',
                'placement': 'Terrain/Placement/instructions; assets -> place_model or deploy_demo_site',
                'equipment': 'Terrain/Equipment/instructions; inspect -> set_controls -> actual pose; Viewer/open_workshop shares the same state',
                'discovery': 'Terrain/Objects/functions(asset_id) returns runnable actions, fields and bound state IDs',
                'defense': 'Terrain/Defense/instructions; observe -> explicit intercept -> events/outcome',
                'orchestration': 'AI/Lobster calls functions; the simulation owns motion and outcomes. No background AI is installed by these tools.',
                'rules': ['Use exact @/owner/remote paths, never wildcard routing.',
                          'Read observed state; command acceptance is not completion.',
                          'Use returned UUIDs, revisions and current action schemas.',
                          'Keep errors and blocked states visible; do not claim success or retry blindly.']}
    if topic not in topics:
        raise ValueError('Unknown topic; choose overview, vehicles, objects, defense, infrastructure, placement, equipment or demo')
    return {'topic': topic, 'documentation': (Path(__file__).parent / topics[topic] / 'README.md').read_text()}
