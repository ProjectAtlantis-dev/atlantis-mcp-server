import unittest
from unittest.mock import patch
from atlantis_simulation.terrain_adapter import water_surface

class WaterRouteGridTests(unittest.TestCase):
    def test_coarse_grid_samples_actual_water_departure_next_to_shore(self):
        start={'x':.6,'y':.6}
        def shoreline(config,grid):
            return [grid['minX']+column*grid['stepM']>=.5
                    for row in range(grid['rows']) for column in range(grid['cols'])]
        with patch('atlantis_simulation.mission_terrain.water_grid',side_effect=shoreline):
            old=water_surface({}, {'lat':0,'lon':0},.5,radius=8,step=4,center={'x':4,'y':4})
            actual=water_surface({}, {'lat':0,'lon':0},.5,radius=8,step=4,center={'x':4,'y':4},grid_anchor=start)
        def at_start(grid):
            x=round((start['x']-grid['minX'])/grid['stepM']);y=round((start['y']-grid['minY'])/grid['stepM'])
            return grid['water'][y*grid['cols']+x]
        self.assertFalse(at_start(old))
        self.assertTrue(at_start(actual))
        self.assertEqual(start,{'x':.6,'y':.6})

    def test_boat_surface_uses_asset_profile_not_private_controller_snapshot_fields(self):
        import sqlite3
        from atlantis_simulation.mission_terrain import make_surface
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE assets(id,properties,enabled,type,min_x,max_x,min_y,max_y)')
        state={'authority':'server-boat-v1','definitionId':'test-boat',
               'navigationFrame':{'origin':{'lat':64,'lon':-51}},
               'position':{'x':0,'y':0,'z':.5}}
        definition={'realLengthM':12,'boat':{'maxSpeedMs':18,'yawDamping':1.05,'rudderTurnRadS2':.9}}
        with patch('atlantis_simulation.mission_terrain.configuration',return_value={'assetDatabase':'unused'}), \
             patch('atlantis_simulation.mission_terrain.water_surface',return_value={}), \
             patch('atlantis_simulation.mission_terrain.catalog',return_value={'vehicle_definitions':{'test-boat':definition}}), \
             patch('atlantis_simulation.mission_terrain.read_only',return_value=db):
            surface=make_surface('test',state)
        self.assertEqual(surface['waterNavigation']['hullRadiusM'],6)
        self.assertAlmostEqual(surface['waterNavigation']['preferredClearanceM'],42)
