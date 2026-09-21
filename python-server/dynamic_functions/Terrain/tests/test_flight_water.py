import unittest
from unittest.mock import patch
from atlantis_simulation.terrain_adapter import flight_elevation_grid

class FlightWaterTests(unittest.TestCase):
    def test_only_verified_water_uses_explicit_water_level(self):
        grid={'heights':[None,None,12,None], 'rows':2,'cols':2}
        with patch('atlantis_simulation.terrain_adapter.elevation_grid',return_value=grid), patch('atlantis_simulation.mission_terrain.water_grid',return_value=[True,False,False,None]):
            result=flight_elevation_grid({}, {'lat':1,'lon':1},water_level_m=.5)
        self.assertEqual(result['heights'],[.5,None,12,None])
        self.assertEqual(result['waterLevelM'],.5)

    def test_water_level_must_be_explicit_and_finite(self):
        with self.assertRaises(ValueError):flight_elevation_grid({}, {},water_level_m=float('nan'))
        with patch('atlantis_simulation.terrain_adapter.elevation_grid',side_effect=ValueError('missing DEM')):
            with self.assertRaisesRegex(ValueError,'missing DEM'):flight_elevation_grid({}, {})
